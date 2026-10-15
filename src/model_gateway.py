"""Configurable model-provider gateway with safe failover metadata."""
from __future__ import annotations

from dataclasses import dataclass, replace
import os
import time


SUPPORTED_PROVIDERS = {"auto", "gemini", "groq", "openai", "openai_compatible"}
MODEL_PURPOSES = {"router", "planner", "agent", "final_response", "default"}
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4.1-mini",
}


class ModelGatewayConfigurationError(RuntimeError):
    """Raised before a provider request when gateway configuration is invalid."""

    def __init__(self, message: str, *, code: str = "MODEL_CONFIGURATION_ERROR"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None = None
    slot: int = 1
    thinking_mode: str | None = None

    def public_payload(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "slot": self.slot,
            "custom_base_url": bool(self.base_url),
            **({"thinking_mode": self.thinking_mode} if self.thinking_mode else {}),
        }


@dataclass(frozen=True)
class ModelGatewayConfig:
    providers: tuple[ProviderConfig, ...]
    purpose: str
    max_attempts: int = 2
    request_timeout: float = 10.0
    circuit_failure_threshold: int = 2
    circuit_cooldown_seconds: float = 30.0


@dataclass(frozen=True)
class ModelCandidate:
    config: ProviderConfig
    client: object


_gateway_state = {"index": 0, "last_call": None, "last_success": None, "last_error": None, "health": {}}


def _value(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None


def _csv(name: str) -> list[str]:
    return [item.strip().lower().replace("-", "_") for item in (_value(name) or "").split(",") if item.strip()]


def _positive_int(name: str, default: int, maximum: int) -> int:
    raw = _value(name)
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise ModelGatewayConfigurationError(f"模型配置 {name} 必须是整数") from exc
    if value < 1 or value > maximum:
        raise ModelGatewayConfigurationError(f"模型配置 {name} 必须在 1 到 {maximum} 之间")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = _value(name)
    try:
        value = float(raw) if raw else default
    except ValueError as exc:
        raise ModelGatewayConfigurationError(f"模型配置 {name} 必须是数字") from exc
    if value <= 0:
        raise ModelGatewayConfigurationError(f"模型配置 {name} 必须大于 0")
    return value


def _provider_configs(provider: str, *, primary: bool, model_override: str | None = None) -> list[ProviderConfig]:
    unified_key = _value("LLM_API_KEY") if primary else None
    unified_model = model_override or (_value("LLM_MODEL") if primary else None)
    unified_base_url = _value("LLM_BASE_URL") if primary else None

    if provider == "gemini":
        pooled = _value("GEMINI_API_KEYS")
        keys = [key.strip() for key in pooled.split(",") if key.strip()] if pooled else []
        if unified_key:
            keys = [unified_key]
        elif not keys and _value("GEMINI_API_KEY"):
            keys = [_value("GEMINI_API_KEY")]
        model = unified_model or _value("GEMINI_MODEL") or DEFAULT_MODELS["gemini"]
        return [ProviderConfig(provider, model, key, slot=index + 1) for index, key in enumerate(keys)]

    if provider == "groq":
        key = unified_key or _value("GROQ_API_KEY")
        model = unified_model or _value("GROQ_MODEL") or DEFAULT_MODELS["groq"]
        return [ProviderConfig(provider, model, key)] if key else []

    if provider == "openai":
        key = unified_key or _value("OPENAI_API_KEY")
        model = unified_model or _value("OPENAI_MODEL") or DEFAULT_MODELS["openai"]
        base_url = unified_base_url or _value("OPENAI_BASE_URL")
        return [ProviderConfig(provider, model, key, base_url=base_url)] if key else []

    if provider == "openai_compatible":
        thinking_mode = _value("OPENAI_COMPATIBLE_THINKING_MODE")
        if thinking_mode not in {None, "enabled", "disabled"}:
            raise ModelGatewayConfigurationError("OPENAI_COMPATIBLE_THINKING_MODE 必须是 enabled 或 disabled")
        key = unified_key or _value("OPENAI_COMPATIBLE_API_KEY")
        model = unified_model or _value("OPENAI_COMPATIBLE_MODEL")
        base_url = unified_base_url or _value("OPENAI_COMPATIBLE_BASE_URL")
        configured = any((key, model, base_url))
        if configured and not all((key, model, base_url)):
            raise ModelGatewayConfigurationError(
                "OpenAI-compatible 接入必须同时配置 API Key、模型名称和 Base URL"
            )
        return [ProviderConfig(provider, model, key, base_url=base_url, thinking_mode=thinking_mode)] if configured else []

    raise ModelGatewayConfigurationError(f"不支持的模型 Provider：{provider}")


def load_gateway_config(*, purpose: str = "default", model: str | None = None) -> ModelGatewayConfig:
    normalized_purpose = purpose.strip().lower().replace("-", "_")
    if normalized_purpose not in MODEL_PURPOSES:
        raise ModelGatewayConfigurationError(f"不支持的模型用途：{purpose}")
    provider = (_value("LLM_PROVIDER") or "auto").lower().replace("-", "_")
    if provider not in SUPPORTED_PROVIDERS:
        raise ModelGatewayConfigurationError(f"不支持的 LLM_PROVIDER：{provider}")
    purpose_model = model or _value(f"LLM_{normalized_purpose.upper()}_MODEL")
    providers: list[ProviderConfig] = []
    if provider == "auto":
        for candidate in ("gemini", "groq", "openai", "openai_compatible"):
            providers.extend(_provider_configs(candidate, primary=False))
        if purpose_model and providers:
            primary_provider = providers[0].provider
            providers = [replace(item, model=purpose_model) if item.provider == primary_provider else item for item in providers]
    else:
        providers.extend(_provider_configs(provider, primary=True, model_override=purpose_model))
        for fallback in _csv("LLM_FALLBACK_PROVIDERS"):
            if fallback == "auto" or fallback == provider or fallback not in SUPPORTED_PROVIDERS:
                if fallback not in {provider, "auto"}:
                    raise ModelGatewayConfigurationError(f"不支持的备用模型 Provider：{fallback}")
                continue
            providers.extend(_provider_configs(fallback, primary=False))
    if not providers:
        raise ModelGatewayConfigurationError(
            "未配置可用模型：请配置 Gemini、Groq、OpenAI 或 OpenAI-compatible 模型 API",
            code="MODEL_UNAVAILABLE",
        )
    return ModelGatewayConfig(
        providers=tuple(providers),
        purpose=normalized_purpose,
        max_attempts=_positive_int("LLM_MAX_ATTEMPTS", 2, 8),
        request_timeout=_positive_float("LLM_REQUEST_TIMEOUT", 10.0),
        circuit_failure_threshold=_positive_int("LLM_CIRCUIT_FAILURE_THRESHOLD", 2, 20),
        circuit_cooldown_seconds=_positive_float("LLM_CIRCUIT_COOLDOWN_SECONDS", 30.0),
    )


def build_provider_client(config: ProviderConfig, *, temperature: float, timeout: float):
    if config.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.model,
            api_key=config.api_key,
            temperature=temperature,
            retries=0,
            request_timeout=timeout,
        )
    if config.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=config.model,
            api_key=config.api_key,
            temperature=temperature,
            max_retries=0,
            timeout=timeout,
        )
    if config.provider in {"openai", "openai_compatible"}:
        from langchain_openai import ChatOpenAI

        options = {"extra_body": {"thinking": {"type": config.thinking_mode}}} if config.thinking_mode else {}
        return ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=temperature,
            max_retries=0,
            timeout=timeout,
            **options,
        )
    raise ModelGatewayConfigurationError(f"不支持的模型 Provider：{config.provider}")


def _failure_kind(exc: Exception) -> str:
    text = str(exc).lower()
    if "429" in text or "rate limit" in text or "resource_exhausted" in text:
        return "RATE_LIMIT"
    if isinstance(exc, TimeoutError) or "timeout" in text or "timed out" in text:
        return "TIMEOUT"
    if any(token in text for token in ("503", "502", "service unavailable", "connection error", "connection refused")):
        return "UNAVAILABLE"
    if any(token in text for token in (
        "401", "403", "unauthorized", "forbidden", "invalid api key", "api key not valid", "authentication",
    )):
        return "AUTH_ERROR"
    return "PROVIDER_ERROR"


def _can_failover(exc: Exception) -> bool:
    return _failure_kind(exc) in {"RATE_LIMIT", "TIMEOUT", "UNAVAILABLE"}


class RotatingChatModel:
    """LangChain-compatible model pool; the runtime owns bounded retries."""

    def __init__(self, models: list, state: dict | None = None, *, max_attempts: int = 2,
                 circuit_failure_threshold: int = 2, circuit_cooldown_seconds: float = 30.0):
        self._candidates = [
            item if isinstance(item, ModelCandidate) else ModelCandidate(ProviderConfig("unknown", "unknown", ""), item)
            for item in models
        ]
        if not self._candidates:
            raise ModelGatewayConfigurationError("No LLM configured：模型候选列表为空")
        self._state = state if state is not None else _gateway_state
        self._state.setdefault("health", {})
        self.max_attempts = max_attempts
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_cooldown_seconds = circuit_cooldown_seconds

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    @property
    def last_call(self) -> dict:
        return dict(self._state.get("last_call") or {})

    def _clone(self, clients: list) -> "RotatingChatModel":
        candidates = [ModelCandidate(old.config, client) for old, client in zip(self._candidates, clients)]
        return RotatingChatModel(
            candidates,
            self._state,
            max_attempts=self.max_attempts,
            circuit_failure_threshold=self.circuit_failure_threshold,
            circuit_cooldown_seconds=self.circuit_cooldown_seconds,
        )

    def bind_tools(self, tools):
        return self._clone([candidate.client.bind_tools(tools) for candidate in self._candidates])

    def bind(self, **kwargs):
        return self._clone([candidate.client.bind(**kwargs) for candidate in self._candidates])

    @staticmethod
    def _health_key(candidate: ModelCandidate) -> str:
        config = candidate.config
        return f"{config.provider}:{config.model}:{config.slot}"

    def _health(self, candidate: ModelCandidate) -> dict:
        return self._state["health"].setdefault(
            self._health_key(candidate),
            {"failure_count": 0, "open_until": 0.0, "last_failure_kind": None},
        )

    def _select_candidate(self) -> tuple[int, ModelCandidate]:
        now = time.time()
        start = self._state.get("index", 0) % len(self._candidates)
        for offset in range(len(self._candidates)):
            index = (start + offset) % len(self._candidates)
            candidate = self._candidates[index]
            if self._health(candidate).get("open_until", 0.0) <= now:
                self._state["index"] = index
                return index, candidate
        retry_after = min(
            max(0.0, self._health(candidate).get("open_until", now) - now)
            for candidate in self._candidates
        )
        call = {
            "status": "SKIPPED",
            "failure_kind": "CIRCUIT_OPEN",
            "retry_after_seconds": round(retry_after, 3),
        }
        self._state["last_call"] = call
        self._state["last_error"] = call
        raise ModelGatewayConfigurationError(
            f"所有模型 Provider 暂时熔断，请在约 {max(1, round(retry_after))} 秒后重试",
            code="MODEL_UNAVAILABLE",
        )

    def invoke_once(self, messages):
        index, candidate = self._select_candidate()
        started = time.monotonic()
        descriptor = {"provider": candidate.config.provider, "model": candidate.config.model, "slot": candidate.config.slot}
        try:
            result = candidate.client.invoke(messages)
        except Exception as exc:
            failure_kind = _failure_kind(exc)
            health = self._health(candidate)
            if _can_failover(exc):
                health["failure_count"] = health.get("failure_count", 0) + 1
                health["last_failure_kind"] = failure_kind
                if health["failure_count"] >= self.circuit_failure_threshold:
                    health["open_until"] = time.time() + self.circuit_cooldown_seconds
            call = {**descriptor, "status": "FAILED", "failure_kind": failure_kind,
                    "duration_ms": round((time.monotonic() - started) * 1000, 2)}
            if health.get("open_until", 0.0) > time.time():
                call["circuit_status"] = "OPEN"
            self._state["last_call"] = call
            self._state["last_error"] = call
            if _can_failover(exc):
                self._state["index"] = (index + 1) % len(self._candidates)
            raise
        health = self._health(candidate)
        health.update(failure_count=0, open_until=0.0, last_failure_kind=None)
        call = {**descriptor, "status": "SUCCEEDED",
                "duration_ms": round((time.monotonic() - started) * 1000, 2)}
        self._state.update(index=index, last_call=call, last_success=call, last_error=None)
        return result

    def invoke(self, messages):
        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                return self.invoke_once(messages)
            except Exception as exc:
                last_error = exc
                if not _can_failover(exc):
                    raise
        raise last_error


def create_chat_model(*, purpose: str = "default", model: str | None = None, temperature: float = 0.0) -> RotatingChatModel:
    config = load_gateway_config(purpose=purpose, model=model)
    candidates = [
        ModelCandidate(item, build_provider_client(item, temperature=temperature, timeout=config.request_timeout))
        for item in config.providers
    ]
    return RotatingChatModel(
        candidates,
        _gateway_state,
        max_attempts=config.max_attempts,
        circuit_failure_threshold=config.circuit_failure_threshold,
        circuit_cooldown_seconds=config.circuit_cooldown_seconds,
    )


def _provider_status(config: ProviderConfig) -> dict:
    payload = config.public_payload()
    health = _gateway_state.get("health", {}).get(
        f"{config.provider}:{config.model}:{config.slot}", {}
    )
    retry_after = max(0.0, health.get("open_until", 0.0) - time.time())
    payload.update(
        health="OPEN" if retry_after > 0 else ("RECOVERING" if health.get("failure_count") else "READY"),
        failure_count=health.get("failure_count", 0),
        retry_after_seconds=round(retry_after, 3),
        last_failure_kind=health.get("last_failure_kind"),
    )
    return payload


def get_model_status() -> dict:
    try:
        config = load_gateway_config()
    except ModelGatewayConfigurationError as exc:
        return {"ready": False, "configured": False, "message": str(exc), "providers": [],
                "error_code": exc.code, "active": _gateway_state.get("last_success"),
                "last_error": _gateway_state.get("last_error")}
    providers = [_provider_status(provider) for provider in config.providers]
    ready = any(provider["health"] != "OPEN" for provider in providers)
    return {
        "ready": ready,
        "configured": True,
        "live_verified": _gateway_state.get("last_success") is not None,
        "message": (
            "所有模型 Provider 暂时熔断，请稍后重试" if not ready else
            "最近一次模型调用成功" if _gateway_state.get("last_success") else
            "模型配置已加载，尚未执行实时请求"
        ),
        "error_code": None if ready else "MODEL_UNAVAILABLE",
        "purpose": config.purpose,
        "max_attempts": config.max_attempts,
        "request_timeout": config.request_timeout,
        "circuit_failure_threshold": config.circuit_failure_threshold,
        "circuit_cooldown_seconds": config.circuit_cooldown_seconds,
        "providers": providers,
        "active": _gateway_state.get("last_success"),
        "last_error": _gateway_state.get("last_error"),
    }


def reset_gateway_state() -> None:
    """Reset process-local provider rotation; intended for tests and diagnostics."""

    _gateway_state.update(index=0, last_call=None, last_success=None, last_error=None, health={})
