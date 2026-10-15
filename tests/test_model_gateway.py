import pytest
from langchain_core.messages import AIMessage

from src import model_gateway
from src.model_gateway import (
    ModelCandidate,
    ModelGatewayConfigurationError,
    ProviderConfig,
    RotatingChatModel,
)


_MODEL_ENV = {
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "LLM_FALLBACK_PROVIDERS",
    "LLM_MAX_ATTEMPTS",
    "LLM_REQUEST_TIMEOUT",
    "LLM_CIRCUIT_FAILURE_THRESHOLD",
    "LLM_CIRCUIT_COOLDOWN_SECONDS",
    "LLM_ROUTER_MODEL",
    "LLM_PLANNER_MODEL",
    "LLM_AGENT_MODEL",
    "LLM_FINAL_RESPONSE_MODEL",
    "GEMINI_API_KEY",
    "GEMINI_API_KEYS",
    "GEMINI_MODEL",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_BASE_URL",
    "OPENAI_COMPATIBLE_API_KEY",
    "OPENAI_COMPATIBLE_MODEL",
    "OPENAI_COMPATIBLE_BASE_URL",
    "OPENAI_COMPATIBLE_THINKING_MODE",
}
_MODEL_ENV.update({'OPENAI_COMPATIBLE_REASONING_EFFORT', 'LLM_GOAL_MODEL'})
_MODEL_ENV.update(f'OPENAI_COMPATIBLE_{purpose}_{option}'
                  for purpose in ('DEFAULT', 'ROUTER', 'GOAL', 'PLANNER', 'AGENT', 'FINAL_RESPONSE')
                  for option in ('THINKING_MODE', 'REASONING_EFFORT'))


@pytest.fixture(autouse=True)
def clean_model_environment(monkeypatch):
    for name in _MODEL_ENV:
        monkeypatch.delenv(name, raising=False)
    model_gateway.reset_gateway_state()


def test_auto_keeps_legacy_provider_order_and_hides_keys(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "gemini-one, gemini-two")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")

    config = model_gateway.load_gateway_config()

    assert [item.provider for item in config.providers] == ["gemini", "gemini", "groq"]
    assert [item.slot for item in config.providers] == [1, 2, 1]
    payload = [item.public_payload() for item in config.providers]
    assert "gemini-one" not in repr(payload)
    assert "groq-secret" not in repr(payload)


def test_openai_compatible_requires_key_model_and_base_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "compatible-secret")

    with pytest.raises(ModelGatewayConfigurationError, match="必须同时配置"):
        model_gateway.load_gateway_config()

    monkeypatch.setenv("LLM_MODEL", "vendor-chat")
    monkeypatch.setenv("LLM_BASE_URL", "https://models.example.test/v1")
    config = model_gateway.load_gateway_config()

    assert config.providers == (
        ProviderConfig(
            "openai_compatible",
            "vendor-chat",
            "compatible-secret",
            base_url="https://models.example.test/v1",
        ),
    )


def test_unconfigured_status_has_stable_code_and_chinese_message():
    status = model_gateway.get_model_status()

    assert status["ready"] is False
    assert status["error_code"] == "MODEL_UNAVAILABLE"
    assert status["message"].startswith("未配置可用模型")


def test_compatible_thinking_mode_is_explicit_and_scoped(monkeypatch):
    monkeypatch.setenv('OPENAI_COMPATIBLE_THINKING_MODE', 'disabled')
    monkeypatch.setenv('LLM_PROVIDER', 'openai_compatible')
    monkeypatch.setenv('LLM_API_KEY', 'test-secret')
    monkeypatch.setenv('LLM_MODEL', 'deepseek-flash')
    monkeypatch.setenv('LLM_BASE_URL', 'https://api.deepseek.com')
    compatible = model_gateway.load_gateway_config().providers[0]
    client = model_gateway.build_provider_client(compatible, temperature=0, timeout=10)
    assert client.extra_body == {'thinking': {'type': 'disabled'}}
    assert compatible.public_payload()['thinking_mode'] == 'disabled'
    monkeypatch.setenv('LLM_PROVIDER', 'openai')
    official = model_gateway.load_gateway_config().providers[0]
    assert official.thinking_mode is None
    assert not model_gateway.build_provider_client(official, temperature=0, timeout=10).extra_body
    monkeypatch.setenv('LLM_PROVIDER', 'openai_compatible')
    monkeypatch.setenv('OPENAI_COMPATIBLE_THINKING_MODE', 'invalid')
    with pytest.raises(ModelGatewayConfigurationError, match='THINKING_MODE'):
        model_gateway.load_gateway_config()


def test_goal_reasoning_is_configurable_without_changing_execution_model(monkeypatch):
    monkeypatch.setenv('LLM_PROVIDER', 'openai_compatible')
    monkeypatch.setenv('LLM_API_KEY', 'test-secret')
    monkeypatch.setenv('LLM_MODEL', 'deepseek-flash')
    monkeypatch.setenv('LLM_BASE_URL', 'https://api.deepseek.com')
    monkeypatch.setenv('OPENAI_COMPATIBLE_THINKING_MODE', 'disabled')
    monkeypatch.setenv('OPENAI_COMPATIBLE_GOAL_THINKING_MODE', 'enabled')
    monkeypatch.setenv('OPENAI_COMPATIBLE_GOAL_REASONING_EFFORT', 'low')
    config = model_gateway.load_gateway_config(purpose='goal').providers[0]
    assert config.model == 'deepseek-flash' and config.thinking_mode == 'enabled' and config.reasoning_effort == 'low'
    planner = model_gateway.load_gateway_config(purpose='planner').providers[0]
    assert planner.thinking_mode == 'disabled' and planner.reasoning_effort is None
    client = model_gateway.build_provider_client(config, temperature=0, timeout=15)
    assert client.reasoning_effort == 'low'
    monkeypatch.setenv('OPENAI_COMPATIBLE_GOAL_REASONING_EFFORT', 'invalid')
    with pytest.raises(ModelGatewayConfigurationError, match='推理强度'):
        model_gateway.load_gateway_config(purpose='goal')


def test_json_mode_preserves_provider_and_does_not_change_unsupported_clients():
    class JsonClient:
        def __init__(self, options=None):
            self.options = options or {}
        def bind(self, **options):
            return JsonClient({**self.options, **options})
        def invoke(self, messages):
            assert self.options['response_format'] == {'type': 'json_object'}
            return AIMessage(content='{"action":"clarify","reason":"需要补充信息"}')
    first = JsonClient()
    unsupported = object()
    model = RotatingChatModel([
        ModelCandidate(ProviderConfig('openai_compatible', 'flash', 'test'), first),
        ModelCandidate(ProviderConfig('unknown', 'other', 'test'), unsupported)])
    encoded = model.json_mode()
    assert encoded._candidates[1].client is unsupported
    assert not first.options
    assert encoded.invoke('json').content.startswith('{')
    assert encoded.last_call['model'] == 'flash'


def test_purpose_override_only_changes_primary_auto_provider(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret")
    monkeypatch.setenv("LLM_PLANNER_MODEL", "gemini-planner")

    config = model_gateway.load_gateway_config(purpose="planner")

    assert config.providers[0].model == "gemini-planner"
    assert config.providers[1].model == model_gateway.DEFAULT_MODELS["groq"]


class _Provider:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def bind(self, **_kwargs):
        return self

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def test_rotating_model_fails_over_and_records_sanitized_provider_metadata():
    limited = _Provider(error=RuntimeError("429 rate limit"))
    healthy = _Provider(result=AIMessage(content="完成"))
    state = {"index": 0}
    model = RotatingChatModel(
        [
            ModelCandidate(ProviderConfig("gemini", "gemini-test", "secret-one"), limited),
            ModelCandidate(ProviderConfig("groq", "groq-test", "secret-two"), healthy),
        ],
        state,
        max_attempts=2,
    )

    response = model.invoke("你好")

    assert response.content == "完成"
    assert limited.calls == healthy.calls == 1
    assert model.last_call["provider"] == "groq"
    assert "secret" not in repr(model.last_call)


def test_authentication_failure_does_not_fall_through_to_another_provider():
    denied = _Provider(error=RuntimeError("401 invalid api key"))
    fallback = _Provider(result=AIMessage(content="不应调用"))
    model = RotatingChatModel(
        [
            ModelCandidate(ProviderConfig("openai", "gpt-test", "bad-secret"), denied),
            ModelCandidate(ProviderConfig("groq", "groq-test", "other-secret"), fallback),
        ],
        {"index": 0},
        max_attempts=2,
    )

    with pytest.raises(RuntimeError, match="invalid api key"):
        model.invoke("你好")

    assert denied.calls == 1
    assert fallback.calls == 0


def test_circuit_breaker_skips_unhealthy_providers_until_cooldown(monkeypatch):
    now = {"value": 1000.0}
    monkeypatch.setattr(model_gateway.time, "time", lambda: now["value"])
    first = _Provider(error=TimeoutError("provider timeout"))
    second = _Provider(error=RuntimeError("503 service unavailable"))
    state = {"index": 0}
    model = RotatingChatModel(
        [
            ModelCandidate(ProviderConfig("gemini", "gemini-test", "secret-one"), first),
            ModelCandidate(ProviderConfig("groq", "groq-test", "secret-two"), second),
        ],
        state,
        max_attempts=2,
        circuit_failure_threshold=1,
        circuit_cooldown_seconds=30,
    )

    with pytest.raises(RuntimeError, match="service unavailable"):
        model.invoke("你好")
    assert first.calls == second.calls == 1

    with pytest.raises(ModelGatewayConfigurationError) as open_error:
        model.invoke_once("你好")
    assert open_error.value.code == "MODEL_UNAVAILABLE"
    assert first.calls == second.calls == 1
    assert state["last_call"]["failure_kind"] == "CIRCUIT_OPEN"

    now["value"] += 31
    with pytest.raises(TimeoutError, match="provider timeout"):
        model.invoke_once("你好")
    assert first.calls == 2


def test_model_status_exposes_sanitized_provider_health(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("LLM_CIRCUIT_FAILURE_THRESHOLD", "3")
    monkeypatch.setenv("LLM_CIRCUIT_COOLDOWN_SECONDS", "45")

    status = model_gateway.get_model_status()

    assert status["circuit_failure_threshold"] == 3
    assert status["circuit_cooldown_seconds"] == 45
    assert status["providers"][0]["health"] == "READY"
    assert status["providers"][0]["failure_count"] == 0
    assert "gemini-secret" not in repr(status)

    key = "gemini:gemini-2.5-flash:1"
    model_gateway._gateway_state["health"][key] = {
        "failure_count": 3,
        "open_until": model_gateway.time.time() + 30,
        "last_failure_kind": "TIMEOUT",
    }
    unavailable = model_gateway.get_model_status()
    assert unavailable["ready"] is False
    assert unavailable["configured"] is True
    assert unavailable["error_code"] == "MODEL_UNAVAILABLE"
    assert unavailable["providers"][0]["health"] == "OPEN"


def test_create_chat_model_builds_clients_without_network(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_API_KEY", "openai-secret")
    monkeypatch.setenv("LLM_MODEL", "gpt-test")
    built = []

    def fake_factory(config, *, temperature, timeout):
        built.append((config.public_payload(), temperature, timeout))
        return _Provider(result=AIMessage(content="完成"))

    monkeypatch.setattr(model_gateway, "build_provider_client", fake_factory)
    model = model_gateway.create_chat_model(purpose="router", temperature=0.2)

    assert model.invoke("你好").content == "完成"
    assert built == [(
        {"provider": "openai", "model": "gpt-test", "slot": 1, "custom_base_url": False},
        0.2,
        10.0,
    )]
