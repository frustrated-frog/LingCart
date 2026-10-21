"""Shared bounded runtime, tool policy and durable replay primitives."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import hashlib
import inspect
import importlib
import json
import math
import queue
import threading
import time
import uuid
from typing import get_type_hints

from langchain_core.messages import AIMessage
from pydantic import ConfigDict, ValidationError, create_model

from src.errors import AgentError, ErrorCode, classify_exception
from src.runtime_state import restore_runtime_snapshot


@dataclass(frozen=True)
class BudgetLimits:
    max_model_calls: int = 8
    max_tool_calls: int = 12
    # 运行时目前用 UTF-8 字节数做保守预留（不是供应商 token 数）；该上限允许一次正常调用
    # 加一次协议修复，真实调用次数仍由 max_model_calls 独立限制。
    max_prompt_tokens: int = 48000
    # 目录理解、独立审查与至多一次修正共享预算；上限不是调用配额，实际调用仍按事件计量。
    max_completion_tokens: int = 6000
    # 推理型 Provider 的审查请求常超过 10 秒。较长单次时限可避免超时后重复计费，
    # 总时限仍限制最坏路径，且所有副作用继续由工具 deadline 与账本独立保护。
    max_seconds: float = 60
    call_timeout: float = 20
    max_steps: int = 8
    max_depth: int = 4
    read_attempts: int = 2
    # 直接语义协议包含受限目录查询 AST 与版本化概念菜单；16k 字符仍保持小上下文，
    # 同时为有来源的跨轮引用保留空间。
    max_context_chars: int = 16000
    max_tool_observation_chars: int = 4000
    max_history_message_chars: int = 6000

    @classmethod
    def adaptive(cls):
        """逐步决策、证据映射审查与复合完成共用的有界预算。"""
        return cls(max_model_calls=16, max_tool_calls=16, max_prompt_tokens=256000,
                   max_completion_tokens=12000, max_seconds=90, call_timeout=45,
                   max_steps=16, max_context_chars=32000)

    def __post_init__(self):
        for key, value in self.__dict__.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid budget: {key}")
            if key not in {"max_seconds", "call_timeout"} and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"Budget must be an integer: {key}")
        if (self.call_timeout <= 0 or self.max_seconds <= 0 or self.read_attempts < 1 or
                self.max_context_chars < 1 or self.max_tool_observation_chars < 1 or
                self.max_history_message_chars < 1):
            raise ValueError("Time and context limits must be positive")


_current_runtime = ContextVar("agent_runtime", default=None)
_workers = threading.BoundedSemaphore(8)


def current_runtime():
    return _current_runtime.get()


@contextmanager
def runtime_scope(runtime):
    token = _current_runtime.set(runtime)
    try:
        yield runtime
    finally:
        _current_runtime.reset(token)


def bounded_call(fn, timeout, source):
    """Deadlines for read-only tools/model requests. Timed-out workers are not killed."""
    started = time.monotonic()
    if not _workers.acquire(timeout=max(0, timeout)):
        raise AgentError(f"{source.upper()}_TIMEOUT", "执行队列已满", retryable=True)
    result_queue = queue.Queue(maxsize=1)

    def work():
        try:
            result_queue.put((True, fn()))
        except BaseException as exc:
            result_queue.put((False, exc))
        finally:
            _workers.release()

    threading.Thread(target=work, daemon=True).start()
    try:
        succeeded, result = result_queue.get(timeout=max(.001, timeout - (time.monotonic() - started)))
    except queue.Empty:
        raise AgentError(f"{source.upper()}_TIMEOUT", "调用超过执行时限", retryable=True) from None
    if not succeeded:
        raise result
    return result


class RunRuntime:
    def __init__(self, run_id, limits=None, *, store=None, token=None, snapshot=None, events=None, approval=None):
        self.run_id = run_id
        self.limits = limits or BudgetLimits()
        self.store, self.token = store, token
        self.snapshot = restore_runtime_snapshot(snapshot, run_id) if snapshot else {}
        self.models = dict(self.snapshot.get("models", {}))
        self.counters = {"model_calls": 0, "tool_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                         "elapsed_seconds": 0.0, **self.snapshot.get("budget", {})}
        self.events = list(events or [])
        self.sequence = max([event.get("sequence", 0) for event in self.events] + [0])
        self.started = time.monotonic()
        self.prior_elapsed = self.counters["elapsed_seconds"]
        self.model_cursor = 0
        self.approval = approval
        self._tool_results = {}
        self._tool_outcomes = {}
        self._tool_state_lock = threading.RLock()
        self._tool_serial_gate = threading.Lock()
        self.phase = self.snapshot.get("phase", "INTAKE")
        self.defer_terminal = False
        self.pending_terminal = None
        from src.context_engine import LifecycleHooks
        self.hooks = LifecycleHooks()
        self.context_engine = None
        self.harness_profile = None
        self.mechanisms = {}
        saved_hook_error = self.snapshot.get('hook_error')
        self.hook_error = AgentError(saved_hook_error['code'], saved_hook_error['message'], retryable=False) if saved_hook_error else None

    @property
    def remaining_seconds(self):
        return self.limits.max_seconds - self.prior_elapsed - (time.monotonic() - self.started)

    def stop_reason(self, operation=None, prompt_tokens=0):
        # Stable priority if several budgets are exhausted at the same boundary.
        if self.remaining_seconds <= 0:
            return "TIME_BUDGET_EXHAUSTED"
        if operation == "model" and self.counters["model_calls"] >= self.limits.max_model_calls:
            return "MODEL_ROUND_BUDGET_EXHAUSTED"
        if operation == "tool" and self.counters["tool_calls"] >= self.limits.max_tool_calls:
            return str(ErrorCode.TOOL_BUDGET_EXHAUSTED)
        if operation == "model" and self.counters["prompt_tokens"] + prompt_tokens > self.limits.max_prompt_tokens:
            return "PROMPT_TOKEN_BUDGET_EXHAUSTED"
        if operation == "model" and self.counters["completion_tokens"] >= self.limits.max_completion_tokens:
            return "COMPLETION_TOKEN_BUDGET_EXHAUSTED"
        return None

    def require(self, operation=None, prompt_tokens=0):
        reason = self.stop_reason(operation, prompt_tokens)
        if reason:
            raise AgentError(reason, "运行预算已耗尽，已保留此前的执行结果", retryable=False)

    def checkpoint(self, **updates):
        self.snapshot.update(updates)
        self.counters["elapsed_seconds"] = self.prior_elapsed + time.monotonic() - self.started
        self.snapshot.update(phase=self.phase, current_node=self.phase, status="RUNNING",
                             models=self.models, budget=self.counters, sequence=self.sequence)
        if self.store:
            self.store.save_checkpoint(self.run_id, self.token, self.snapshot)

    def record(self, event_type, **data):
        if event_type == "terminal" and self.defer_terminal:
            self.pending_terminal = data
            return {"run_id": self.run_id, "type": "terminal", **data}
        self.sequence += 1
        event = {"run_id": self.run_id, "sequence": self.sequence, "timestamp": datetime.now(UTC).isoformat(),
                 "type": event_type, **data}
        self.events.append(event)
        self.counters["elapsed_seconds"] = self.prior_elapsed + time.monotonic() - self.started
        self.snapshot.update(phase=self.phase, current_node=self.phase, status="RUNNING",
                             models=self.models, budget=self.counters, sequence=self.sequence)
        if self.store:
            self.store.save_checkpoint(self.run_id, self.token, self.snapshot, event)
        return event

    def model_call(self, model, messages, *, purpose, agent=None, allow_tools=True, schemas=None):
        from src.session_control import apply_controls
        apply_controls(self)
        self.hooks.emit(self, 'before_model', {'purpose': purpose, 'agent': agent})
        if self.context_engine:
            messages = self.context_engine.model_messages(messages)
        from src.context_budget import compact_context
        from src.llm import extract_usage
        self.model_cursor += 1
        key = str(self.model_cursor)
        effective_messages, compaction = compact_context(
            messages,
            max_chars=self.limits.max_context_chars,
            max_tool_observation_chars=self.limits.max_tool_observation_chars,
            max_history_message_chars=self.limits.max_history_message_chars,
        )
        canonical = effective_messages if isinstance(effective_messages, str) else [
            {"type": getattr(message, "type", "ai" if hasattr(message, "tool_calls") else "human"),
             "content": getattr(message, "content", str(message)), "tool_calls": getattr(message, "tool_calls", []),
             "tool_call_id": getattr(message, "tool_call_id", None)} for message in effective_messages]
        encoded = json.dumps([purpose, agent, canonical, schemas], ensure_ascii=False, default=str)
        # This is deliberately conservative. Actual usage remains separately
        # recorded; providers without usage metadata do not report fake zeroes.
        estimate = len(encoded.encode("utf-8")) + 512
        cached = self.models.get(key)
        if cached:
            if cached["input"] != encoded:
                raise AgentError("CHECKPOINT_CONFLICT", "恢复时的模型输入与已保存步骤不同", retryable=False)
            self.record("model_reused", purpose=purpose, agent=agent)
            return AIMessage(content=cached["content"], tool_calls=cached["tool_calls"], usage_metadata=cached.get("usage_metadata"))
        if compaction["changed"]:
            self.record("context_compacted", purpose=purpose, agent=agent, **{
                key: value for key, value in compaction.items() if key != "changed"
            })
        self.require("model", estimate)
        self.counters["model_calls"] += 1
        self.counters["prompt_tokens"] += estimate
        self.phase = "MODEL_CALL"
        self.record("model_started", purpose=purpose, agent=agent, prompt_budget_reserved=estimate)
        completion_limit = self.limits.max_completion_tokens - self.counters["completion_tokens"]
        if hasattr(model, "bind"):
            model = model.bind(max_tokens=completion_limit)
        started = time.monotonic()
        max_attempts = getattr(model, "max_attempts", 2)
        for attempt in range(max_attempts):
            apply_controls(self)
            try:
                invoke = getattr(model, "invoke_once", model.invoke)
                response = bounded_call(
                    lambda: invoke(effective_messages),
                    min(self.limits.call_timeout, self.remaining_seconds),
                    "model",
                )
                break
            except Exception as exc:
                error = classify_exception(exc, source="model")
                provider = getattr(model, "last_call", {})
                self.record("error", source="model", purpose=purpose, agent=agent, code=error.code,
                            message=str(error), retryable=error.retryable,
                            provider=provider.get("provider"), model=provider.get("model"),
                            provider_failure=provider.get("failure_kind"))
                if not error.retryable or attempt + 1 == max_attempts:
                    raise error from exc
                self.require("model", estimate)
                self.counters["model_calls"] += 1
                self.counters["prompt_tokens"] += estimate
                self.record("model_retry", purpose=purpose, agent=agent, attempt=attempt + 2)
        usage = extract_usage(response)
        if usage.get("input_tokens") is not None:
            self.counters["prompt_tokens"] += usage["input_tokens"] - estimate
        output_tokens = usage.get("output_tokens")
        if output_tokens is None:
            output_tokens = len(str(response.content).encode("utf-8")) + len(json.dumps(getattr(response, "tool_calls", [])))
        self.counters["completion_tokens"] += output_tokens
        self.models[key] = {"input": encoded, "content": response.content,
                            "tool_calls": getattr(response, "tool_calls", []), "usage_metadata": getattr(response, "usage_metadata", None)}
        provider = getattr(model, "last_call", {})
        self.record("model_call", purpose=purpose, agent=agent, status="SUCCEEDED", usage=usage,
                    provider=provider.get("provider"), model=provider.get("model"), provider_slot=provider.get("slot"),
                    budget_usage_source="provider" if all(usage.get(k) is not None for k in ("input_tokens", "output_tokens")) else "conservative_estimate",
                    duration_ms=round((time.monotonic() - started) * 1000, 2), allow_tools=allow_tools)
        return response


class TraceRecorder:
    def __init__(self, run_id):
        self.run_id = run_id
        self.events = []

    def record(self, event_type, **data):
        context = current_runtime()
        if context and context.run_id == self.run_id:
            event = context.record(event_type, **data)
        else:
            event = {"run_id": self.run_id, "sequence": len(self.events) + 1,
                     "timestamp": datetime.now(UTC).isoformat(), "type": event_type, **data}
        self.events.append(event)
        return event


class ApprovalRequired(AgentError):
    def __init__(self, pending):
        super().__init__("APPROVAL_REQUIRED", "状态变更需要用户确认", retryable=False)
        self.pending = pending


class ToolRiskLevel(StrEnum):
    READ_ONLY = "READ_ONLY"
    GUARDED = "GUARDED"
    HIGH = "HIGH"


class ToolEffect(StrEnum):
    READ = "READ"
    WRITE = "WRITE"


class ReplayPolicy(StrEnum):
    REPLAY_SAFE = "REPLAY_SAFE"
    AT_MOST_ONCE = "AT_MOST_ONCE"
    REQUIRES_APPROVAL = "REQUIRES_APPROVAL"


class ConcurrencyPolicy(StrEnum):
    PARALLEL_SAFE = "PARALLEL_SAFE"
    SERIAL_PER_RUN = "SERIAL_PER_RUN"


@dataclass(frozen=True)
class ToolSpec:
    function: object
    schema: object = None
    side_effect: bool = False
    transactional: bool = False
    risk_level: ToolRiskLevel | None = None
    effect: ToolEffect | None = None
    replay_policy: ReplayPolicy | None = None
    concurrency_policy: ConcurrencyPolicy | None = None
    approval_action: str | None = None

    def __post_init__(self):
        # side_effect 保留旧调用兼容性；新策略字段一旦明确指定就不能被静默覆盖。
        if type(self.side_effect) is not bool or type(self.transactional) is not bool:
            raise AgentError("INVALID_TOOL_POLICY", "工具效果与事务标记必须是布尔值", retryable=False)
        defaults = {
            "effect": ToolEffect.WRITE if self.side_effect else ToolEffect.READ,
            "risk_level": ToolRiskLevel.HIGH if self.side_effect else ToolRiskLevel.READ_ONLY,
            "replay_policy": ReplayPolicy.REQUIRES_APPROVAL if self.side_effect else ReplayPolicy.REPLAY_SAFE,
            "concurrency_policy": ConcurrencyPolicy.SERIAL_PER_RUN if self.side_effect else ConcurrencyPolicy.PARALLEL_SAFE,
        }
        kinds = {"effect": ToolEffect, "risk_level": ToolRiskLevel, "replay_policy": ReplayPolicy,
                 "concurrency_policy": ConcurrencyPolicy}
        try:
            for key, default in defaults.items():
                value = getattr(self, key)
                object.__setattr__(self, key, kinds[key](default if value is None else value))
        except ValueError as exc:
            raise AgentError("INVALID_TOOL_POLICY", "工具策略包含未知标识", retryable=False) from exc
        invalid = (
            (self.side_effect and self.effect != ToolEffect.WRITE)
            or (self.transactional and (self.effect != ToolEffect.WRITE or self.replay_policy != ReplayPolicy.REQUIRES_APPROVAL))
            or (self.effect == ToolEffect.WRITE and (
                self.risk_level == ToolRiskLevel.READ_ONLY
                or self.replay_policy == ReplayPolicy.REPLAY_SAFE
                or self.concurrency_policy != ConcurrencyPolicy.SERIAL_PER_RUN
            ))
            or (self.effect == ToolEffect.READ and self.risk_level == ToolRiskLevel.HIGH)
            or (self.approval_action is not None and self.replay_policy != ReplayPolicy.REQUIRES_APPROVAL)
        )
        if invalid:
            raise AgentError("INVALID_TOOL_POLICY", "工具效果、风险、重放与并发策略不一致", retryable=False)
        object.__setattr__(self, "side_effect", self.effect == ToolEffect.WRITE)

    @property
    def name(self):
        return getattr(self.function, "name", None) or self.function.__name__

    @property
    def args_schema(self):
        return self.schema or _argument_schema(self.name, self.function)

    def policy_payload(self) -> dict:
        return {
            "risk_level": self.risk_level.value,
            "effect": self.effect.value,
            "replay_policy": self.replay_policy.value,
            "concurrency_policy": self.concurrency_policy.value,
            "transactional": self.transactional,
        }


def _argument_schema(name, fn):
    schema = getattr(fn, "args_schema", None)
    function = getattr(fn, "func", None) or fn
    if schema is not None:
        return schema
    fields = {}
    try:
        # Preserve Annotated constraints so provider/tool schemas expose the
        # same numeric and string boundaries enforced before backend calls.
        hints = get_type_hints(function, include_extras=True)
    except (NameError, TypeError):
        hints = {}
    for key, parameter in inspect.signature(function).parameters.items():
        if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            continue
        annotation = hints.get(key, parameter.annotation)
        if annotation is inspect.Parameter.empty:
            from typing import Any
            annotation = Any
        fields[key] = (annotation, ... if parameter.default is inspect.Parameter.empty else parameter.default)
    return create_model(f"{name}Arguments", __config__=ConfigDict(extra="forbid", strict=True), **fields)


# 注册表列出每个生产入口；同名新函数不会自动获得只读权限。
_PRODUCTION_TOOL_POLICIES = {
    "src.tools": {
        "read_clock": {}, "calculate": {},
        "get_order_status": {}, "get_order_details": {}, "track_shipment": {},
        "get_payment_status": {}, "check_refund_eligibility": {}, "answer_faq": {},
        "search_products": {}, "get_catalog_facets": {}, "discover_products": {}, "compare_products": {},
        "process_payment": {"side_effect": True, "transactional": True, "approval_action": "PROCESS_PAYMENT"},
        "initiate_refund": {"side_effect": True, "transactional": True, "approval_action": "INITIATE_REFUND"},
        "escalate_to_human": {"effect": ToolEffect.WRITE, "risk_level": ToolRiskLevel.GUARDED,
                              "replay_policy": ReplayPolicy.AT_MOST_ONCE,
                              "concurrency_policy": ConcurrencyPolicy.SERIAL_PER_RUN},
    },
    "src.agents.order_agent": {"get_order_status": {}, "get_order_details": {}, "track_shipment": {}},
    "src.agents.payment_agent": {
        "get_payment_status": {},
        "process_payment": {"side_effect": True, "transactional": True, "approval_action": "PROCESS_PAYMENT"},
    },
    "src.agents.refund_agent": {
        "get_order_status": {}, "check_refund_eligibility": {},
        "initiate_refund": {"side_effect": True, "transactional": True, "approval_action": "INITIATE_REFUND"},
    },
    "src.agents.faq_agent": {"answer_faq": {}},
    "src.catalog_query": {"query_catalog": {}},
    "src.constraint_planning": {"solve_constraint_plan": {}},
}


def tool_spec(name, fn):
    if isinstance(fn, ToolSpec):
        function = getattr(fn.function, "func", None) or fn.function
        return ToolSpec(function, fn.schema or _argument_schema(name, fn.function), fn.side_effect,
                        fn.transactional, fn.risk_level, fn.effect, fn.replay_policy, fn.concurrency_policy,
                        fn.approval_action)
    function = getattr(fn, "func", None) or fn
    # 依据注册对象身份匹配，不能仅凭工具名或伪造 __module__ 取得权限。
    for module_name, definitions in _PRODUCTION_TOOL_POLICIES.items():
        if name not in definitions:
            continue
        module = importlib.import_module(module_name)
        registered = getattr(module, name)
        registered_function = getattr(registered, "func", None) or registered
        if function is registered_function:
            return ToolSpec(function, _argument_schema(name, fn), **definitions[name])
    raise AgentError("TOOL_POLICY_MISSING", f"工具 {name} 尚未显式注册执行策略", retryable=False)


class ToolExecutor:
    def __init__(self, registry, recorder, max_calls=8, *, approved_action=None, observation_scope=None):
        self.recorder, self.max_calls = recorder, max_calls
        self.observation_scope = observation_scope
        self.registry = {}
        for name, fn in registry.items():
            try:
                self.registry[name] = tool_spec(name, fn)
            except AgentError as error:
                self._fail(error, name, {})
        self.call_count = 0
        self._results, self._outcomes = {}, {}
        self._state_lock, self._serial_gate = threading.RLock(), threading.Lock()
        self.approval = approved_action

    @staticmethod
    def fingerprint(tool_name, args):
        canonical = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(f"{tool_name.strip()}\n{canonical}".encode("utf-8")).hexdigest()

    def _fail(self, error, name, args):
        self.recorder.record("error", tool=name, args=args, code=error.code, message=str(error), retryable=error.retryable,
                             action_fingerprint=self.fingerprint(name, args))
        raise error

    def _approval_matches(self, spec, name, args, approval):
        if not isinstance(approval, dict):
            return False
        if approval.get("action_fingerprint"):
            return approval["action_fingerprint"] == self.fingerprint(name, args)
        if approval.get("tool") == name and isinstance(approval.get("args"), dict):
            return approval["args"] == args
        # 仅历史内置审批允许扁平业务字段；新工具必须提供完整参数或指纹。
        legacy_action = {"process_payment": "PROCESS_PAYMENT", "initiate_refund": "INITIATE_REFUND"}.get(name)
        if legacy_action:
            from src import tools
            if spec.function is not getattr(tools, name):
                return False
        expected_action = spec.approval_action or legacy_action
        if not legacy_action or not expected_action or approval.get("type") != expected_action or approval.get("order_id") != args.get("order_id"):
            return False
        return all(approval.get(key) == value
                   for key, value in args.items() if key != "order_id")

    def execute(self, name, args):
        from src.session_control import apply_controls
        active = current_runtime()
        if active and active.run_id == self.recorder.run_id:
            apply_controls(active)
        if name not in self.registry:
            self._fail(AgentError(ErrorCode.UNKNOWN_TOOL, f"工具 {name} 不在允许列表中", retryable=False), name, args)
        spec = self.registry[name]
        policy = spec.policy_payload()
        try:
            if not isinstance(args, dict):
                raise TypeError("工具参数必须是对象")
            if spec.schema and set(args) - set(spec.schema.model_fields):
                raise TypeError("包含未知工具参数")
            # Preserve only caller-supplied values. Function defaults remain the
            # function's responsibility; otherwise adding an optional schema
            # field changes fingerprints and breaks compatible wrappers.
            args = spec.schema.model_validate(args, strict=True).model_dump(exclude_unset=True) if spec.schema else args
        except (ValidationError, TypeError) as exc:
            self._fail(AgentError(ErrorCode.INVALID_TOOL_ARGUMENTS, str(exc), retryable=False), name, args)
        context = current_runtime()
        context = context if context and context.run_id == self.recorder.run_id else None
        if context:
            context.hooks.emit(context, 'before_tool', {'tool': name, 'args': args, 'policy': policy})
        approval = context.approval if context and context.store else (context.approval if context else None) or self.approval
        if spec.replay_policy == ReplayPolicy.REQUIRES_APPROVAL and not self._approval_matches(spec, name, args, approval):
            action = spec.approval_action or {"process_payment": "PROCESS_PAYMENT", "initiate_refund": "INITIATE_REFUND"}.get(name)
            pending = {**args, "type": action or name.upper(), "tool": name, "args": args,
                       "action_fingerprint": self.fingerprint(name, args), "requires_approval": True}
            if name == "process_payment":
                from src import tools
                self.registry.setdefault("get_payment_status", tool_spec("get_payment_status", tools.get_payment_status))
                payment, _ = self.execute("get_payment_status", {"order_id": args["order_id"]})
                if payment.get("error"):
                    self._fail(AgentError("BUSINESS_DATA_NOT_FOUND", payment["error"], retryable=False), name, args)
                pending["amount"] = payment.get("amount")
            self.recorder.record("approval_required", tool=name, args=args, policy=policy,
                                 action_fingerprint=self.fingerprint(name, args))
            raise ApprovalRequired(pending)
        fingerprint = self.fingerprint(name, args)
        scope = self.observation_scope if spec.effect == ToolEffect.READ and spec.replay_policy == ReplayPolicy.REPLAY_SAFE else None
        scope_sha256 = self.fingerprint("observation_scope", {"value": scope}) if scope is not None else None
        scope_evidence = {"observation_scope_sha256": scope_sha256} if scope_sha256 else {}
        observation_key = fingerprint if scope is None else self.fingerprint(name, {'args': args, 'observation_scope': scope})
        key = context.store.ledger_key(context.run_id, name, args, approval if spec.effect == ToolEffect.WRITE else None) if context and context.store else None
        if key and scope is not None:
            key = hashlib.sha256(f'{key}:{scope}'.encode()).hexdigest()
        transactional = bool(key and spec.transactional)
        state_lock = context._tool_state_lock if context else self._state_lock
        results = context._tool_results if context else self._results
        outcomes = context._tool_outcomes if context else self._outcomes
        serial_gate = context._tool_serial_gate if context else self._serial_gate
        with state_lock:
            saved = context.store.ledger(key, name, args) if key else None
            previous = saved["status"] if saved else outcomes.get(observation_key)
            unsafe_replay = previous in {"UNKNOWN", "RUNNING"} and spec.replay_policy != ReplayPolicy.REPLAY_SAFE
            if unsafe_replay and not (transactional and saved and previous == "RUNNING"):
                self._fail(AgentError("UNCERTAIN_TOOL_OUTCOME", "外部操作结果不确定，禁止自动重放", retryable=False), name, args)
            if previous == "FAILED" and spec.replay_policy == ReplayPolicy.AT_MOST_ONCE:
                self._fail(AgentError("TOOL_REPLAY_FORBIDDEN", "工具策略禁止失败后再次执行", retryable=False), name, args)
            cached = saved["result"] if saved and saved["status"] == "SUCCEEDED" else results.get(observation_key)
            if (saved and saved["status"] == "SUCCEEDED") or observation_key in results:
                unfinished = None
                if context and saved:
                    finished_ids = {e.get('execution_id') for e in context.events if e['type'] in {'tool_result', 'error'}}
                    unfinished = next((e.get('execution_id') for e in reversed(context.events)
                        if e['type'] == 'tool_call' and e.get('action_fingerprint') == fingerprint
                        and e.get('ledger_key', key) == key
                        and e.get('execution_id') not in finished_ids), None)
                evidence_events = context.events if context else self.recorder.events
                source_result = next((e for e in reversed(evidence_events)
                    if e.get('type') == 'tool_result' and not e.get('reused')
                    and e.get('tool') == name and e.get('args') == args
                    and e.get('action_fingerprint') == fingerprint
                    and e.get('observation_scope_sha256') == scope_sha256
                    and e.get('result') == cached and e.get('execution_id')), None)
                reuse_source = ({
                    'source_execution_id': source_result['execution_id'],
                    'source_sequence': source_result.get('sequence'),
                } if source_result else {})
                event = self.recorder.record("tool_result", tool=name, args=args, result=cached,
                    status="BUSINESS_ERROR" if isinstance(cached, dict) and cached.get("error") else "REUSED",
                    reused=True, duration_ms=0.0, policy=policy, action_fingerprint=fingerprint,
                    execution_id=unfinished, source_ledger=key if saved else None,
                    **scope_evidence, **reuse_source)
                if context:
                    if context.context_engine:
                        context.context_engine.observe(name, args, cached, reference=f'{context.run_id}:{event["sequence"]}')
                        context.checkpoint(context=context.context_engine.payload())
                    context.hooks.emit(context, 'after_tool', {'tool': name, 'args': args, 'result': cached, 'status': event['status']})
                return cached, event
            outcomes[observation_key] = "RUNNING"
        attempts = context.limits.read_attempts if context and spec.replay_policy == ReplayPolicy.REPLAY_SAFE else 1
        for attempt in range(attempts):
            if context:
                apply_controls(context)
            try:
                with state_lock:
                    if self.call_count >= self.max_calls:
                        raise AgentError(ErrorCode.TOOL_BUDGET_EXHAUSTED, "工具调用预算已耗尽", retryable=False)
                    if context:
                        context.require("tool")
                        context.counters["tool_calls"] += 1
                        context.phase = "TOOL_CALL"
                    self.call_count += 1
            except AgentError as error:
                outcomes.pop(observation_key, None)
                self._fail(error, name, args)
            execution_id = uuid.uuid4().hex
            self.recorder.record("tool_call", tool=name, args=args, call_index=self.call_count, attempt=attempt + 1,
                                 policy=policy, action_fingerprint=fingerprint, execution_id=execution_id,
                                 ledger_key=key, **scope_evidence)
            started = time.monotonic()
            deadline = started + min(context.limits.call_timeout, context.remaining_seconds) if context else None

            def invoke(deadline=deadline):
                acquired = False
                if spec.concurrency_policy == ConcurrencyPolicy.SERIAL_PER_RUN:
                    acquired = serial_gate.acquire(timeout=max(0, deadline - time.monotonic())) if deadline else serial_gate.acquire()
                    if not acquired:
                        raise AgentError("TOOL_TIMEOUT", "等待串行工具执行超时", retryable=True)
                try:
                    if deadline and time.monotonic() >= deadline:
                        raise AgentError("TOOL_TIMEOUT", "工具执行时限已到，未开始业务操作", retryable=False)
                    if transactional:
                        return context.store.execute_write(context.run_id, context.token, key, name, args, approval, lambda: spec.function(**args))
                    return spec.function(**args)
                finally:
                    if acquired:
                        serial_gate.release()
            try:
                if key and not transactional:
                    context.store.save_ledger(context.run_id, context.token, key, name, args, "RUNNING")
                # 本地事务保留原线程数据库会话；外部调用的锁由工作线程持有到真正结束。
                result = invoke() if not context or transactional else bounded_call(invoke, max(.001, deadline - time.monotonic()), "tool")
                if key and not transactional:
                    context.store.save_ledger(context.run_id, context.token, key, name, args, "SUCCEEDED", result)
            except Exception as exc:
                error = classify_exception(exc, source="tool")
                outcome = "UNKNOWN" if spec.effect == ToolEffect.WRITE and not transactional else "FAILED"
                with state_lock:
                    outcomes[observation_key] = outcome
                if key:
                    context.store.save_ledger(context.run_id, context.token, key, name, args, outcome,
                                              {"error": str(error), "code": error.code})
                self.recorder.record("error", tool=name, args=args, code=error.code, message=str(error), retryable=error.retryable,
                                     duration_ms=round((time.monotonic() - started) * 1000, 2), policy=policy,
                                     action_fingerprint=fingerprint, execution_id=execution_id, **scope_evidence)
                if spec.replay_policy == ReplayPolicy.REPLAY_SAFE and error.retryable and attempt + 1 < attempts and self.call_count < self.max_calls:
                    self.recorder.record("tool_retry", tool=name, attempt=attempt + 2)
                    continue
                raise error from exc
            with state_lock:
                results[observation_key], outcomes[observation_key] = result, "SUCCEEDED"
            status = "BUSINESS_ERROR" if isinstance(result, dict) and result.get("error") else (
                "BUSINESS_REJECTED" if isinstance(result, dict) and result.get("success") is False else "SUCCEEDED")
            event = self.recorder.record("tool_result", tool=name, args=args, result=result, status=status,
                                         duration_ms=round((time.monotonic() - started) * 1000, 2), policy=policy,
                                         action_fingerprint=fingerprint, execution_id=execution_id, **scope_evidence)
            if context:
                if context.context_engine:
                    context.context_engine.observe(name, args, result, reference=f'{context.run_id}:{event["sequence"]}')
                    context.checkpoint(context=context.context_engine.payload())
                context.hooks.emit(context, 'after_tool', {'tool': name, 'args': args, 'result': result, 'status': status})
            return result, event
