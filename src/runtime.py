"""Shared bounded runtime, tool policy and durable replay primitives."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
import inspect
import json
import math
import queue
import threading
import time
from typing import get_type_hints

from langchain_core.messages import AIMessage
from pydantic import ConfigDict, ValidationError, create_model

from src.errors import AgentError, ErrorCode, classify_exception


@dataclass(frozen=True)
class BudgetLimits:
    max_model_calls: int = 8
    max_tool_calls: int = 12
    max_prompt_tokens: int = 16000
    max_completion_tokens: int = 3000
    max_seconds: float = 30
    call_timeout: float = 10
    max_steps: int = 8
    max_depth: int = 4
    read_attempts: int = 2

    def __post_init__(self):
        for key, value in self.__dict__.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid budget: {key}")
            if key not in {"max_seconds", "call_timeout"} and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"Budget must be an integer: {key}")
        if self.call_timeout <= 0 or self.max_seconds <= 0 or self.read_attempts < 1:
            raise ValueError("Time limits and read_attempts must be positive")


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
        self.snapshot = dict(snapshot or {})
        self.models = dict(self.snapshot.get("models", {}))
        self.counters = {"model_calls": 0, "tool_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                         "elapsed_seconds": 0.0, **self.snapshot.get("budget", {})}
        self.events = list(events or [])
        self.sequence = max([event.get("sequence", 0) for event in self.events] + [0])
        self.started = time.monotonic()
        self.prior_elapsed = self.counters["elapsed_seconds"]
        self.model_cursor = 0
        self.approval = approval
        self.phase = self.snapshot.get("phase", "INTAKE")

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
        self.snapshot.update(phase=self.phase, models=self.models, budget=self.counters, sequence=self.sequence)
        if self.store:
            self.store.save_checkpoint(self.run_id, self.token, self.snapshot)

    def record(self, event_type, **data):
        self.sequence += 1
        event = {"run_id": self.run_id, "sequence": self.sequence, "timestamp": datetime.now(UTC).isoformat(),
                 "type": event_type, **data}
        self.events.append(event)
        self.counters["elapsed_seconds"] = self.prior_elapsed + time.monotonic() - self.started
        self.snapshot.update(phase=self.phase, models=self.models, budget=self.counters, sequence=self.sequence)
        if self.store:
            self.store.save_checkpoint(self.run_id, self.token, self.snapshot, event)
        return event

    def model_call(self, model, messages, *, purpose, agent=None, allow_tools=True, schemas=None):
        from src.llm import extract_usage
        self.model_cursor += 1
        key = str(self.model_cursor)
        canonical = messages if isinstance(messages, str) else [
            {"type": getattr(message, "type", "ai" if hasattr(message, "tool_calls") else "human"),
             "content": getattr(message, "content", str(message)), "tool_calls": getattr(message, "tool_calls", []),
             "tool_call_id": getattr(message, "tool_call_id", None)} for message in messages]
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
        self.require("model", estimate)
        self.counters["model_calls"] += 1
        self.counters["prompt_tokens"] += estimate
        self.phase = "MODEL_CALL"
        self.record("model_started", purpose=purpose, agent=agent, prompt_budget_reserved=estimate)
        completion_limit = self.limits.max_completion_tokens - self.counters["completion_tokens"]
        if hasattr(model, "bind"):
            model = model.bind(max_tokens=completion_limit)
        started = time.monotonic()
        for attempt in range(2):
            try:
                invoke = getattr(model, "invoke_once", model.invoke)
                response = bounded_call(lambda: invoke(messages), min(self.limits.call_timeout, self.remaining_seconds), "model")
                break
            except Exception as exc:
                error = classify_exception(exc, source="model")
                self.record("error", source="model", purpose=purpose, agent=agent, code=error.code,
                            message=str(error), retryable=error.retryable)
                if not error.retryable or attempt == 1:
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
        self.record("model_call", purpose=purpose, agent=agent, status="SUCCEEDED", usage=usage,
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


@dataclass
class ToolSpec:
    function: object
    schema: object = None
    side_effect: bool = False
    transactional: bool = False


def tool_spec(name, fn):
    if isinstance(fn, ToolSpec):
        return fn
    schema = getattr(fn, "args_schema", None)
    function = getattr(fn, "func", None) or fn
    if schema is None:
        fields = {}
        try:
            hints = get_type_hints(function)
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
        schema = create_model(f"{name}Arguments", __config__=ConfigDict(extra="forbid", strict=True), **fields)
    side_effect = name in {"process_payment", "initiate_refund"}
    transactional = side_effect and getattr(function, "__module__", "") in {"src.tools", "src.agents.payment_agent", "src.agents.refund_agent"}
    return ToolSpec(function, schema, side_effect, transactional)


class ToolExecutor:
    def __init__(self, registry, recorder, max_calls=8, *, approved_action=None):
        self.registry = {name: tool_spec(name, fn) for name, fn in registry.items()}
        self.recorder, self.max_calls = recorder, max_calls
        self.call_count = 0
        self._results = {}
        self.approval = approved_action

    @staticmethod
    def fingerprint(tool_name, args):
        return f"{tool_name}:" + json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    def _fail(self, error, name, args):
        self.recorder.record("error", tool=name, args=args, code=error.code, message=str(error), retryable=error.retryable)
        raise error

    def execute(self, name, args):
        if name not in self.registry:
            self._fail(AgentError(ErrorCode.UNKNOWN_TOOL, f"工具 {name} 不在允许列表中", retryable=False), name, args)
        spec = self.registry[name]
        try:
            if not isinstance(args, dict):
                raise TypeError("工具参数必须是对象")
            if spec.schema and set(args) - set(spec.schema.model_fields):
                raise TypeError("包含未知工具参数")
            args = spec.schema.model_validate(args, strict=True).model_dump() if spec.schema else args
        except (ValidationError, TypeError) as exc:
            self._fail(AgentError(ErrorCode.INVALID_TOOL_ARGUMENTS, str(exc), retryable=False), name, args)
        context = current_runtime()
        context = context if context and context.run_id == self.recorder.run_id else None
        approval = context.approval if context and context.store else (context.approval if context else None) or self.approval
        if spec.side_effect and (not approval or approval.get("order_id") != args.get("order_id") or
                                approval.get("type") != ("PROCESS_PAYMENT" if name == "process_payment" else "INITIATE_REFUND")):
            pending = {"type": "PROCESS_PAYMENT" if name == "process_payment" else "INITIATE_REFUND",
                       "order_id": args["order_id"], "requires_approval": True,
                       **{key: value for key, value in args.items() if key in {"reason", "method"}}}
            if name == "process_payment":
                from src import tools
                # Approval preparation is a real, budgeted read through the
                # same executor; it must not become an invisible policy call.
                self.registry.setdefault("get_payment_status", tool_spec("get_payment_status", tools.get_payment_status))
                payment, _ = self.execute("get_payment_status", {"order_id": args["order_id"]})
                if payment.get("error"):
                    self._fail(AgentError("BUSINESS_DATA_NOT_FOUND", payment["error"], retryable=False), name, args)
                pending["amount"] = payment.get("amount")
            self.recorder.record("approval_required", tool=name, args=args)
            raise ApprovalRequired(pending)
        fingerprint = self.fingerprint(name, args)
        key = context.store.ledger_key(context.run_id, name, args, approval if spec.side_effect else None) if context and context.store else None
        saved = context.store.ledger(key, name, args) if key else None
        if saved and (saved["status"] == "UNKNOWN" or (saved["status"] == "RUNNING" and spec.side_effect and not spec.transactional)):
            self._fail(AgentError("UNCERTAIN_TOOL_OUTCOME", "外部操作结果不确定，禁止自动重放", retryable=False), name, args)
        cached = saved["result"] if saved and saved["status"] == "SUCCEEDED" else self._results.get(fingerprint)
        if (saved and saved["status"] == "SUCCEEDED") or fingerprint in self._results:
            event = self.recorder.record("tool_result", tool=name, args=args, result=cached,
                                         status="BUSINESS_ERROR" if isinstance(cached, dict) and cached.get("error") else "REUSED", duration_ms=0.0)
            return cached, event
        if self.call_count >= self.max_calls:
            self._fail(AgentError(ErrorCode.TOOL_BUDGET_EXHAUSTED, "工具调用预算已耗尽", retryable=False), name, args)
        attempts = context.limits.read_attempts if context and not spec.side_effect else 1
        for attempt in range(attempts):
            if self.call_count >= self.max_calls:
                self._fail(AgentError(ErrorCode.TOOL_BUDGET_EXHAUSTED, "工具调用预算已耗尽", retryable=False), name, args)
            if context:
                context.require("tool")
                context.counters["tool_calls"] += 1
                context.phase = "TOOL_CALL"
            self.call_count += 1
            self.recorder.record("tool_call", tool=name, args=args, call_index=self.call_count, attempt=attempt + 1)
            started = time.monotonic()
            try:
                if key and spec.transactional:
                    result = context.store.execute_write(context.run_id, context.token, key, name, args, approval, lambda: spec.function(**args))
                else:
                    if key:
                        context.store.save_ledger(context.run_id, context.token, key, name, args, "RUNNING")
                    if context and not spec.side_effect:
                        result = bounded_call(lambda: spec.function(**args), min(context.limits.call_timeout, context.remaining_seconds), "tool")
                    elif context and spec.side_effect:
                        # Non-transactional writes may finish after a deadline;
                        # UNKNOWN is deliberately not eligible for automatic replay.
                        result = bounded_call(lambda: spec.function(**args), min(context.limits.call_timeout, context.remaining_seconds), "tool")
                    else:
                        result = spec.function(**args)
                    if key:
                        context.store.save_ledger(context.run_id, context.token, key, name, args, "SUCCEEDED", result)
            except Exception as exc:
                error = classify_exception(exc, source="tool")
                if key:
                    context.store.save_ledger(context.run_id, context.token, key, name, args,
                                              "UNKNOWN" if spec.side_effect and not spec.transactional else "FAILED", {"error": str(error), "code": error.code})
                self.recorder.record("error", tool=name, args=args, code=error.code, message=str(error), retryable=error.retryable,
                                     duration_ms=round((time.monotonic() - started) * 1000, 2))
                if not spec.side_effect and error.retryable and attempt + 1 < attempts and self.call_count < self.max_calls:
                    self.recorder.record("tool_retry", tool=name, attempt=attempt + 2)
                    continue
                raise error from exc
            self._results[fingerprint] = result
            status = "BUSINESS_ERROR" if isinstance(result, dict) and result.get("error") else (
                "BUSINESS_REJECTED" if isinstance(result, dict) and result.get("success") is False else "SUCCEEDED")
            event = self.recorder.record("tool_result", tool=name, args=args, result=result, status=status,
                                         duration_ms=round((time.monotonic() - started) * 1000, 2))
            return result, event
