"""Small reliability primitives shared by deterministic and model-driven runs."""

from __future__ import annotations

import inspect
import json
import time
from datetime import UTC, datetime

from src.errors import AgentError, ErrorCode, classify_exception


class TraceRecorder:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.events: list[dict] = []

    def record(self, event_type: str, **data) -> dict:
        event = {
            "run_id": self.run_id,
            "sequence": len(self.events) + 1,
            "timestamp": datetime.now(UTC).isoformat(),
            "type": event_type,
            **data,
        }
        self.events.append(event)
        return event


class ToolExecutor:
    """Validate, bound and observe plain Python business-tool calls.

    Exact duplicate calls reuse their recorded result. This is safe for the
    read-only V1 fast path and prevents accidental duplicate work in one run.
    Side-effect tools are deliberately kept out of the fast-path registry.
    """

    def __init__(self, registry: dict[str, object], recorder: TraceRecorder, max_calls: int = 8):
        self.registry = registry
        self.recorder = recorder
        self.max_calls = max_calls
        self.call_count = 0
        self._results: dict[str, object] = {}

    @staticmethod
    def fingerprint(tool_name: str, args: dict) -> str:
        normalized = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return f"{tool_name}:{normalized}"

    def execute(self, tool_name: str, args: dict) -> tuple[object, dict]:
        if tool_name not in self.registry:
            error = AgentError(ErrorCode.UNKNOWN_TOOL, f"工具 {tool_name} 不在允许列表中", retryable=False)
            self.recorder.record("error", code=error.code, message=str(error), tool=tool_name, retryable=False)
            raise error
        if self.call_count >= self.max_calls:
            error = AgentError(ErrorCode.TOOL_BUDGET_EXHAUSTED, "工具调用预算已耗尽", retryable=False)
            self.recorder.record("error", code=error.code, message=str(error), tool=tool_name, retryable=False)
            raise error

        fn = self.registry[tool_name]
        try:
            inspect.signature(fn).bind(**args)
        except TypeError as exc:
            error = AgentError(ErrorCode.INVALID_TOOL_ARGUMENTS, str(exc), retryable=False)
            self.recorder.record("error", code=error.code, message=str(error), tool=tool_name, args=args, retryable=False)
            raise error from exc

        fingerprint = self.fingerprint(tool_name, args)
        if fingerprint in self._results:
            result = self._results[fingerprint]
            event = self.recorder.record(
                "tool_result",
                tool=tool_name,
                args=args,
                result=result,
                status="REUSED",
                duration_ms=0.0,
            )
            return result, event

        self.call_count += 1
        self.recorder.record("tool_call", tool=tool_name, args=args, call_index=self.call_count)
        started_at = time.perf_counter()
        try:
            result = fn(**args)
        except Exception as exc:  # noqa: BLE001 - converted into typed runtime semantics
            error = classify_exception(exc, source="tool")
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            self.recorder.record(
                "error",
                code=error.code,
                message=str(error),
                tool=tool_name,
                args=args,
                retryable=error.retryable,
                duration_ms=duration_ms,
            )
            raise error from exc

        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        self._results[fingerprint] = result
        status = "BUSINESS_ERROR" if isinstance(result, dict) and result.get("error") else "SUCCEEDED"
        event = self.recorder.record(
            "tool_result",
            tool=tool_name,
            args=args,
            result=result,
            status=status,
            duration_ms=duration_ms,
        )
        return result, event
