import pytest

from src.errors import AgentError, ErrorCode
from src.runtime import ToolExecutor, TraceRecorder


def echo(value: str) -> dict:
    return {"value": value}


def explode(value: str) -> dict:
    raise RuntimeError(f"boom:{value}")


def test_tool_executor_records_real_call_and_result():
    recorder = TraceRecorder("run-1")
    executor = ToolExecutor({"echo": echo}, recorder)

    result, event = executor.execute("echo", {"value": "ok"})

    assert result == {"value": "ok"}
    assert event["status"] == "SUCCEEDED"
    assert [item["type"] for item in recorder.events] == ["tool_call", "tool_result"]
    assert all(item["run_id"] == "run-1" for item in recorder.events)


def test_exact_duplicate_reuses_result_without_second_execution():
    calls = {"count": 0}

    def counted(value: str):
        calls["count"] += 1
        return {"value": value}

    recorder = TraceRecorder("run-2")
    executor = ToolExecutor({"counted": counted}, recorder)
    executor.execute("counted", {"value": "same"})
    result, event = executor.execute("counted", {"value": "same"})

    assert result == {"value": "same"}
    assert event["status"] == "REUSED"
    assert calls["count"] == 1


def test_unknown_tool_has_stable_error_code():
    recorder = TraceRecorder("run-3")
    executor = ToolExecutor({}, recorder)
    with pytest.raises(AgentError) as caught:
        executor.execute("missing", {})
    assert caught.value.code == ErrorCode.UNKNOWN_TOOL
    assert recorder.events[-1]["type"] == "error"


def test_invalid_arguments_are_rejected_before_execution():
    recorder = TraceRecorder("run-4")
    executor = ToolExecutor({"echo": echo}, recorder)
    with pytest.raises(AgentError) as caught:
        executor.execute("echo", {"wrong": "value"})
    assert caught.value.code == ErrorCode.INVALID_TOOL_ARGUMENTS


def test_tool_exception_is_typed_and_observable():
    recorder = TraceRecorder("run-5")
    executor = ToolExecutor({"explode": explode}, recorder)
    with pytest.raises(AgentError) as caught:
        executor.execute("explode", {"value": "x"})
    assert caught.value.code == ErrorCode.TOOL_ERROR
    assert recorder.events[-1]["type"] == "error"
    assert recorder.events[-1]["retryable"] is False


def test_tool_budget_is_enforced():
    recorder = TraceRecorder("run-6")
    executor = ToolExecutor({"echo": echo}, recorder, max_calls=1)
    executor.execute("echo", {"value": "one"})
    with pytest.raises(AgentError) as caught:
        executor.execute("echo", {"value": "two"})
    assert caught.value.code == ErrorCode.TOOL_BUDGET_EXHAUSTED
