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
    assert len(recorder.events[0]["action_fingerprint"]) == 64
    assert recorder.events[0]["policy"] == {
        "risk_level": "READ_ONLY",
        "effect": "READ",
        "replay_policy": "REPLAY_SAFE",
        "concurrency_policy": "PARALLEL_SAFE",
        "transactional": False,
    }


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


def test_deadline_is_bounded_and_does_not_start_a_second_write():
    import time
    from src.runtime import BudgetLimits, RunRuntime, ToolSpec, runtime_scope
    calls = []
    def slow(value: str):
        calls.append(value)
        time.sleep(.1)
        return {"value": value}
    context = RunRuntime("deadline", BudgetLimits(call_timeout=.01, read_attempts=1))
    started = time.monotonic()
    with runtime_scope(context), pytest.raises(AgentError) as error:
        ToolExecutor({"slow": slow}, TraceRecorder(context.run_id)).execute("slow", {"value": "a"})
    assert error.value.code == "TOOL_TIMEOUT"
    assert time.monotonic() - started < .08
    assert calls == ["a"]


def test_budget_priority_and_strict_argument_types():
    from src.runtime import BudgetLimits, RunRuntime, runtime_scope
    context = RunRuntime("limits", BudgetLimits(max_model_calls=0, max_tool_calls=0, max_prompt_tokens=0))
    assert context.stop_reason("model", 1) == "MODEL_ROUND_BUDGET_EXHAUSTED"
    assert context.stop_reason("tool") == "TOOL_BUDGET_EXHAUSTED"
    context.started -= 31
    assert context.stop_reason("model", 1) == "TIME_BUDGET_EXHAUSTED"
    with pytest.raises(AgentError) as error:
        ToolExecutor({"echo": echo}, TraceRecorder("strict")).execute("echo", {"value": 123})
    assert error.value.code == "INVALID_TOOL_ARGUMENTS"


def test_final_model_round_cannot_execute_tools(monkeypatch):
    from langchain_core.messages import AIMessage
    from langchain_core.tools import tool
    from src.agents import base
    from src.runtime import BudgetLimits, RunRuntime, runtime_scope
    calls = []
    @tool
    def final_tool(value: str):
        """A test tool that must not execute on the final model round."""
        calls.append(value)
        return value
    class Model:
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            return AIMessage(content="", tool_calls=[{"name": "final_tool", "args": {"value": "a"}, "id": "one"}])
    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: Model())
    context = RunRuntime("final", BudgetLimits(max_model_calls=1))
    with runtime_scope(context):
        result = base.run_tool_agent("test", "system", [final_tool], "hello", run_id=context.run_id)
    assert result["terminal_reason"] == "FINAL_TOOL_CALL_FORBIDDEN"
    assert calls == []
    tool_messages = [message for message in result["messages"] if getattr(message, "type", None) == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "one"
    assert any(
        event["type"] == "tool_protocol" and event["status"] == "CANCELLED_BY_RUNTIME"
        for event in result["events"]
    )


def test_side_effect_tool_uses_high_risk_serial_policy():
    from src import tools
    from src.runtime import tool_spec

    policy = tool_spec("process_payment", tools.process_payment).policy_payload()

    assert policy == {
        "risk_level": "HIGH",
        "effect": "WRITE",
        "replay_policy": "REQUIRES_APPROVAL",
        "concurrency_policy": "SERIAL_PER_RUN",
        "transactional": True,
    }


def test_provider_rotation_counts_each_attempt_against_runtime_budget():
    from langchain_core.messages import AIMessage
    from src.llm import RotatingChatModel
    from src.runtime import BudgetLimits, RunRuntime
    calls = []
    class Provider:
        def __init__(self, name):
            self.name = name
        def bind(self, **kwargs):
            return self
        def invoke(self, messages):
            calls.append(self.name)
            if self.name == "rate_limited":
                raise RuntimeError("429 rate limit")
            return AIMessage(content="done")
    context = RunRuntime("rotation", BudgetLimits(max_model_calls=2))
    result = context.model_call(RotatingChatModel([Provider("rate_limited"), Provider("okay")], {"index": 0}), "hi", purpose="test")
    assert result.content == "done"
    assert calls == ["rate_limited", "okay"]
    assert context.counters["model_calls"] == 2
