from langchain_core.tools import tool

from src.agents import base


class FakeToolCallMessage:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeLLM:
    """Replays a scripted sequence of AIMessage-like responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    def bind_tools(self, _tools):
        return self

    def invoke(self, _messages):
        msg = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return msg


class FailingAfterToolLLM(FakeLLM):
    def invoke(self, _messages):
        if self._i == 0:
            self._i += 1
            return self._responses[0]
        raise RuntimeError("simulated model outage after tool result")


@tool
def echo_tool(value: str) -> str:
    """Echoes back a value."""
    return f"echoed:{value}"


@tool
def failing_tool(value: str) -> str:
    """Always fails for runtime error testing."""
    raise RuntimeError(f"tool failed:{value}")


def test_run_tool_agent_returns_normal_reply(monkeypatch):
    responses = [
        FakeToolCallMessage(tool_calls=[{"name": "echo_tool", "args": {"value": "hi"}, "id": "1"}]),
        FakeToolCallMessage(content="Here is your answer."),
    ]
    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: FakeLLM(responses))

    result = base.run_tool_agent("test", "system prompt", [echo_tool], "hello")

    assert result["reply"] == "Here is your answer."
    assert len(result["trace"]) == 1


def test_run_tool_agent_falls_back_when_model_never_answers(monkeypatch):
    # Model calls the same tool with the same args every round and never
    # produces closing text - reproduces the real bug seen with Groq/Llama
    # under long conversation history.
    stuck_response = FakeToolCallMessage(
        content="", tool_calls=[{"name": "echo_tool", "args": {"value": "x"}, "id": "1"}]
    )
    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: FakeLLM([stuck_response] * 10))

    result = base.run_tool_agent("test", "system prompt", [echo_tool], "hello")

    assert result["reply"] == "echoed:x"  # falls back to the tool result, never blank
    assert len(result["trace"]) == 1  # loop-detection stopped after the first repeat


def test_run_tool_agent_stops_on_repeated_identical_call(monkeypatch):
    responses = [
        FakeToolCallMessage(tool_calls=[{"name": "echo_tool", "args": {"value": "a"}, "id": "1"}]),
        FakeToolCallMessage(tool_calls=[{"name": "echo_tool", "args": {"value": "a"}, "id": "2"}]),
        FakeToolCallMessage(content="should never get here"),
    ]
    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: FakeLLM(responses))

    result = base.run_tool_agent("test", "system prompt", [echo_tool], "hello")

    assert len(result["trace"]) == 1
    assert result["reply"] == "echoed:a"
    assert result["terminal_reason"] == "REPEATED_TOOL_CALL"


def test_runtime_closes_every_tool_call_when_a_batch_is_stopped(monkeypatch):
    responses = [
        FakeToolCallMessage(tool_calls=[{"name": "echo_tool", "args": {"value": "a"}, "id": "first"}]),
        FakeToolCallMessage(tool_calls=[
            {"name": "echo_tool", "args": {"value": "a"}, "id": "duplicate"},
            {"name": "echo_tool", "args": {"value": "b"}, "id": "remaining"},
        ]),
    ]
    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: FakeLLM(responses))

    result = base.run_tool_agent("test", "system prompt", [echo_tool], "hello")

    tool_messages = [message for message in result["messages"] if getattr(message, "type", None) == "tool"]
    assert [message.tool_call_id for message in tool_messages] == ["first", "duplicate", "remaining"]
    closed = [event for event in result["events"] if event["type"] == "tool_protocol"]
    assert [(event["tool_call_id"], event["status"]) for event in closed] == [
        ("first", "SUCCEEDED"),
        ("duplicate", "CANCELLED_BY_RUNTIME"),
        ("remaining", "CANCELLED_BY_RUNTIME"),
    ]


def test_run_tool_agent_rejects_unknown_tool_without_key_error(monkeypatch):
    responses = [
        FakeToolCallMessage(tool_calls=[{"name": "not_allowed", "args": {}, "id": "1"}]),
        FakeToolCallMessage(content="I could not use that tool."),
    ]
    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: FakeLLM(responses))

    result = base.run_tool_agent("test", "system prompt", [echo_tool], "hello")

    assert result["status"] == "FAILED"
    assert result["error"]["code"] == "UNKNOWN_TOOL"
    assert result["trace"][0]["status"] == "FAILED"


def test_run_tool_agent_records_invalid_tool_arguments(monkeypatch):
    responses = [
        FakeToolCallMessage(tool_calls=[{"name": "echo_tool", "args": {"wrong": "x"}, "id": "1"}]),
        FakeToolCallMessage(content="The arguments were invalid."),
    ]
    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: FakeLLM(responses))

    result = base.run_tool_agent("test", "system prompt", [echo_tool], "hello")

    assert result["status"] == "FAILED"
    assert result["error"]["code"] == "INVALID_TOOL_ARGUMENTS"


def test_run_tool_agent_records_tool_exception(monkeypatch):
    responses = [
        FakeToolCallMessage(tool_calls=[{"name": "failing_tool", "args": {"value": "x"}, "id": "1"}]),
        FakeToolCallMessage(content="The tool failed."),
    ]
    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: FakeLLM(responses))

    result = base.run_tool_agent("test", "system prompt", [failing_tool], "hello")

    assert result["status"] == "FAILED"
    assert result["error"]["code"] == "TOOL_ERROR"
    assert result["trace"][0]["status"] == "FAILED"


def test_model_failure_after_tool_does_not_replay_tool(monkeypatch):
    first = FakeToolCallMessage(tool_calls=[{"name": "echo_tool", "args": {"value": "safe"}, "id": "1"}])
    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: FailingAfterToolLLM([first]))

    result = base.run_tool_agent("test", "system prompt", [echo_tool], "hello")

    assert len(result["trace"]) == 1
    assert result["reply"] == "echoed:safe"
    assert result["status"] == "PARTIAL"
    assert result["terminal_reason"] == "MODEL_FAILED_AFTER_TOOL_RESULT"
