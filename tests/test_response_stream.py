import pytest
from langchain_core.messages import AIMessageChunk

from src.errors import AgentError
from src.response_stream import content_sha256, response_text
from src.runtime import BudgetLimits, RunRuntime


class StreamingModel:
    max_attempts = 2

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = 0

    def bind(self, **_kwargs):
        return self

    def stream(self, _messages):
        self.calls += 1
        yield from self.chunks


def test_final_model_response_is_persisted_as_ordered_deltas():
    context = RunRuntime("stream-success")
    model = StreamingModel([
        AIMessageChunk(content="我在"),
        AIMessageChunk(content="为你核验"),
        AIMessageChunk(content="。"),
    ])

    result = context.model_call(
        model,
        "你好",
        purpose="final_response",
        stream_response=True,
        allow_tools=False,
    )

    events = context.events
    started = next(event for event in events if event["type"] == "response_started")
    completed = next(event for event in events if event["type"] == "response_completed")
    assert result.content == "我在为你核验。"
    assert response_text(events, started["message_id"]) == result.content
    assert completed["content_sha256"] == content_sha256(result.content)
    assert completed["chunk_count"] == 2
    assert model.calls == 1
    assert [event["type"] for event in events].index("response_completed") < [
        event["type"] for event in events
    ].index("model_call")


def test_partial_provider_stream_is_not_retried_or_spliced():
    class BrokenModel(StreamingModel):
        def stream(self, _messages):
            self.calls += 1
            yield AIMessageChunk(content="已经输出")
            raise TimeoutError("provider timeout")

    context = RunRuntime("stream-partial", BudgetLimits(max_model_calls=2))
    model = BrokenModel([])

    with pytest.raises(AgentError) as error:
        context.model_call(model, "你好", purpose="final_response", stream_response=True)

    failed = next(event for event in context.events if event["type"] == "response_failed")
    assert error.value.code == "MODEL_TIMEOUT"
    assert failed["partial"] is True
    assert failed["chunk_count"] == 1
    assert model.calls == 1
    assert not any(event["type"] == "model_retry" for event in context.events)


def test_assembled_response_is_idempotent_and_reuses_model_stream():
    context = RunRuntime("stream-dedupe")
    model = StreamingModel([AIMessageChunk(content="最终回答")])
    result = context.model_call(model, "你好", purpose="final_response", stream_response=True)

    first = context.publish_response(result.content)
    second = context.publish_response(result.content)

    assert first == second
    assert sum(event["type"] == "response_started" for event in context.events) == 1
    assert sum(event["type"] == "response_completed" for event in context.events) == 1


def test_response_reassembly_rejects_gaps_and_conflicts():
    gap = [
        {"type": "response_delta", "message_id": "m", "chunk_index": 1, "delta": "后"},
    ]
    with pytest.raises(AgentError, match="不连续"):
        response_text(gap, "m")

    conflict = [
        {"type": "response_delta", "message_id": "m", "chunk_index": 0, "delta": "甲"},
        {"type": "response_delta", "message_id": "m", "chunk_index": 0, "delta": "乙"},
    ]
    with pytest.raises(AgentError, match="冲突"):
        response_text(conflict, "m")
