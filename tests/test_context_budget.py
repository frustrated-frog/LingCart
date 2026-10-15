from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.context_budget import compact_context


def _assert_no_orphan_tool_messages(messages):
    pending = set()
    for message in messages:
        if getattr(message, "tool_calls", None):
            pending.update(call["id"] for call in message.tool_calls)
        if getattr(message, "type", None) == "tool":
            assert message.tool_call_id in pending
            pending.remove(message.tool_call_id)


def test_compaction_drops_old_protocol_groups_atomically():
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="old question " * 30),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": "old-call"}]),
        ToolMessage(content="old observation " * 30, tool_call_id="old-call"),
        HumanMessage(content="new question"),
    ]

    compacted, stats = compact_context(
        messages,
        max_chars=120,
        max_tool_observation_chars=80,
        max_history_message_chars=80,
    )

    assert stats["changed"]
    assert compacted[0].type == "system"
    assert compacted[-1].content == "new question"
    _assert_no_orphan_tool_messages(compacted)


def test_compaction_preserves_latest_tool_request_and_result_pair():
    messages = [
        SystemMessage(content="system"),
        HumanMessage(content="old " * 100),
        HumanMessage(content="当前目标"),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": "latest"}]),
        ToolMessage(content="result " * 100, tool_call_id="latest"),
    ]

    compacted, stats = compact_context(
        messages,
        max_chars=180,
        max_tool_observation_chars=100,
        max_history_message_chars=100,
    )

    assert stats["dropped_messages"] == 1
    assert any(message.content == "当前目标" for message in compacted)
    assert any(getattr(message, "tool_calls", None) for message in compacted)
    assert any(getattr(message, "tool_call_id", None) == "latest" for message in compacted)
    _assert_no_orphan_tool_messages(compacted)
