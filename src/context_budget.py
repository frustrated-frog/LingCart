"""Deterministic context compaction that preserves tool-call protocol groups."""
from __future__ import annotations

import json


def _text_length(message) -> int:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        size = len(content)
    else:
        size = len(json.dumps(content, ensure_ascii=False, default=str))
    return size + len(json.dumps(getattr(message, "tool_calls", []), ensure_ascii=False, default=str))


def _truncate_text(value, limit: int):
    if not isinstance(value, str) or len(value) <= limit:
        return value, False
    marker = "\n…（上下文已压缩）…\n"
    if limit <= len(marker) + 2:
        return value[:limit], True
    head = (limit - len(marker)) * 2 // 3
    tail = limit - len(marker) - head
    return value[:head] + marker + value[-tail:], True


def _copy_with_content(message, content):
    if content == getattr(message, "content", None):
        return message
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content})
    clone = __import__("copy").copy(message)
    clone.content = content
    return clone


def _protocol_groups(messages: list) -> list[list]:
    """Keep an assistant tool request and all immediately following results atomic."""

    groups, index = [], 0
    while index < len(messages):
        message = messages[index]
        group = [message]
        if getattr(message, "tool_calls", None):
            index += 1
            while index < len(messages) and getattr(messages[index], "type", None) == "tool":
                group.append(messages[index])
                index += 1
            groups.append(group)
            continue
        groups.append(group)
        index += 1
    return groups


def compact_context(messages, *, max_chars: int, max_tool_observation_chars: int,
                    max_history_message_chars: int):
    """Return compacted messages and observable statistics.

    The system message and newest protocol group are retained. Older groups are
    evicted atomically, so compaction never creates an orphan ``ToolMessage``.
    """

    if isinstance(messages, str):
        compacted, changed = _truncate_text(messages, max_chars)
        return compacted, {
            "changed": changed,
            "input_chars": len(messages),
            "output_chars": len(compacted),
            "dropped_messages": 0,
            "truncated_messages": int(changed),
        }

    original = list(messages)
    prepared, truncated = [], 0
    for message in original:
        limit = max_tool_observation_chars if getattr(message, "type", None) == "tool" else max_history_message_chars
        content, changed = _truncate_text(getattr(message, "content", ""), limit)
        prepared.append(_copy_with_content(message, content))
        truncated += int(changed)

    groups = _protocol_groups(prepared)
    input_chars = sum(_text_length(message) for message in original)
    total = sum(_text_length(message) for message in prepared)
    dropped = 0
    # The first group normally contains the system instruction. Preserve it and
    # the latest conversational/protocol group; evict oldest middle history.
    while total > max_chars and len(groups) > 2:
        removed = groups.pop(1)
        removed_size = sum(_text_length(message) for message in removed)
        total -= removed_size
        dropped += len(removed)

    compacted = [message for group in groups for message in group]
    # Very large system/current messages still need a hard cap. Shrink the
    # largest contents first while keeping message and protocol identities.
    while total > max_chars:
        candidates = [
            (len(getattr(message, "content", "")), index, message)
            for index, message in enumerate(compacted)
            if isinstance(getattr(message, "content", None), str) and len(message.content) > 64
        ]
        if not candidates:
            break
        _, index, message = max(candidates)
        target = max(64, len(message.content) - (total - max_chars))
        content, changed = _truncate_text(message.content, target)
        compacted[index] = _copy_with_content(message, content)
        if changed:
            truncated += 1
        total = sum(_text_length(item) for item in compacted)

    output_chars = sum(_text_length(message) for message in compacted)
    return compacted, {
        "changed": dropped > 0 or truncated > 0,
        "input_chars": input_chars,
        "output_chars": output_chars,
        "dropped_messages": dropped,
        "truncated_messages": truncated,
    }
