"""在字符预算内压缩上下文，保留目标、指令和工具协议。"""
from __future__ import annotations

import copy
import json
import re

from src.errors import AgentError


class ContextBudgetError(AgentError):
    def __init__(self):
        super().__init__("CONTEXT_BUDGET_EXHAUSTED", "系统指令、当前目标与必要工具证据超过上下文预算，已停止模型调用", retryable=False)


def _text_length(message) -> int:
    content = getattr(message, "content", "")
    size = len(content) if isinstance(content, str) else len(json.dumps(content, ensure_ascii=False, default=str))
    return size + len(json.dumps(getattr(message, "tool_calls", []), ensure_ascii=False, default=str))


def _truncate_text(value, limit: int):
    if not isinstance(value, str) or len(value) <= limit:
        return value, False
    marker = "…（上下文已压缩）…"
    if limit < len(marker):
        return marker[:limit], True
    head = (limit - len(marker)) * 2 // 3
    tail = limit - len(marker) - head
    return value[:head] + marker + (value[-tail:] if tail else ""), True


def _copy_with_content(message, content):
    if content == getattr(message, "content", None):
        return message
    if hasattr(message, "model_copy"):
        return message.model_copy(update={"content": content})
    clone = copy.copy(message)
    clone.content = content
    return clone


# 这些字段承担业务断言或证据定位职责，不能为了省预算改变其值。
_BUSINESS_FIELDS = {
    "status", "success", "eligible", "amount", "price", "currency", "method", "reason", "error", "code",
    "total", "count", "stock", "quantity", "name", "item", "customer_name", "order_date", "timestamp",
    "tracking_message", "data_source", "source", "sources", "reference", "references", "ref", "refs",
    "url", "uri", "link", "citation", "citations",
}


def _protected_field(key):
    key = str(key).lower()
    return key in _BUSINESS_FIELDS or key == "id" or key.endswith(("_id", "_ids", "_url", "_uri", "_ref"))


def _project_observation(value, text_limit, *, protected=False):
    if protected:
        return value
    if isinstance(value, dict):
        return {key: _project_observation(item, text_limit, protected=_protected_field(key)) for key, item in value.items()}
    if isinstance(value, list):
        # 不删除记录：每个商品、订单及其证据仍能一一对应。
        return [_project_observation(item, text_limit) for item in value]
    if isinstance(value, str):
        # 字符串内的引用也不能被头尾截断抹去。
        references = re.findall(r"https?://[^\s<>\"']+", value)
        text, changed = _truncate_text(value, text_limit)
        return {"text": text, "references": references} if changed and references else text
    return value


def _compact_observation(content, limit):
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)
    if len(content) <= limit:
        return content, False
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        value = {"text": content}
    for text_limit in (160, 80, 32, 0):
        projected = _project_observation(value, text_limit)
        if isinstance(projected, dict):
            projected = {**projected, "_context_compacted": True}
        else:
            projected = {"observation": projected, "_context_compacted": True}
        encoded = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= limit:
            return encoded, True
    raise ContextBudgetError()


def _protocol_groups(messages):
    groups, index = [], 0
    while index < len(messages):
        group = [(index, messages[index])]
        if getattr(messages[index], "tool_calls", None):
            index += 1
            while index < len(messages) and getattr(messages[index], "type", None) == "tool":
                group.append((index, messages[index]))
                index += 1
        else:
            index += 1
        groups.append(group)
    return groups


def compact_context(messages, *, max_chars: int, max_tool_observation_chars: int,
                    max_history_message_chars: int):
    """指令与最新用户目标保持原文；最小有效上下文仍过大时返回有类型错误。"""
    if isinstance(messages, str):
        # 路由和规划仍使用整段字符串 prompt，无法可靠区分指令和目标。
        if len(messages) > max_chars:
            raise ContextBudgetError()
        return messages, dict(changed=False, input_chars=len(messages), output_chars=len(messages),
                              dropped_messages=0, truncated_messages=0)

    original = list(messages)
    input_chars = sum(_text_length(message) for message in original)
    latest_user = next((i for i in range(len(original) - 1, -1, -1)
                        if getattr(original[i], "type", None) == "human"), None)
    protected = {i for i, message in enumerate(original) if getattr(message, "type", None) == "system"}
    if latest_user is not None:
        protected.add(latest_user)
    groups = _protocol_groups(original)
    if groups:
        protected.update(i for i, _message in groups[-1])
    prepared, truncated = [], 0
    for index, message in enumerate(original):
        content = getattr(message, "content", "")
        if getattr(message, "type", None) == "tool":
            # 仅压缩仍被保留的观察，旧组可先原子删除，避免无必要的预算错误。
            prepared.append(message)
            continue
        if index not in protected:
            content, changed = _truncate_text(content, max_history_message_chars)
            truncated += int(changed)
        prepared.append(_copy_with_content(message, content))
    groups = _protocol_groups(prepared)
    dropped = 0
    while sum(_text_length(m) for group in groups for _, m in group) > max_chars:
        removable = next((i for i, group in enumerate(groups) if not any(index in protected for index, _ in group)), None)
        if removable is None:
            break
        dropped += len(groups.pop(removable))
    compacted = [message for group in groups for _, message in group]
    for index, message in enumerate(compacted):
        if getattr(message, "type", None) != "tool":
            continue
        content, changed = _compact_observation(message.content, max_tool_observation_chars)
        compacted[index] = _copy_with_content(message, content)
        truncated += int(changed)
    # 进一步缩减工具观察，但不能截断 JSON 或关键业务字段。
    for index, message in enumerate(compacted):
        total = sum(_text_length(item) for item in compacted)
        if total <= max_chars:
            break
        if getattr(message, "type", None) == "tool":
            target = len(message.content) - (total - max_chars)
            content, changed = _compact_observation(message.content, max(1, target))
            compacted[index] = _copy_with_content(message, content)
            truncated += int(changed)
    output_chars = sum(_text_length(message) for message in compacted)
    if output_chars > max_chars:
        raise ContextBudgetError()
    return compacted, dict(changed=dropped > 0 or truncated > 0, input_chars=input_chars,
                           output_chars=output_chars, dropped_messages=dropped, truncated_messages=truncated)
