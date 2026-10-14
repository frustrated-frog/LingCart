"""Project low-level runtime events into one stable, UI-friendly trace view."""
from __future__ import annotations


_PHASES = {
    "context_compacted": "CONTEXT",
    "model_started": "MODEL",
    "model_call": "MODEL",
    "model_reused": "MODEL",
    "model_retry": "MODEL",
    "tool_call": "TOOL",
    "tool_retry": "TOOL",
    "tool_result": "TOOL",
    "tool_protocol": "TOOL_PROTOCOL",
    "approval_required": "APPROVAL",
    "loop_warning": "GUARD",
    "loop_blocked": "GUARD",
    "error": "ERROR",
    "terminal": "TERMINAL",
}

_LABELS = {
    "context_compacted": "上下文已压缩",
    "model_started": "模型调用开始",
    "model_call": "模型调用完成",
    "model_reused": "复用模型决策",
    "model_retry": "重试模型调用",
    "tool_call": "工具调用开始",
    "tool_retry": "重试工具调用",
    "tool_result": "工具调用完成",
    "tool_protocol": "工具协议已闭合",
    "approval_required": "等待用户确认",
    "loop_warning": "检测到无进展循环",
    "loop_blocked": "已阻断无进展循环",
    "error": "运行错误",
    "terminal": "任务结束",
}


def project_trace(events: list[dict]) -> dict:
    ordered = sorted(events or [], key=lambda event: event.get("sequence", 0))
    run_ids = {event.get("run_id") for event in ordered if event.get("run_id")}
    expected = list(range(1, len(ordered) + 1))
    actual = [event.get("sequence") for event in ordered]
    timeline = []
    for event in ordered:
        event_type = event.get("type", "unknown")
        timeline.append({
            "sequence": event.get("sequence"),
            "timestamp": event.get("timestamp"),
            "type": event_type,
            "phase": _PHASES.get(event_type, "RUNTIME"),
            "label": _LABELS.get(event_type, "运行事件"),
            "status": event.get("status"),
            "agent": event.get("agent"),
            "tool": event.get("tool"),
            "purpose": event.get("purpose"),
            "code": event.get("code"),
            "action_fingerprint": event.get("action_fingerprint"),
            "tool_call_id": event.get("tool_call_id"),
            "duration_ms": event.get("duration_ms"),
            "policy": event.get("policy"),
        })
    terminal = next((event for event in reversed(ordered) if event.get("type") == "terminal"), None)
    return {
        "schema_version": 1,
        "run_id": next(iter(run_ids)) if len(run_ids) == 1 else None,
        "integrity": {
            "sequential": actual == expected,
            "single_run": len(run_ids) <= 1,
            "event_count": len(ordered),
        },
        "summary": {
            "model_calls": sum(event.get("type") == "model_call" for event in ordered),
            "tool_calls": sum(event.get("type") == "tool_call" for event in ordered),
            "tool_results": sum(event.get("type") == "tool_result" for event in ordered),
            "protocol_results": sum(event.get("type") == "tool_protocol" for event in ordered),
            "errors": sum(event.get("type") == "error" for event in ordered),
            "guard_warnings": sum(event.get("type") == "loop_warning" for event in ordered),
            "terminal_status": (terminal or {}).get("status"),
            "terminal_reason": (terminal or {}).get("reason"),
        },
        "timeline": timeline,
    }
