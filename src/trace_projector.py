"""Project low-level runtime events into one stable, UI-friendly trace view."""
from __future__ import annotations


_PHASES = {
    "constraint_bindings_refined": "PLAN",
    "request_audited": "VERIFY",
    "context_compacted": "CONTEXT",
    "model_started": "MODEL",
    "model_call": "MODEL",
    "model_reused": "MODEL",
    "model_retry": "MODEL",
    "tool_requested": "TOOL_PROTOCOL",
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
    "constraint_bindings_refined": "已依据证据修订执行映射",
    "request_audited": "原始需求与证据审查完成",
    "context_compacted": "上下文已压缩",
    "model_started": "模型调用开始",
    "model_call": "模型调用完成",
    "model_reused": "复用模型决策",
    "model_retry": "重试模型调用",
    "tool_requested": "模型请求工具调用",
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


def _audit_pairs(events):
    pending, requested, issues = {}, {}, []
    seen_execution, seen_protocol = set(), set()
    evidence = []
    consumed_evidence = set()
    completed_fingerprints = set()
    unverified_reuse = False
    legacy_pending = []
    legacy_evidence = False
    protocol_evidence = False
    for event in events:
        event_type = event.get("type")
        if event_type == "tool_requested":
            protocol_evidence = True
            key = event.get("protocol_id")
            if not key or key in seen_protocol:
                issues.append("DUPLICATE_OR_MISSING_PROTOCOL_ID")
            else:
                requested[key] = event
                seen_protocol.add(key)
        elif event_type == "tool_protocol":
            key = event.get("protocol_id")
            if key is None:
                legacy_evidence = True
            else:
                request = requested.pop(key, None)
                if request is None or any(event.get(field) != request.get(field) for field in
                                          ("run_id", "tool", "tool_call_id", "action_fingerprint")):
                    issues.append("UNMATCHED_TOOL_PROTOCOL")
                else:
                    status = event.get("status")
                    candidates = [item for item in evidence if item.get("sequence") not in consumed_evidence and item.get("sequence", 0) > request.get("sequence", 0)
                                  and item.get("run_id") == request.get("run_id") and item.get("tool") == request.get("tool")]
                    if status in {"SUCCEEDED", "REUSED", "BUSINESS_REJECTED", "BUSINESS_ERROR"}:
                        proof = next((item for item in candidates if item.get("type") == "tool_result" and item.get("status") == status
                                      and item.get("action_fingerprint") == request.get("action_fingerprint")), None)
                        supported = proof is not None
                        if proof:
                            consumed_evidence.add(proof.get("sequence"))
                    elif status == "FAILED":
                        proof = next((item for item in candidates if item.get("type") == "error"
                                     and item.get("action_fingerprint") == request.get("action_fingerprint")), None)
                        supported = proof is not None
                        if proof:
                            consumed_evidence.add(proof.get("sequence"))
                    elif status == "WAITING_APPROVAL":
                        proof = next((item for item in candidates if item.get("type") == "approval_required"
                                     and item.get("action_fingerprint") == request.get("action_fingerprint")), None)
                        supported = proof is not None
                        if proof:
                            consumed_evidence.add(proof.get("sequence"))
                    else:
                        supported = status == "CANCELLED_BY_RUNTIME"
                    if not supported:
                        issues.append("UNSUPPORTED_PROTOCOL_OUTCOME")
        elif event_type == "tool_call":
            key = event.get("execution_id")
            if key:
                if key in seen_execution:
                    issues.append("DUPLICATE_TOOL_EXECUTION_ID")
                pending[key] = event
                seen_execution.add(key)
            else:
                legacy_evidence = True
                legacy_pending.append(event)
        elif event_type in {"tool_result", "error"}:
            evidence.append(event)
            key = event.get("execution_id")
            identity = (event.get("run_id"), event.get("tool"), event.get("action_fingerprint"))
            if event_type == "tool_result" and (event.get("reused") or event.get("status") == "REUSED"):
                recovered = pending.pop(key, None) if key else None
                if recovered and any(event.get(field) != recovered.get(field) for field in ('run_id', 'tool', 'action_fingerprint')):
                    issues.append('UNMATCHED_TOOL_EXECUTION')
                if recovered and event.get('source_ledger'):
                    completed_fingerprints.add(identity)
                if not event.get("action_fingerprint") or identity not in completed_fingerprints:
                    unverified_reuse = True
                continue
            if key:
                call = pending.pop(key, None)
                if call is None or any(event.get(field) != call.get(field) for field in
                                       ("run_id", "tool", "action_fingerprint")):
                    issues.append("UNMATCHED_TOOL_EXECUTION")
                elif event_type == "tool_result":
                    completed_fingerprints.add(identity)
            elif event.get("tool") and (event_type == "tool_result" or event.get("duration_ms") is not None):
                legacy_evidence = True
                # 旧事件没有执行编号，仍核对名称、顺序和已提供的参数/指纹。
                match = next((i for i, call in enumerate(legacy_pending)
                              if call.get("tool") == event.get("tool")
                              and all(call.get(field) == event.get(field) for field in ("args", "action_fingerprint")
                                      if field in call and field in event)), None)
                if match is None:
                    issues.append("UNMATCHED_TOOL_EXECUTION")
                else:
                    legacy_pending.pop(match)
                    if event_type == "tool_result":
                        completed_fingerprints.add(identity)
        elif event_type == "approval_required":
            evidence.append(event)
    if pending or legacy_pending:
        issues.append("MISSING_TOOL_OUTCOME")
    if requested:
        issues.append("MISSING_TOOL_PROTOCOL")
    tool_pairs = not any(issue in issues for issue in (
        "DUPLICATE_TOOL_EXECUTION_ID", "UNMATCHED_TOOL_EXECUTION", "MISSING_TOOL_OUTCOME"))
    protocol_pairs = not any(issue in issues for issue in (
        "DUPLICATE_OR_MISSING_PROTOCOL_ID", "UNMATCHED_TOOL_PROTOCOL", "MISSING_TOOL_PROTOCOL", "UNSUPPORTED_PROTOCOL_OUTCOME"))
    if legacy_evidence and protocol_pairs:
        # 没有请求证据时，不用相等计数声称协议完整。
        protocol_pairs = None
    return tool_pairs, protocol_pairs, (None if unverified_reuse else True), legacy_evidence, issues


def project_trace(events: list[dict]) -> dict:
    original = list(events or [])
    ordered = sorted(original, key=lambda event: event.get("sequence") if isinstance(event.get("sequence"), int) else -1)
    run_ids = {event.get("run_id") for event in ordered if event.get("run_id")}
    expected = list(range(1, len(ordered) + 1))
    actual = [event.get("sequence") for event in original]
    timeline = []
    for event in ordered:
        event_type = event.get("type", "unknown")
        timeline.append({
            "sequence": event.get("sequence"), "timestamp": event.get("timestamp"), "type": event_type,
            "phase": _PHASES.get(event_type, "RUNTIME"), "label": _LABELS.get(event_type, "运行事件"),
            **{key: event.get(key) for key in ("status", "agent", "tool", "purpose", "code", "action_fingerprint",
                                              "tool_call_id", "protocol_id", "execution_id", "duration_ms", "policy")},
        })
    terminals = [event for event in ordered if event.get("type") == "terminal"]
    terminal = terminals[-1] if terminals else None
    tool_pairs, protocol_pairs, replay_evidence, legacy, issues = _audit_pairs(ordered)
    checks = {
        "sequential": actual == expected and all(type(value) is int for value in actual),
        "single_run": len(run_ids) == 1 and all(event.get("run_id") for event in ordered),
        "single_terminal": len(terminals) == 1 and ordered[-1] is terminal,
        "tool_pairs": tool_pairs,
        "protocol_pairs": protocol_pairs,
        "replay_evidence": replay_evidence,
    }
    complete = False if any(value is False for value in checks.values()) else (
        None if any(value is None for value in checks.values()) else True)
    return {
        "schema_version": 2,
        "run_id": next(iter(run_ids)) if len(run_ids) == 1 else None,
        "integrity": {**checks, "complete": complete, "legacy_evidence": legacy, "issues": issues,
                      "event_count": len(ordered)},
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
