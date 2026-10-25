"""Project low-level runtime events into one stable, UI-friendly trace view."""
from __future__ import annotations


_PHASES = {
    "evidence_semantic_assessed": "VERIFY",
    "evidence_semantic_reused": "VERIFY",
    "case_plan_revised": "PLAN",
    "case_state_projected": "CONTEXT",
    "case_obligation_verified": "VERIFY",
    "case_autonomous_action_selected": "PLAN",
    "case_autonomous_action_completed": "TOOL",
    "case_execution_paused": "GUARD",
    "agent_status_changed": "RUNTIME",
    "response_started": "RESPONSE",
    "response_delta": "RESPONSE",
    "response_completed": "RESPONSE",
    "response_failed": "RESPONSE",
    "harness_profile_selected": "HARNESS",
    "constraint_binding_proposed": "PLAN",
    "constraint_binding_unresolved": "PLAN",
    "constraint_binding_rejected": "VERIFY",
    "constraint_bindings_refined": "PLAN",
    "decision_action_surface": "GUARD",
    "trajectory_monitor_evaluated": "MONITOR",
    "trajectory_drift_detected": "MONITOR",
    "corrective_advice_generated": "ADVISOR",
    "corrective_advice_failed": "ADVISOR",
    "intervention_applied": "INTERVENTION",
    "request_audited": "VERIFY",
    "catalog_request_audited": "VERIFY",
    "compilation_coverage": "PLAN",
    "catalog_goal_verified": "VERIFY",
    "catalog_delivery_verified": "VERIFY",
    "context_compacted": "CONTEXT",
    "pg_guidance": "CONTEXT",
    "pg_decision_reviewed": "VERIFY",
    "pg_action_outcome": "VERIFY",
    "pg_prediction_outcome": "VERIFY",
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
    "evidence_semantic_assessed": "已独立核验证据语义与冲突",
    "evidence_semantic_reused": "已复用持久化证据语义核验",
    "case_plan_revised": "售后案件证据与行动计划已修订",
    "case_state_projected": "已投影政策约束下的案件状态",
    "case_obligation_verified": "已核验下一项政策义务",
    "case_autonomous_action_selected": "已选择受限自主动作",
    "case_autonomous_action_completed": "受限自主动作已产生版本进展",
    "case_execution_paused": "政策感知执行已在边界处暂停",
    "agent_status_changed": "Agent 状态变化",
    "response_started": "回答流开始",
    "response_delta": "回答流增量",
    "response_completed": "回答流完成",
    "response_failed": "回答流中断",
    "harness_profile_selected": "已选择 Harness Profile",
    "constraint_binding_proposed": "已生成目录语义映射提案",
    "constraint_binding_unresolved": "目录语义映射等待兼容证据",
    "constraint_binding_rejected": "目录语义映射未通过独立审查",
    "constraint_bindings_refined": "已依据证据修订执行映射",
    "decision_action_surface": "已按剩余预算收敛决策动作面",
    "trajectory_monitor_evaluated": "已评估执行轨迹",
    "trajectory_drift_detected": "检测到轨迹漂移",
    "corrective_advice_generated": "已生成单步纠偏建议",
    "corrective_advice_failed": "纠偏模型失败并安全降级",
    "intervention_applied": "已应用受限干预",
    "request_audited": "原始需求与证据审查完成",
    "catalog_request_audited": "目录原始需求双向审查完成",
    "compilation_coverage": "目录目标编译覆盖已生成",
    "catalog_goal_verified": "目录目标核验通过",
    "catalog_delivery_verified": "目录结果交付核验通过",
    "context_compacted": "上下文已压缩",
    "pg_guidance": "程序图指导已生成",
    "pg_decision_reviewed": "程序图指导采用已审查",
    "pg_action_outcome": "程序图行动真实进展已记录",
    "pg_prediction_outcome": "程序图预演与真实结果已核对",
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


def _audit_response_streams(events):
    from src.response_stream import content_sha256

    streams, issues = {}, []
    for event in events:
        event_type = event.get("type")
        if event_type not in {"response_started", "response_delta", "response_completed", "response_failed"}:
            continue
        message_id = event.get("message_id")
        if not message_id:
            issues.append("MISSING_RESPONSE_MESSAGE_ID")
            continue
        state = streams.setdefault(message_id, {"started": 0, "chunks": {}, "terminal": []})
        if event_type == "response_started":
            state["started"] += 1
        elif event_type == "response_delta":
            index, delta = event.get("chunk_index"), event.get("delta")
            if type(index) is not int or index < 0 or not isinstance(delta, str):
                issues.append("INVALID_RESPONSE_CHUNK")
            elif index in state["chunks"] and state["chunks"][index] != delta:
                issues.append("CONFLICTING_RESPONSE_CHUNK")
            else:
                state["chunks"][index] = delta
        else:
            state["terminal"].append(event)
    for state in streams.values():
        if state["started"] != 1:
            issues.append("INVALID_RESPONSE_START_COUNT")
        if len(state["terminal"]) != 1:
            issues.append("INVALID_RESPONSE_TERMINAL_COUNT")
            continue
        chunks = state["chunks"]
        if chunks and sorted(chunks) != list(range(max(chunks) + 1)):
            issues.append("RESPONSE_CHUNK_GAP")
        terminal = state["terminal"][0]
        if terminal.get("type") == "response_completed":
            content = "".join(chunks[index] for index in sorted(chunks))
            if terminal.get("chunk_count") != len(chunks) or terminal.get("content_sha256") != content_sha256(content):
                issues.append("RESPONSE_COMPLETION_MISMATCH")
    return (not issues), issues, len(streams)


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
            "phase": _PHASES.get(event_type, "RUNTIME"),
            "label": event.get("label") if event_type == "agent_status_changed" else _LABELS.get(event_type, "运行事件"),
            **{key: event.get(key) for key in ("status", "agent", "tool", "purpose", "code", "action_fingerprint",
                                              "tool_call_id", "protocol_id", "execution_id", "duration_ms", "policy",
                                              "transaction_id", "resolution_id", "mode", "reason", "drift_id",
                                              "intervention_id", "outcome", "signal_codes", "source",
                                              "message_id", "chunk_index", "char_count")},
        })
    terminals = [event for event in ordered if event.get("type") == "terminal"]
    terminal = terminals[-1] if terminals else None
    tool_pairs, protocol_pairs, replay_evidence, legacy, issues = _audit_pairs(ordered)
    has_response_events = any(event.get("type", "").startswith("response_") for event in ordered)
    response_stream, response_issues, response_count = _audit_response_streams(ordered)
    issues.extend(response_issues)
    checks = {
        "sequential": actual == expected and all(type(value) is int for value in actual),
        "single_run": len(run_ids) == 1 and all(event.get("run_id") for event in ordered),
        "single_terminal": len(terminals) == 1 and ordered[-1] is terminal,
        "tool_pairs": tool_pairs,
        "protocol_pairs": protocol_pairs,
        "replay_evidence": replay_evidence,
    }
    if has_response_events:
        checks["response_stream"] = response_stream
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
            "drift_detections": sum(event.get("type") == "trajectory_drift_detected" for event in ordered),
            "advisor_calls": sum(event.get("type") == "model_call" and event.get("purpose") == "liveplan_advisor"
                                 for event in ordered),
            "interventions": sum(event.get("type") == "intervention_applied"
                                 and event.get("outcome") == "ACTIVATED" for event in ordered),
            "action_rejections": sum(event.get("type") == "intervention_applied"
                                     and event.get("outcome") == "ACTION_REJECTED" for event in ordered),
            "response_streams": response_count,
            "response_chunks": sum(event.get("type") == "response_delta" for event in ordered),
            "terminal_status": (terminal or {}).get("status"),
            "terminal_reason": (terminal or {}).get("reason"),
        },
        "timeline": timeline,
    }
