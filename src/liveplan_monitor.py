"""Deterministic, replayable trajectory projection and drift detection."""
from __future__ import annotations

from copy import deepcopy

from pydantic import ValidationError

from src.liveplan_contracts import DriftSignal, ProgressVector, TrajectoryAction, fingerprint


_PHASE_BY_TOOL = {
    "get_catalog_facets": "DISCOVER",
    "search_products": "DISCOVER",
    "discover_products": "DISCOVER",
    "compare_products": "COMPARE",
    "answer_faq": "INVESTIGATE",
    "get_order_status": "INVESTIGATE",
    "get_order_details": "INVESTIGATE",
    "track_shipment": "INVESTIGATE",
    "check_refund_eligibility": "INVESTIGATE",
    "get_payment_status": "INVESTIGATE",
    "read_clock": "INVESTIGATE",
    "calculate": "INVESTIGATE",
}


def _stable_result(value):
    """Remove run-local observation metadata before comparing evidence value."""

    if isinstance(value, dict):
        return {key: _stable_result(item) for key, item in value.items()
                if key not in {"observation", "timestamp", "reference", "duration_ms"}}
    if isinstance(value, list):
        return [_stable_result(item) for item in value]
    return value


def decision_phase(decision: dict) -> str:
    action = decision.get("action")
    if action == "tool":
        return _PHASE_BY_TOOL.get(decision.get("tool"), "INVESTIGATE")
    return {
        "inspect": "EVIDENCE",
        "refine_constraints": "BIND",
        "recommend": "RECOMMEND",
        "complete": "COMPLETE",
        "clarify": "CLARIFY",
        "finish": "TERMINATE",
        "respond": "RESPOND",
        "unsupported": "TERMINATE",
        "handoff": "HANDOFF",
        "revise_plan": "PLAN",
    }.get(action, "UNDERSTAND")


def decision_signature(decision) -> str:
    payload = decision.model_dump() if hasattr(decision, "model_dump") else deepcopy(decision)
    action = payload.get("action")
    identity: dict = {"action": action}
    if action == "tool":
        identity.update(tool=payload.get("tool"), args=payload.get("args") or {})
    elif action == "inspect":
        identity.update(reference=payload.get("evidence_ref"), offset=payload.get("offset"), limit=payload.get("limit"))
    elif action == "recommend":
        identity.update(product_id=payload.get("product_id"), objective_id=payload.get("objective_id"),
                        evidence_refs=payload.get("evidence_refs") or [])
    elif action == "complete":
        identity["claims"] = payload.get("claims") or []
    elif action == "refine_constraints":
        identity["bindings"] = payload.get("bindings") or []
    return fingerprint(identity)


def _progress(context, state: dict, plan) -> ProgressVector:
    ledger = state.get("objective_ledger", [])
    evidence = {
        fingerprint({"tool": step.tool, "args": step.args, "result": _stable_result(step.result)})
        for step in plan.steps
        if step.status in {"SUCCEEDED", "REUSED"} and step.result is not None
    }
    return ProgressVector(
        goal_revision=max(0, int(state.get("goal_revision", 0))),
        verified_objectives=sum(row.get("status") == "VERIFIED" for row in ledger),
        pending_objectives=sum(row.get("status") != "VERIFIED" for row in ledger),
        binding_revision=max(0, int(state.get("binding_revision", 0))),
        distinct_evidence=len(evidence),
        evidence_sha256=fingerprint(sorted(evidence)),
        remaining_model_calls=max(0, context.limits.max_model_calls - context.counters["model_calls"]),
        remaining_tool_calls=max(0, context.limits.max_tool_calls - context.counters["tool_calls"]),
    )


def project_trajectory(context, state: dict, plan) -> dict:
    completed = {event.get("step_id"): event for event in context.events
                 if event.get("type") == "step_completed" and event.get("step_id")}
    actions = []
    for event in context.events:
        if event.get("type") != "adaptive_decision" or not isinstance(event.get("decision"), dict):
            continue
        decision = event["decision"]
        step_event = completed.get(event.get("step_id"))
        actions.append(TrajectoryAction(
            sequence=event["sequence"],
            action=(f"tool:{decision.get('tool')}" if decision.get("action") == "tool"
                    else str(decision.get("action") or "unknown")),
            phase=decision_phase(decision),
            signature=decision_signature(decision),
            step_id=event.get("step_id"),
            status=(step_event or {}).get("status"),
        ))
    failures = [event for event in context.events if event.get("type") == "verification_failed"]
    return {"actions": actions, "verification_failures": failures, "progress": _progress(context, state, plan)}


def recent_actions(context) -> list[TrajectoryAction]:
    """Project action identity without touching raw tool observations."""

    actions = []
    for event in context.events:
        decision = event.get("decision")
        if event.get("type") != "adaptive_decision" or not isinstance(decision, dict):
            continue
        actions.append(TrajectoryAction(
            sequence=event["sequence"],
            action=(f"tool:{decision.get('tool')}" if decision.get("action") == "tool"
                    else str(decision.get("action") or "unknown")),
            phase=decision_phase(decision),
            signature=decision_signature(decision),
            step_id=event.get("step_id"),
        ))
    return actions


def _prior_evaluation(events: list[dict], *, action_count: int) -> dict | None:
    candidates = [event for event in events if event.get("type") == "trajectory_monitor_evaluated"
                  and isinstance(event.get("action_count"), int) and event["action_count"] <= action_count]
    return candidates[-1] if candidates else None


def _same_progress(left: ProgressVector, right: dict | None) -> bool:
    if not right:
        return False
    try:
        return left.semantic_key() == ProgressVector.model_validate(right).semantic_key()
    except ValidationError:
        return False


def detect_drifts(context, projection: dict, *, action_repeat_threshold: int,
                  phase_dwell_threshold: int, evidence_stagnation_threshold: int,
                  verification_repeat_threshold: int) -> list[DriftSignal]:
    actions: list[TrajectoryAction] = projection["actions"]
    progress: ProgressVector = projection["progress"]
    signals: list[DriftSignal] = []

    if len(actions) >= action_repeat_threshold:
        window = actions[-action_repeat_threshold:]
        prior = _prior_evaluation(context.events, action_count=len(actions) - 1)
        if len({item.signature for item in window}) == 1 and _same_progress(progress, (prior or {}).get("progress")):
            signals.append(DriftSignal(
                code="ACTION_OSCILLATION", severity="blocking",
                evidence_sequences=[item.sequence for item in window],
                detail="相同的规范化动作重复执行，但目标、映射与权威证据均未变化。",
            ))
    if len(actions) >= 4:
        window = actions[-4:]
        prior = _prior_evaluation(context.events, action_count=len(actions) - 2)
        if (window[0].signature == window[2].signature and window[1].signature == window[3].signature
                and window[0].signature != window[1].signature
                and _same_progress(progress, (prior or {}).get("progress"))):
            signals.append(DriftSignal(
                code="ACTION_OSCILLATION", severity="blocking",
                evidence_sequences=[item.sequence for item in window],
                detail="最近动作形成无语义进展的双节点循环。",
            ))

    if len(actions) >= phase_dwell_threshold:
        window = actions[-phase_dwell_threshold:]
        prior = _prior_evaluation(context.events, action_count=len(actions) - phase_dwell_threshold + 1)
        if len({item.phase for item in window}) == 1 and _same_progress(progress, (prior or {}).get("progress")):
            signals.append(DriftSignal(
                code="PHASE_STAGNATION", severity="warning",
                evidence_sequences=[item.sequence for item in window],
                detail=f"连续 {phase_dwell_threshold} 个动作停留在 {window[-1].phase}，没有语义进展。",
            ))

    if len(actions) >= evidence_stagnation_threshold:
        window = actions[-evidence_stagnation_threshold:]
        prior = _prior_evaluation(context.events, action_count=len(actions) - evidence_stagnation_threshold + 1)
        evidence_phases = {"DISCOVER", "COMPARE", "EVIDENCE", "INVESTIGATE"}
        if (all(item.phase in evidence_phases for item in window)
                and _same_progress(progress, (prior or {}).get("progress"))):
            signals.append(DriftSignal(
                code="EVIDENCE_STAGNATION", severity="warning",
                evidence_sequences=[item.sequence for item in window],
                detail="连续证据动作没有增加新的去重权威观察。",
            ))

    failures = projection["verification_failures"]
    if len(failures) >= verification_repeat_threshold:
        window = failures[-verification_repeat_threshold:]
        keys = {fingerprint({"code": item.get("code"), "failures": item.get("failures")}) for item in window}
        if len(keys) == 1:
            signals.append(DriftSignal(
                code="VERIFICATION_LOOP", severity="warning",
                evidence_sequences=[item["sequence"] for item in window],
                detail="相同原因的核验失败重复出现。",
            ))

    if progress.pending_objectives and progress.remaining_model_calls <= 2:
        signals.append(DriftSignal(
            code="BUDGET_RISK", severity="warning",
            evidence_sequences=[actions[-1].sequence],
            detail="剩余模型调用仅够一次决策和一次独立终态审查。",
        ))
    # A single assessment never repeats the same signal code.
    return list({signal.code: signal for signal in signals}.values())
