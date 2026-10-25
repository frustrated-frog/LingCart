"""把 PG 运行事件投影为可审计的干预链；不推断模型隐藏推理。"""
from __future__ import annotations

from collections import Counter
import hashlib
import json


def _record(value):
    return value if isinstance(value, dict) else {}


def _sequence(event, fallback):
    value = event.get("sequence")
    return value if isinstance(value, int) else fallback


def _bounded(value, limit=1600):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(text) <= limit:
        return value
    return {
        "truncated": True,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "preview": text[:limit],
        "original_chars": len(text),
    }


def _invalid_tool_result(event):
    result = event.get("result")
    if event.get("status") in {"FAILED", "TIMEOUT", "REJECTED"}:
        return True
    if not isinstance(result, dict):
        return False
    return bool(result.get("error") or result.get("success") is False or result.get("products") == [])


def _alignment(guidance, decision, review=None):
    if review:
        return {
            "status": review.get("status") or "UNKNOWN",
            "matched_edge_ids": list(review.get("matched_edge_ids") or []),
            "expected_actions": list(review.get("recommended_actions") or []),
            "actual_action": review.get("actual_action"),
            "reason": review.get("reason"),
            "blocked": bool(review.get("blocked")),
            "review_id": review.get("review_id"),
        }
    if not decision:
        return {"status": "NO_DECISION", "matched_edge_ids": []}
    subgraph = _record(guidance.get("subgraph"))
    nodes = {item.get("id"): item for item in subgraph.get("nodes", []) if isinstance(item, dict)}
    selected = set(guidance.get("edge_ids") or [])
    edges = [item for item in subgraph.get("edges", [])
             if isinstance(item, dict) and item.get("id") in selected]
    actionable = [item for item in edges if _record(nodes.get(item.get("target"))).get("kind") == "ACTION"]
    if not actionable:
        return {"status": "UNKNOWN", "matched_edge_ids": [],
                "reason": "选中关系没有直接指向行动节点，不能据此判断模型是否采纳"}
    payload = _record(decision.get("decision"))
    action = payload.get("tool") or payload.get("action")
    matched = [item["id"] for item in actionable if item.get("target") == action]
    return {"status": "ALIGNED" if matched else "DIVERGED", "matched_edge_ids": matched,
            "expected_actions": sorted({item.get("target") for item in actionable}), "actual_action": action}


def project_turn(turn, turn_index=0):
    result = _record(turn.get("result"))
    source = list(result.get("events") or [])
    events = [dict(event, _order=index) for index, event in enumerate(source) if isinstance(event, dict)]
    events.sort(key=lambda event: (_sequence(event, event["_order"]), event["_order"]))
    guidances = [event for event in events if event.get("type") == "pg_guidance"]
    interventions = []
    for index, guidance in enumerate(guidances):
        start = _sequence(guidance, guidance["_order"])
        end = (_sequence(guidances[index + 1], guidances[index + 1]["_order"])
               if index + 1 < len(guidances) else float("inf"))
        window = [event for event in events if start < _sequence(event, event["_order"]) < end]
        exposure = next((event for event in window if event.get("type") == "evidence_context"
                         and event.get("pg_decision_key") == guidance.get("decision_key")), None)
        applied = next((event for event in window if event.get("type") == "mechanism_applied"
                        and event.get("decision_key") == guidance.get("decision_key")), None)
        review = next((event for event in window if event.get("type") == "pg_decision_reviewed"
                       and event.get("decision_key") == guidance.get("decision_key")), None)
        outcome = next((event for event in window if event.get("type") == "pg_action_outcome"
                        and (not review or event.get("review_id") == review.get("review_id"))), None)
        decision = next((event for event in window if event.get("type") == "adaptive_decision"), None)
        decision_sequence = _sequence(decision, -1) if decision else -1
        action = _record(_record(decision).get("decision"))
        tool_name = action.get("tool") if action.get("action") == "tool" else None
        tool_call = next((event for event in window if event.get("type") == "tool_call"
                          and _sequence(event, event["_order"]) > decision_sequence
                          and (not tool_name or event.get("tool") == tool_name)), None)
        call_sequence = _sequence(tool_call, decision_sequence) if tool_call else decision_sequence
        tool_result = next((event for event in window if event.get("type") == "tool_result"
                            and _sequence(event, event["_order"]) > call_sequence
                            and (not tool_name or event.get("tool") == tool_name)), None)
        verification = next((event for event in reversed(window)
                             if event.get("type") in {"verification", "verification_failed"}), None)
        status = _record(exposure).get("pg_status") or guidance.get("status")
        localization = _record(guidance.get("localization"))
        shadow = _record(localization.get("shadow"))
        interventions.append({
            "intervention_id": guidance.get("decision_key"),
            "turn_index": turn_index,
            "graph": {key: guidance.get(key) for key in ("graph_id", "graph_version", "graph_sha256")},
            "localization": {
                "anchor": guidance.get("anchor"),
                "fallback": bool(guidance.get("fallback")),
                "abstained": bool(guidance.get("abstained")),
                "algorithm": localization.get("algorithm") or "action-anchor-v1",
                "status": localization.get("status"),
                "reason": localization.get("reason"),
                "confidence": localization.get("confidence"),
                "candidates": list(localization.get("candidates") or []),
                "shadow": shadow or None,
                "shadow_changed": bool(shadow) and (
                    shadow.get("anchor") != guidance.get("anchor")
                    or bool(shadow.get("fallback")) != bool(guidance.get("fallback"))
                ),
                "node_ids": list(guidance.get("node_ids") or []),
                "state_snapshot": guidance.get("state_snapshot"),
                "sequence": start,
            },
            "retrieval": {
                "retrieved_edge_ids": list(guidance.get("retrieved_edge_ids") or []),
                "selected_edge_ids": list(guidance.get("edge_ids") or []),
                "excluded_edge_ids": list(localization.get("excluded_edge_ids") or []),
                "subgraph": guidance.get("subgraph"),
            },
            "applicability": {
                "strategy": _record(guidance.get("applicability")).get("strategy"),
                "status": _record(guidance.get("applicability")).get("status"),
                "reason": _record(guidance.get("applicability")).get("reason"),
                "evaluations": list(_record(guidance.get("applicability")).get("evaluations") or []),
                "selected_edge_ids": list(_record(guidance.get("applicability")).get("selected_edge_ids") or []),
                "false_edge_ids": list(_record(guidance.get("applicability")).get("false_edge_ids") or []),
                "unknown_edge_ids": list(_record(guidance.get("applicability")).get("unknown_edge_ids") or []),
                "projected_paths": list(guidance.get("projected_paths") or []),
            },
            "guidance": {
                "status": status,
                "reason": guidance.get("reason"),
                "content": guidance.get("guidance") or "",
                "exposed": bool(applied) and _record(exposure).get("pg_status") == "EXPOSED",
                "generation_mode": _record(guidance.get("generation")).get("mode") or "model-contextualization-v1",
                "generation_model_calls": _record(guidance.get("generation")).get("model_calls"),
                "contract": guidance.get("guidance_contract"),
                "sequence": start,
            },
            "decision": ({"sequence": _sequence(decision, decision["_order"]),
                          "source": decision.get("source"), "payload": action} if decision else None),
            "action_alignment": _alignment(guidance, decision, review),
            "action_progress": ({
                "status": outcome.get("progress_status"),
                "reason": outcome.get("progress_reason"),
                "signals": list(outcome.get("signals") or []),
                "cost_delta": _record(outcome.get("cost_delta")),
                "outcome_id": outcome.get("outcome_id"),
                "review_id": outcome.get("review_id"),
                "matched_edge_ids": list(outcome.get("matched_edge_ids") or []),
                "before_sha256": outcome.get("before_sha256"),
                "after_sha256": outcome.get("after_sha256"),
            } if outcome else None),
            "tool": ({"name": tool_result.get("tool"), "args": tool_result.get("args"),
                      "status": tool_result.get("status"), "sequence": _sequence(tool_result, tool_result["_order"]),
                      "result": _bounded(tool_result.get("result")),
                      "invalid_result": _invalid_tool_result(tool_result)} if tool_result else None),
            "verification": ({key: verification.get(key) for key in
                              ("type", "sequence", "code", "all_goals_satisfied", "conversation_complete")}
                             if verification else None),
        })
    environment = _record(result.get("evaluation_environment"))
    passed = bool(turn.get("passed", not turn.get("failures")))
    return {
        "turn_index": turn_index,
        "request": turn.get("message"),
        "passed": passed,
        "failures": list(turn.get("failures") or []),
        "environment": {"kind": environment.get("kind"), "fired": bool(environment.get("fired"))},
        "interventions": interventions,
    }


def _failure_stage(turns):
    if all(turn["passed"] for turn in turns):
        return None
    interventions = [item for turn in turns for item in turn["interventions"]]
    if not interventions:
        return "BASELINE_TASK_FAILURE"
    if any(item["localization"]["abstained"] for item in interventions):
        return "LOCALIZATION_ABSTAINED"
    if any(item["localization"]["fallback"] for item in interventions):
        return "LOCALIZATION_FALLBACK"
    if any(item["guidance"]["status"] in {"FAILED", "SKIPPED", "ABSTAINED"} for item in interventions):
        return "GUIDANCE_UNAVAILABLE"
    if any(not item["guidance"]["exposed"] for item in interventions):
        return "GUIDANCE_NOT_EXPOSED"
    if any(_record(item.get("action_progress")).get("status") == "FAILED" for item in interventions):
        return "GUIDED_ACTION_FAILED"
    if any(_record(item.get("action_progress")).get("status") == "NO_PROGRESS" for item in interventions):
        return "GUIDED_ACTION_NO_PROGRESS"
    if any(item["decision"] is None for item in interventions):
        if any(item["action_alignment"].get("blocked") for item in interventions):
            return "NO_PROGRESS_REPEAT_BLOCKED"
        return "NO_FOLLOWUP_DECISION"
    if any(_record(item.get("tool")).get("invalid_result") for item in interventions):
        return "TOOL_OUTCOME_FAILURE"
    if any(item["action_alignment"]["status"] == "DIVERGED" for item in interventions):
        return "GUIDANCE_ACTION_DIVERGENCE"
    if any(item.get("verification") or turn["failures"] for turn in turns for item in turn["interventions"]):
        return "GOAL_VERIFICATION_FAILURE"
    return "UNCLASSIFIED_TASK_FAILURE"


def project_sample_evidence(sample):
    turns = [project_turn(turn, index) for index, turn in enumerate(sample.get("turns", []))]
    interventions = [item for turn in turns for item in turn["interventions"]]
    alignments = Counter(item["action_alignment"]["status"] for item in interventions)
    progress = Counter(_record(item.get("action_progress")).get("status") for item in interventions)
    fired = any(turn["environment"]["fired"] for turn in turns)
    passed = bool(sample.get("passed"))
    return {
        "schema_version": 4,
        "case_id": sample.get("case_id"),
        "repetition": sample.get("repetition"),
        "passed": passed,
        "summary": {
            "interventions": len(interventions),
            "exposed": sum(item["guidance"]["exposed"] for item in interventions),
            "aligned": alignments["ALIGNED"],
            "diverged": alignments["DIVERGED"],
            "unknown_alignment": alignments["UNKNOWN"] + alignments["NO_DECISION"],
            "blocked_repeats": alignments["BLOCKED"],
            "progressed_actions": progress["PROGRESSED"],
            "no_progress_actions": progress["NO_PROGRESS"],
            "failed_actions": progress["FAILED"],
            "deferred_actions": progress["DEFERRED"],
            "unknown_progress_actions": progress["UNKNOWN"],
            "typed_applicability_interventions": sum(
                bool(item["applicability"].get("strategy")) for item in interventions),
            "deterministic_contracts": sum(
                item["guidance"]["generation_mode"] == "deterministic-contract-v1" for item in interventions),
            "guidance_model_calls": sum(
                item["guidance"]["generation_model_calls"] or 0 for item in interventions),
            "localization_fallbacks": sum(item["localization"]["fallback"] for item in interventions),
            "localization_abstentions": sum(item["localization"]["abstained"] for item in interventions),
            "shadow_differences": sum(item["localization"]["shadow_changed"] for item in interventions),
            "shadow_fallbacks": sum(bool(_record(item["localization"].get("shadow")).get("fallback"))
                                    for item in interventions),
            "environment_triggered": fired,
            "recovered": passed and fired,
            "failure_stage": _failure_stage(turns),
        },
        "turns": turns,
    }
