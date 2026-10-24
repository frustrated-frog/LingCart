"""PG 指导后的目标进展快照与结果归因；只读取 Runtime 权威状态。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.procedural_graph import fingerprint


_DONE = {"SUCCEEDED", "REUSED", "REJECTED", "FAILED", "SKIPPED", "CANCELLED"}
_FAILED = {"REJECTED", "FAILED", "SKIPPED", "CANCELLED"}


def _has_useful_observation(step) -> bool:
    if getattr(step, "status", None) in _FAILED or getattr(step, "error", None):
        return False
    result = getattr(step, "result", None)
    if not isinstance(result, dict) or result.get("error") or result.get("success") is False:
        return False
    if isinstance(result.get("products"), list):
        return bool(result["products"])
    return bool(result)


def progress_snapshot(state: dict, plan, runtime) -> dict:
    """生成可持久化的小型快照，不复制商品、订单或工具正文。"""
    ledger = [row for row in state.get("objective_ledger") or [] if isinstance(row, dict)]
    evidence_refs = sorted({ref for row in ledger for ref in row.get("evidence_refs") or []
                            if isinstance(ref, str)})
    steps = list(getattr(plan, "steps", ()) or ())
    payload = {
        "schema_version": 1,
        "goal_revision": state.get("goal_revision", 0),
        "binding_revision": state.get("binding_revision", 0),
        "goal_defined": bool(state.get("goal")),
        "objective_count": len(ledger),
        "verified_objective_count": sum(row.get("status") == "VERIFIED" for row in ledger),
        "evidence_refs": evidence_refs,
        "completed_step_ids": [step.step_id for step in steps if getattr(step, "status", None) in _DONE],
        "useful_observation_step_ids": [step.step_id for step in steps if _has_useful_observation(step)],
        "feedback_code": (state.get("feedback") or {}).get("code"),
        "model_calls": runtime.counters.get("model_calls", 0),
        "tool_calls": runtime.counters.get("tool_calls", 0),
    }
    return {**payload, "snapshot_sha256": fingerprint(payload)}


def assess_action_progress(before: dict, after: dict, decision: dict,
                           *, step: Any | None = None) -> dict:
    """比较真实执行前后快照；失败或空观察不能被包装成进展。"""
    signals: list[str] = []
    if not before.get("goal_defined") and after.get("goal_defined"):
        signals.append("GOAL_DEFINED")
    if after.get("verified_objective_count", 0) > before.get("verified_objective_count", 0):
        signals.append("OBJECTIVES_VERIFIED")
    if len(after.get("evidence_refs") or []) > len(before.get("evidence_refs") or []):
        signals.append("EVIDENCE_ADDED")
    if len(after.get("useful_observation_step_ids") or []) > len(before.get("useful_observation_step_ids") or []):
        signals.append("TRUSTED_OBSERVATION_ADDED")
    if after.get("binding_revision", 0) > before.get("binding_revision", 0):
        signals.append("BINDING_REFINED")

    action = decision.get("tool") if decision.get("action") == "tool" else decision.get("action")
    step_status = getattr(step, "status", None)
    step_error = deepcopy(getattr(step, "error", None))
    if step_status in _FAILED or step_error:
        status, reason = "FAILED", "ACTION_FAILED"
    elif decision.get("action") in {"clarify", "handoff"}:
        status, reason = "DEFERRED", "WAITING_FOR_EXTERNAL_INPUT"
    elif signals:
        status, reason = "PROGRESSED", "VERIFIABLE_STATE_CHANGE"
    elif decision.get("action") in {"respond", "finish", "unsupported"}:
        status, reason = "UNKNOWN", "TERMINAL_WITHOUT_COMPARABLE_STATE"
    else:
        status, reason = "NO_PROGRESS", "NO_VERIFIABLE_STATE_CHANGE"

    return {
        "schema_version": 1,
        "progress_status": status,
        "progress_reason": reason,
        "action": action,
        "step_id": getattr(step, "step_id", None),
        "step_status": step_status,
        "step_error": step_error,
        "signals": signals,
        "cost_delta": {
            "model_calls": max(0, after.get("model_calls", 0) - before.get("model_calls", 0)),
            "tool_calls": max(0, after.get("tool_calls", 0) - before.get("tool_calls", 0)),
        },
        "before_sha256": before.get("snapshot_sha256"),
        "after_sha256": after.get("snapshot_sha256"),
    }
