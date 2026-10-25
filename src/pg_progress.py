"""PG 行动进展：区分执行事实与基于内容身份的可信信息变化。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.procedural_graph import fingerprint


_DONE = {"SUCCEEDED", "REUSED", "REJECTED", "FAILED", "SKIPPED", "CANCELLED"}
_FAILED = {"REJECTED", "FAILED", "SKIPPED", "CANCELLED"}
_SUCCESS = {"SUCCEEDED", "REUSED"}


def _observation_units(step) -> dict[str, str]:
    """按业务范围生成摘要；不把 Runtime 时间戳、步骤 ID 或查询回显当作事实。"""
    if getattr(step, "status", None) not in _SUCCESS or getattr(step, "error", None):
        return {}
    result = getattr(step, "result", None)
    if not isinstance(result, dict) or result.get("error") or result.get("success") is False:
        return {}
    if step.tool == "read_evidence":
        return {}  # 归档读取复用来源身份，由 evidence_units 解析。
    content = {key: value for key, value in result.items() if key != "observation"}
    source = content.get("data_source")
    if isinstance(content.get("products"), list):
        units = {}
        version = {key: content[key] for key in ("source_version", "data_version", "revision") if key in content}
        for product in content["products"]:
            if not isinstance(product, dict) or not product.get("product_id"):
                continue  # 缺少对象身份的结果不能猜测为新增可信商品。
            scope = {"kind": "product", "source": product.get("data_source", source),
                     "product_id": product["product_id"]}
            units[fingerprint(scope)] = fingerprint({"record": product, "version": version})
        return units
    if not content:
        return {}
    scope = {"kind": "tool", "tool": step.tool, "args": step.args, "source": source}
    return {fingerprint(scope): fingerprint(content)}


def progress_snapshot(state: dict, plan, runtime) -> dict:
    """仅持久化身份和摘要；在恢复后可由权威计划重建，保持最后一次成功观察。"""
    ledger = [row for row in state.get("objective_ledger") or [] if isinstance(row, dict)]
    steps = list(getattr(plan, "steps", ()) or ())
    indexed = {step.step_id: step for step in steps}
    step_units = {step.step_id: _observation_units(step) for step in steps}
    observations: dict[str, str] = {}
    for step in steps:
        if step.status == "SUCCEEDED" and step.tool not in {"read_evidence", "select_product"}:
            observations.update(step_units[step.step_id])

    def evidence_units(ref, visited=frozenset()):
        if not isinstance(ref, str) or ref in visited or ref not in indexed:
            return {}
        step = indexed[ref]
        if step.status not in _SUCCESS or step.error:
            return {}
        if step.tool == "read_evidence":
            return evidence_units(step.args.get("reference"), visited | {ref})
        return step_units[ref]

    evidence = set()
    for row in ledger:
        for ref in row.get("evidence_refs") or []:
            for scope, content in evidence_units(ref).items():
                evidence.add(fingerprint({"objective_id": row.get("objective_id"),
                                          "scope": scope, "content": content}))
    payload = {
        "schema_version": 2,
        "goal_revision": state.get("goal_revision", 0),
        "binding_revision": state.get("binding_revision", 0),
        "bindings_sha256": fingerprint({ref: {key: value.get(key) for key in ("field", "op", "value")}
            for ref, value in (state.get("constraint_bindings") or {}).items() if isinstance(value, dict)}),
        "goal_defined": bool(state.get("goal")),
        "verified_objective_ids": sorted({row["objective_id"] for row in ledger
            if row.get("status") == "VERIFIED" and isinstance(row.get("objective_id"), str)}),
        "evidence_fingerprints": sorted(evidence),
        "observations": observations,
        "completed_step_ids": [step.step_id for step in steps if step.status in _DONE],
        "feedback_code": (state.get("feedback") or {}).get("code"),
        "model_calls": runtime.counters.get("model_calls", 0),
        "tool_calls": runtime.counters.get("tool_calls", 0),
    }
    return {**payload, "snapshot_sha256": fingerprint(payload)}


def assess_action_progress(before: dict, after: dict, decision: dict,
                           *, step: Any | None = None) -> dict:
    """旧快照不猜测信息增益；新的执行记录单独审计，不生成正向学习信号。"""
    comparable = (before.get("schema_version") == after.get("schema_version") == 2
                  and before.get("goal_revision") == after.get("goal_revision"))
    signals: list[str] = []
    new_observations = []
    new_evidence = []
    verified = []
    if comparable:
        new_observations = sorted(scope for scope, content in after["observations"].items()
                                  if before["observations"].get(scope) != content)
        new_evidence = sorted(set(after["evidence_fingerprints"]) - set(before["evidence_fingerprints"]))
        verified = sorted(set(after["verified_objective_ids"]) - set(before["verified_objective_ids"]))
        if not before.get("goal_defined") and after.get("goal_defined"):
            signals.append("GOAL_DEFINED")
        if verified:
            signals.append("OBJECTIVES_VERIFIED")
        if new_evidence:
            signals.append("EVIDENCE_ADDED")
        if new_observations:
            signals.append("TRUSTED_OBSERVATION_ADDED")
        if before["bindings_sha256"] != after["bindings_sha256"]:
            signals.append("BINDING_REFINED")

    action = decision.get("tool") if decision.get("action") == "tool" else decision.get("action")
    step_status = getattr(step, "status", None)
    step_error = deepcopy(getattr(step, "error", None))
    if step_status in _FAILED or step_error:
        status, reason = "FAILED", "ACTION_FAILED"
    elif decision.get("action") in {"clarify", "handoff"}:
        status, reason = "DEFERRED", "WAITING_FOR_EXTERNAL_INPUT"
    elif not comparable:
        status, reason = "UNKNOWN", "INCOMPARABLE_PROGRESS_SNAPSHOTS"
    elif signals:
        status, reason = "PROGRESSED", "VERIFIABLE_STATE_CHANGE"
    elif decision.get("action") in {"respond", "finish", "unsupported"}:
        status, reason = "UNKNOWN", "TERMINAL_WITHOUT_COMPARABLE_STATE"
    else:
        status, reason = "NO_PROGRESS", "NO_VERIFIABLE_STATE_CHANGE"

    information_gain = ("UNKNOWN" if not comparable or status in {"FAILED", "DEFERRED", "UNKNOWN"}
                        else "NEW_INFORMATION" if new_observations or new_evidence or verified
                        else "NO_NEW_INFORMATION")
    return {
        "schema_version": 2,
        "progress_status": status,
        "progress_reason": reason,
        "learning_eligible": status == "PROGRESSED" and comparable,
        "execution_progress": bool(set(after.get("completed_step_ids") or [])
                                   - set(before.get("completed_step_ids") or [])),
        "information_gain": {
            "status": information_gain,
            "changed_observation_scopes": new_observations,
            "new_evidence_count": len(new_evidence),
            "newly_verified_objective_ids": verified,
        },
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
