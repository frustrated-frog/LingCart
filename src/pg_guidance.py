"""把冻结程序图编译为可审计的决策契约，不额外调用模型。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.procedural_graph import fingerprint


NO_PROGRESS_OUTCOMES = {"EMPTY", "ERROR"}


def _record(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def decision_action(decision: dict) -> str | None:
    payload = _record(decision)
    action = payload.get("action")
    if action == "tool":
        tool = payload.get("tool")
        return tool if isinstance(tool, str) and tool else None
    return action if isinstance(action, str) and action else None


def decision_fingerprint(decision: dict) -> str:
    """指纹覆盖真实决策参数；只用于阻止完全相同的无进展重放。"""
    payload = _record(decision)
    action = payload.get("action")
    if action == "tool":
        basis = {"action": action, "tool": payload.get("tool"), "args": _record(payload.get("args"))}
    elif action == "inspect":
        basis = {"action": action, "evidence_ref": payload.get("evidence_ref"),
                 "offset": payload.get("offset", 0), "limit": payload.get("limit", 1)}
    elif action == "recommend":
        basis = {"action": action, "product_id": payload.get("product_id"),
                 "objective_id": payload.get("objective_id"),
                 "evidence_refs": list(payload.get("evidence_refs") or [])}
    elif action == "complete":
        basis = {"action": action, "claims": list(payload.get("claims") or [])}
    else:
        basis = {"action": action}
    return fingerprint(basis)


def compile_guidance_contract(graph, selection: dict, trajectory: dict) -> dict:
    """将局部子图编译为候选后继与无进展重放约束。

    图边的自然语言 condition 仍是供求解模型判断的数据，不在内核中执行。
    唯一硬阻断是已有真实 EMPTY/ERROR 结果的完全相同决策指纹。
    """
    nodes = {node.id: node for node in graph.nodes}
    instructions = []
    recommended_actions = []
    for edge in selection.get("edges") or []:
        target = nodes.get(edge.get("target"))
        if target is None or target.kind != "ACTION":
            continue
        depth = edge.get("retrieval_depth", 1)
        if depth == 1:
            recommended_actions.append(target.id)
        instructions.append({
            "edge_id": edge["id"],
            "source": edge["source"],
            "target": edge["target"],
            "retrieval_depth": depth,
            "condition": edge.get("condition") or "",
            "guidance": edge.get("guidance") or "",
            "pitfalls": edge.get("pitfalls") or "",
        })
    recommended_actions = list(dict.fromkeys(recommended_actions))

    forbidden_repeats = []
    for item in (trajectory.get("recent_actions") or [])[-1:]:
        identity = item.get("decision_fingerprint")
        if item.get("outcome") not in NO_PROGRESS_OUTCOMES or not isinstance(identity, str):
            continue
        forbidden_repeats.append({
            "decision_fingerprint": identity,
            "action": item.get("action"),
            "outcome": item.get("outcome"),
            "sequence": item.get("sequence"),
            "reason": "EXACT_NO_PROGRESS_REPEAT",
        })

    basis = {
        "schema_version": 1,
        "graph_sha256": graph.fingerprint,
        "localization_anchor": selection.get("anchor"),
        "localization_confidence": selection.get("confidence"),
        "edge_ids": [item["edge_id"] for item in instructions],
        "recommended_actions": recommended_actions,
        "forbidden_repeats": forbidden_repeats,
        "policy": "ADVISORY_EXCEPT_EXACT_NO_PROGRESS_REPEAT",
    }
    return {**basis, "contract_id": fingerprint(basis), "instructions": instructions}


def render_guidance_contract(contract: dict) -> str:
    parts = []
    actions = contract.get("recommended_actions") or []
    if actions:
        parts.append("程序图候选后继：" + "、".join(actions) + "；必须结合边条件与当前证据判断是否适用。")
    for item in contract.get("instructions") or []:
        row = f"[{item['edge_id']}] 条件：{item['condition'] or '未声明'}；建议：{item['guidance']}"
        if item.get("pitfalls"):
            row += f"；避免：{item['pitfalls']}"
        parts.append(row + "。")
    if contract.get("forbidden_repeats"):
        parts.append("不得原样重复已返回空结果或错误的相同决策；可以改变有依据的参数、选择其他目标或如实停止。")
    return "\n".join(parts)


def review_guidance_contract(contract: dict, decision: dict) -> dict:
    actual = decision_action(decision)
    identity = decision_fingerprint(decision)
    forbidden = {item.get("decision_fingerprint"): item
                 for item in contract.get("forbidden_repeats") or []}
    if identity in forbidden:
        return {
            "status": "BLOCKED",
            "blocked": True,
            "reason": "EXACT_NO_PROGRESS_REPEAT",
            "actual_action": actual,
            "actual_decision_fingerprint": identity,
            "recommended_actions": list(contract.get("recommended_actions") or []),
            "matched_edge_ids": [],
            "blocked_by": deepcopy(forbidden[identity]),
        }
    recommendations = list(contract.get("recommended_actions") or [])
    instructions = contract.get("instructions") or []
    matched = [item["edge_id"] for item in instructions
               if item.get("retrieval_depth", 1) == 1 and item.get("target") == actual]
    status = "ALIGNED" if matched else "DIVERGED" if recommendations else "UNKNOWN"
    return {
        "status": status,
        "blocked": False,
        "reason": "RECOMMENDED_ACTION" if matched else "ADVISORY_DIVERGENCE" if recommendations else "NO_ACTIONABLE_EDGE",
        "actual_action": actual,
        "actual_decision_fingerprint": identity,
        "recommended_actions": recommendations,
        "matched_edge_ids": matched,
        "blocked_by": None,
    }
