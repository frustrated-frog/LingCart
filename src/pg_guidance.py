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
    typed_applicability = isinstance(selection.get("applicability"), dict)
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
            "applicability": deepcopy(edge.get("applicability")),
            "path_score": edge.get("path_score"),
        })
    recommended_actions = list(dict.fromkeys(recommended_actions))

    forbidden_repeats = []
    for item in (trajectory.get("recent_actions") or [])[-1:]:
        identity = item.get("decision_fingerprint")
        if item.get("outcome") not in NO_PROGRESS_OUTCOMES or not isinstance(identity, str):
            continue
        forbidden_repeats.append({
            "decision_fingerprint": identity,
            "raw_decision_fingerprint": item.get("raw_decision_fingerprint"),
            "decision_identity_basis": item.get("decision_identity_basis", "model_decision"),
            "action": item.get("action"),
            "outcome": item.get("outcome"),
            "sequence": item.get("sequence"),
            "reason": "EXACT_NO_PROGRESS_REPEAT",
        })

    basis = {
        "schema_version": 2 if typed_applicability else 1,
        "graph_sha256": graph.fingerprint,
        "localization_anchor": selection.get("anchor"),
        "localization_confidence": selection.get("confidence"),
        "edge_ids": [item["edge_id"] for item in instructions],
        "recommended_actions": recommended_actions,
        "forbidden_repeats": forbidden_repeats,
        "policy": ("STATE_CONDITIONED_ADVISORY_EXCEPT_EXACT_NO_PROGRESS_REPEAT"
                   if typed_applicability else "ADVISORY_EXCEPT_EXACT_NO_PROGRESS_REPEAT"),
    }
    if typed_applicability:
        basis.update(
            applicability_strategy=selection["applicability"].get("strategy"),
            applicability_status=selection["applicability"].get("status"),
            condition_evaluations=deepcopy(selection["applicability"].get("evaluations") or []),
            projected_paths=deepcopy(selection.get("paths") or []),
        )
    preplay = selection.get("preplay") if isinstance(selection.get("preplay"), dict) else None
    if preplay is not None:
        selected = preplay.get("selected_path") if isinstance(preplay.get("selected_path"), dict) else None
        basis["schema_version"] = 3
        basis["preplay"] = {
            "plan_id": preplay.get("plan_id"),
            "method": preplay.get("method"),
            "status": preplay.get("status"),
            "reason": preplay.get("reason"),
            "causal_claim": False,
            "selected_path": ({key: selected.get(key) for key in (
                "path_id", "edge_ids", "node_ids", "first_action", "score",
                "executable_prefix", "future_edges_require_revalidation",
            )} if selected else None),
        }
    return {**basis, "contract_id": fingerprint(basis), "instructions": instructions}


def render_guidance_contract(contract: dict) -> str:
    parts = []
    actions = contract.get("recommended_actions") or []
    if actions:
        qualifier = ("这些后继的受限条件已由可信轨迹证据判定为 TRUE"
                     if contract.get("schema_version") == 2 else "必须结合边条件与当前证据判断是否适用")
        parts.append("程序图候选后继：" + "、".join(actions) + f"；{qualifier}。")
    for item in contract.get("instructions") or []:
        row = f"[{item['edge_id']}] 条件：{item['condition'] or '未声明'}；建议：{item['guidance']}"
        if item.get("pitfalls"):
            row += f"；避免：{item['pitfalls']}"
        applicability = item.get("applicability") or {}
        if applicability.get("status"):
            bindings = applicability.get("evidence_bindings") or []
            row += "；适用性：" + applicability["status"]
            if bindings:
                row += "（" + "，".join(
                    f"{binding.get('fact')}={binding.get('actual')}" for binding in bindings
                ) + "）"
        parts.append(row + "。")
    if contract.get("projected_paths"):
        parts.append("路径中只有第一条边是当前适用建议；后续边必须在状态变化后重新求值，不能提前执行。")
    preplay = contract.get("preplay") if isinstance(contract.get("preplay"), dict) else {}
    selected = preplay.get("selected_path") if isinstance(preplay.get("selected_path"), dict) else {}
    if selected and selected.get("first_action"):
        parts.append(
            "有界程序预演首选当前行动："
            + str(selected.get("first_action") or "无行动")
            + "；后续路径是预测假设，不是事实、授权或完成证据，每次真实观察后必须重新规划。"
        )
    elif selected and preplay.get("status") == "TERMINAL":
        parts.append("有界程序预演仅到达终态，不生成额外行动建议；任务完成仍以独立目标核验为准。")
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
