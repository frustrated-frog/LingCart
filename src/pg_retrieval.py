"""基于持久轨迹状态的程序图定位与有界局部子图检索。"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any

from src.pg_guidance import decision_fingerprint


DONE_STATUSES = {"SUCCEEDED", "REUSED", "REJECTED", "FAILED", "SKIPPED", "CANCELLED"}
FAILED_STATUSES = {"FAILED", "REJECTED", "SKIPPED", "CANCELLED"}


def _record(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _result_outcome(step) -> str:
    if step is None:
        return "UNKNOWN"
    if getattr(step, "status", None) in FAILED_STATUSES or getattr(step, "error", None):
        return "ERROR"
    result = getattr(step, "result", None)
    if isinstance(result, dict):
        if result.get("error") or result.get("success") is False:
            return "ERROR"
        if result.get("products") == []:
            return "EMPTY"
        if isinstance(result.get("products"), list):
            return "NON_EMPTY"
    if result is not None:
        return "SUCCESS"
    return "PENDING" if getattr(step, "status", None) not in DONE_STATUSES else "UNKNOWN"


def build_trajectory_state(state: dict, plan, events: list[dict], *, recent_limit: int = 6) -> dict:
    """把 Runtime、目标账本和计划投影为无业务关键词的稳定定位输入。"""
    revision = state.get("goal_revision", 0)
    steps = {step.step_id: step for step in plan.steps}
    actions = []
    for event in events:
        if event.get("type") != "adaptive_decision" or event.get("goal_revision", 0) != revision:
            continue
        decision = _record(event.get("decision"))
        action = decision.get("tool") if decision.get("action") == "tool" else decision.get("action")
        if not isinstance(action, str) or not action:
            continue
        step = steps.get(event.get("step_id"))
        actions.append({
            "action": action,
            "sequence": event.get("sequence"),
            "step_id": event.get("step_id"),
            "step_status": getattr(step, "status", None),
            "outcome": _result_outcome(step),
            "decision_fingerprint": decision_fingerprint(decision),
        })

    goal = _record(state.get("goal"))
    objective_kinds = {item.get("id"): item.get("kind") for item in goal.get("objectives", [])
                       if isinstance(item, dict)}
    objectives = []
    for row in state.get("objective_ledger", []):
        if not isinstance(row, dict):
            continue
        objectives.append({
            "objective_id": row.get("objective_id"),
            "kind": objective_kinds.get(row.get("objective_id")),
            "status": row.get("status"),
            "evidence_count": len(row.get("evidence_refs") or []),
            "failure_codes": list(row.get("failures") or [])[:8],
        })
    pending_steps = [{"step_id": step.step_id, "tool": step.tool, "status": step.status,
                      "objective_id": step.objective_id}
                     for step in plan.steps if step.status not in DONE_STATUSES]
    unresolved = [item for item in objectives if item["status"] != "VERIFIED"]
    feedback = _record(state.get("feedback"))
    snapshot = {
        "schema_version": 1,
        "goal_revision": revision,
        "binding_revision": state.get("binding_revision", 0),
        "goal_defined": bool(goal),
        "goal_objective_count": len(goal.get("objectives", [])),
        "objectives": objectives,
        "unresolved_objective_kinds": sorted({item["kind"] for item in unresolved if item.get("kind")}),
        "all_objectives_verified": bool(objectives) and not unresolved,
        "feedback_code": feedback.get("code"),
        "recent_actions": actions[-recent_limit:],
        "pending_steps": pending_steps[:12],
    }
    # 保留 v1 证据字段，旧 UI 和历史投影无需猜测迁移语义。
    snapshot["unfinished_plan_steps"] = deepcopy(snapshot["pending_steps"])
    return snapshot


def _distances_to_targets(graph, targets: set[str]) -> dict[str, int]:
    incoming: dict[str, list[str]] = {}
    for edge in graph.edges:
        incoming.setdefault(edge.target, []).append(edge.source)
    distances = {target: 0 for target in targets}
    queue = deque(targets)
    while queue:
        target = queue.popleft()
        for source in incoming.get(target, []):
            distance = distances[target] + 1
            if distance <= 4 and (source not in distances or distance < distances[source]):
                distances[source] = distance
                queue.append(source)
    return distances


def retrieve_state_aware(graph, trajectory: dict, *, hops: int = 2, max_edges: int = 6,
                         min_score: float = 0.55) -> dict:
    """定位一个可解释锚点；不确定时弃权，绝不回退整张图。"""
    if not 1 <= hops <= 3:
        raise ValueError("局部图跳数必须为 1 至 3")
    if not 1 <= max_edges <= 16:
        raise ValueError("局部图边数必须为 1 至 16")
    if not 0.0 < min_score <= 1.0:
        raise ValueError("定位阈值必须位于 0 到 1 之间")

    nodes = {node.id: node for node in graph.nodes}
    node_order = {node.id: index for index, node in enumerate(graph.nodes)}
    scores: dict[str, float] = {}
    components: dict[str, list[dict]] = {}

    def add(node_id: str, value: float, reason: str):
        if node_id not in nodes or value <= 0:
            return
        scores[node_id] = scores.get(node_id, 0.0) + value
        components.setdefault(node_id, []).append({"signal": reason, "weight": round(value, 4)})

    goal_defined = bool(trajectory.get("goal_defined"))
    actions = list(trajectory.get("recent_actions") or [])
    open_kinds = set(trajectory.get("unresolved_objective_kinds") or [])
    target_nodes = open_kinds & set(nodes)
    target_distances = _distances_to_targets(graph, target_nodes)
    graph_covers_open_objective = not open_kinds or bool(target_nodes)

    if not goal_defined:
        add("Start", 1.2, "goal_missing")
    elif not actions and graph_covers_open_objective:
        add("set_goal" if "set_goal" in nodes else "Start", 1.15, "goal_defined_without_action")

    if trajectory.get("all_objectives_verified"):
        add("complete", 1.2, "all_objectives_verified")

    latest_action = actions[-1].get("action") if actions else None
    latest_known = latest_action in nodes
    for distance, item in enumerate(reversed(actions)):
        action = item.get("action")
        if action not in nodes:
            continue
        reaches_open_goal = action in target_distances
        recovering = item.get("outcome") in {"EMPTY", "ERROR"}
        if not graph_covers_open_objective:
            continue
        # 未知动作之后不能仅凭久远历史猜锚点；只有仍关联未完成目标或明确失败恢复时保留。
        if distance and not latest_known and not reaches_open_goal and not recovering:
            continue
        add(action, max(0.2, 1.0 - distance * 0.18), f"recent_action:{distance}")
        if recovering:
            add(action, 0.3, f"recover_from:{item.get('outcome')}")

    for item in trajectory.get("pending_steps") or []:
        if graph_covers_open_objective:
            add(item.get("tool"), 0.55, "pending_plan_step")

    for target in target_nodes:
        add(target, 0.5, "unresolved_objective")
    for node_id, distance in target_distances.items():
        if distance:
            add(node_id, 0.36 / distance, f"path_to_unresolved_objective:{distance}")

    candidates = [{
        "node_id": node_id,
        "score": round(score, 4),
        "confidence": round(min(1.0, score / 1.2), 4),
        "components": components[node_id],
    } for node_id, score in sorted(scores.items(), key=lambda item: (-item[1], node_order[item[0]]))]
    selected = candidates[0] if candidates else None
    if selected is None or selected["confidence"] < min_score:
        return {
            "algorithm": "trajectory-state-v2",
            "status": "ABSTAINED",
            "reason": "NO_CANDIDATE" if selected is None else "LOW_CONFIDENCE",
            "anchor": None,
            "confidence": selected["confidence"] if selected else 0.0,
            "fallback": False,
            "abstained": True,
            "hops": hops,
            "nodes": [],
            "edges": [],
            "candidates": candidates[:8],
            "excluded_edge_ids": [],
        }

    anchor = selected["node_id"]
    reached, frontier = {anchor}, [anchor]
    chosen, depth_by_edge = [], {}
    for depth in range(1, hops + 1):
        outgoing = [edge for edge in graph.edges if edge.source in frontier and edge.id not in depth_by_edge]
        outgoing.sort(key=lambda edge: (-scores.get(edge.target, 0.0), node_order[edge.target], edge.id))
        next_frontier = []
        for edge in outgoing:
            if len(chosen) >= max_edges:
                break
            chosen.append(edge)
            depth_by_edge[edge.id] = depth
            if edge.target not in reached:
                reached.add(edge.target)
                next_frontier.append(edge.target)
        frontier = next_frontier
        if not frontier or len(chosen) >= max_edges:
            break
    chosen_ids = {edge.id for edge in chosen}
    reachable_edges = {edge.id for edge in graph.edges if edge.source in reached}
    return {
        "algorithm": "trajectory-state-v2",
        "status": "LOCALIZED",
        "reason": selected["components"][0]["signal"],
        "anchor": anchor,
        "confidence": selected["confidence"],
        "fallback": False,
        "abstained": False,
        "hops": hops,
        "nodes": [node.model_dump() for node in graph.nodes if node.id in reached],
        "edges": [dict(edge.model_dump(), retrieval_depth=depth_by_edge[edge.id]) for edge in chosen],
        "candidates": candidates[:8],
        "excluded_edge_ids": sorted(reachable_edges - chosen_ids),
    }


def shadow_action_anchor(graph, trajectory: dict, *, hops: int = 2) -> dict:
    """只记录旧定位器会做什么；结果永不进入 v2 的决策上下文。"""
    actions = trajectory.get("recent_actions") or []
    anchor = actions[-1].get("action") if actions else "Start"
    selection = graph.retrieve(anchor, hops=hops)
    return {
        "algorithm": "action-anchor-v1",
        "anchor": selection["anchor"],
        "fallback": selection["fallback"],
        "node_ids": [item["id"] for item in selection["nodes"]],
        "edge_ids": [item["id"] for item in selection["edges"]],
    }


def bounded_trajectory_snapshot(trajectory: dict) -> dict:
    """调用方持久化前取得副本，避免定位过程持有可变运行状态。"""
    return deepcopy(trajectory)
