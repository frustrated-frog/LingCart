"""PG V6 有界程序预演；预测只生成候选行动假设，不修改权威状态。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import Field

from src.pg_conditions import evaluate_edge_condition
from src.procedural_graph import FrozenModel, fingerprint


EffectName = Literal[
    "GOAL_DEFINITION",
    "INFORMATION_GAIN",
    "OBJECTIVE_PROGRESS",
    "EXTERNAL_INPUT",
    "TASK_TERMINATION",
    "UNKNOWN_EFFECT",
]


class TransitionPrediction(FrozenModel):
    edge_id: str
    source: str
    target: str
    action: str | None
    state_basis: Literal["CURRENT_VERIFIED", "PREDICTED_ACTION_ONLY"]
    condition_status: Literal["TRUE", "FALSE", "UNKNOWN"]
    condition_reason: str
    effect_hypotheses: tuple[EffectName, ...] = Field(min_length=1, max_length=4)
    reliability_score: float = Field(ge=0, le=1)
    support_count: int = Field(ge=0)
    counter_count: int = Field(ge=0)
    utility_components: dict[str, float]
    projected_state_sha256: str


class PreplayPath(FrozenModel):
    path_id: str
    edge_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    node_ids: tuple[str, ...] = Field(min_length=2, max_length=5)
    first_action: str | None
    score: float
    executable_prefix: Literal[1] = 1
    future_edges_require_revalidation: tuple[str, ...] = ()
    predictions: tuple[TransitionPrediction, ...] = Field(min_length=1, max_length=4)


_ACTION_MODELS: dict[str, dict[str, Any]] = {
    "set_goal": dict(effects=("GOAL_DEFINITION",), information=0.35, goal=0.45, cost=0.20, uncertainty=0.20),
    "search_products": dict(effects=("INFORMATION_GAIN",), information=0.90, goal=0.25, cost=1.00, uncertainty=0.40),
    "discover_products": dict(effects=("INFORMATION_GAIN",), information=0.90, goal=0.25, cost=1.00, uncertainty=0.40),
    "compare_products": dict(effects=("INFORMATION_GAIN",), information=0.75, goal=0.35, cost=0.80, uncertainty=0.30),
    "inspect": dict(effects=("INFORMATION_GAIN",), information=0.70, goal=0.30, cost=0.50, uncertainty=0.25),
    "answer_faq": dict(effects=("INFORMATION_GAIN",), information=0.65, goal=0.30, cost=0.70, uncertainty=0.30),
    "recommend": dict(effects=("OBJECTIVE_PROGRESS",), information=0.15, goal=0.90, cost=0.35, uncertainty=0.25),
    "complete": dict(effects=("OBJECTIVE_PROGRESS", "TASK_TERMINATION"), information=0.00, goal=1.00, cost=0.10, uncertainty=0.15),
    "finish": dict(effects=("TASK_TERMINATION",), information=0.00, goal=0.80, cost=0.10, uncertainty=0.15),
    "clarify": dict(effects=("EXTERNAL_INPUT",), information=0.20, goal=0.20, cost=0.20, uncertainty=0.20),
    "End": dict(effects=("TASK_TERMINATION",), information=0.00, goal=0.75, cost=0.00, uncertainty=0.10),
}
_UNKNOWN_ACTION = dict(
    effects=("UNKNOWN_EFFECT",), information=0.00, goal=0.00, cost=0.50, uncertainty=0.80,
)
_WEIGHTS = {
    "goal_progress": 1.0,
    "information_gain": 0.7,
    "reliability": 0.4,
    "cost": -0.3,
    "uncertainty": -0.6,
}


def _action_model(action: str) -> dict[str, Any]:
    return _ACTION_MODELS.get(action, _UNKNOWN_ACTION)


def _edge_reliability(trajectory: dict, edge_id: str) -> tuple[float, int, int]:
    history = trajectory.get("edge_feedback") if isinstance(trajectory.get("edge_feedback"), dict) else {}
    row = history.get(edge_id) if isinstance(history.get(edge_id), dict) else {}
    support = max(0, int(row.get("progressed", 0)))
    counter = max(0, int(row.get("no_progress", 0))) + max(0, int(row.get("failed", 0)))
    # Laplace 平滑只是有界排序特征，不解释为成功概率或因果效应。
    return (support + 1) / (support + counter + 2), support, counter


def _project_action(trajectory: dict, action: str) -> dict:
    """只投影“将尝试某行动”；结果保持 UNKNOWN，且不提升目标或证据状态。"""
    projected = deepcopy(trajectory)
    recent = list(projected.get("recent_actions") or [])
    recent.append({"action": action, "outcome": "UNKNOWN", "prediction_only": True})
    projected["recent_actions"] = recent[-6:]
    return projected


def _predict_transition(edge, trajectory: dict, *, depth: int) -> tuple[TransitionPrediction, dict, float]:
    evaluation = evaluate_edge_condition(edge.applicability, trajectory)
    action = edge.target
    model = _action_model(action)
    reliability, support, counter = _edge_reliability(trajectory, edge.id)
    condition_uncertainty = 0.0 if evaluation["status"] == "TRUE" else 0.45
    evidence_uncertainty = 0.25 / (support + counter + 1)
    uncertainty = min(1.0, float(model["uncertainty"]) + condition_uncertainty + evidence_uncertainty)
    components = {
        "goal_progress": float(model["goal"]),
        "information_gain": float(model["information"]),
        "reliability": reliability,
        "cost": float(model["cost"]),
        "uncertainty": uncertainty,
    }
    increment = sum(components[name] * weight for name, weight in _WEIGHTS.items())
    projected = _project_action(trajectory, action)
    prediction = TransitionPrediction(
        edge_id=edge.id,
        source=edge.source,
        target=edge.target,
        action=None if action == "End" else action,
        state_basis="CURRENT_VERIFIED" if depth == 1 else "PREDICTED_ACTION_ONLY",
        condition_status=evaluation["status"],
        condition_reason=evaluation["reason"],
        effect_hypotheses=tuple(model["effects"]),
        reliability_score=round(reliability, 6),
        support_count=support,
        counter_count=counter,
        utility_components={key: round(value, 6) for key, value in components.items()},
        projected_state_sha256=fingerprint(projected),
    )
    return prediction, projected, increment


def _path_payload(item: dict) -> PreplayPath:
    edge_ids = tuple(prediction.edge_id for prediction in item["predictions"])
    node_ids = (item["predictions"][0].source, *(prediction.target for prediction in item["predictions"]))
    basis = {
        "edge_ids": edge_ids,
        "state_sha256": item["state_sha256"],
        "method": "bounded-transition-search-v1",
    }
    return PreplayPath(
        path_id=fingerprint(basis)[:20],
        edge_ids=edge_ids,
        node_ids=tuple(node_ids),
        first_action=item["predictions"][0].action,
        score=round(item["score"], 6),
        future_edges_require_revalidation=edge_ids[1:],
        predictions=tuple(item["predictions"]),
    )


def plan_preplay(graph, selection: dict, trajectory: dict, *, max_depth: int = 3,
                 beam_width: int = 4, max_paths: int = 6) -> dict:
    """从当前已证明适用的一跳边开始，有限预演未来路径但只推荐第一步。"""
    if not 1 <= max_depth <= 4:
        raise ValueError("预演深度必须位于 1 到 4 之间")
    if not 1 <= beam_width <= 8:
        raise ValueError("预演 Beam 宽度必须位于 1 到 8 之间")
    if not 1 <= max_paths <= 12:
        raise ValueError("预演路径上限必须位于 1 到 12 之间")
    state_sha256 = fingerprint(trajectory)
    graph_edges = {edge.id: edge for edge in graph.edges}
    selected_ids = [row.get("id") for row in selection.get("edges") or []]
    initial = []
    for edge_id in selected_ids:
        edge = graph_edges.get(edge_id)
        if edge is None:
            continue
        prediction, projected, increment = _predict_transition(edge, trajectory, depth=1)
        if prediction.condition_status != "TRUE":
            continue
        initial.append({
            "predictions": [prediction],
            "projected": projected,
            "score": increment,
            "state_sha256": state_sha256,
        })
    initial.sort(key=lambda item: (-item["score"], [p.edge_id for p in item["predictions"]]))
    frontier = initial[:beam_width]
    for depth in range(2, max_depth + 1):
        expanded = []
        for item in frontier:
            tail = item["predictions"][-1].target
            visited = {prediction.edge_id for prediction in item["predictions"]}
            outgoing = sorted((edge for edge in graph.edges
                               if edge.source == tail and edge.id not in visited), key=lambda edge: edge.id)
            viable = False
            for edge in outgoing:
                prediction, projected, increment = _predict_transition(edge, item["projected"], depth=depth)
                if prediction.condition_status == "FALSE":
                    continue
                viable = True
                expanded.append({
                    "predictions": [*item["predictions"], prediction],
                    "projected": projected,
                    "score": item["score"] + (0.7 ** (depth - 1)) * increment,
                    "state_sha256": state_sha256,
                })
            if not viable:
                expanded.append(item)
        expanded.sort(key=lambda item: (-item["score"], [p.edge_id for p in item["predictions"]]))
        frontier = expanded[:beam_width]
    paths = tuple(_path_payload(item) for item in frontier[:max_paths])
    selected = paths[0] if paths else None
    terminal_only = bool(selected and selected.first_action is None and selected.node_ids[-1] == "End")
    basis = {
        "schema_version": 1,
        "method": "bounded-transition-search-v1",
        "graph_sha256": graph.fingerprint,
        "state_sha256": state_sha256,
        "max_depth": max_depth,
        "beam_width": beam_width,
        "max_paths": max_paths,
        "selected_path_id": selected.path_id if selected else None,
        "path_ids": [path.path_id for path in paths],
    }
    return {
        **basis,
        "plan_id": fingerprint(basis),
        "status": "TERMINAL" if terminal_only else "READY" if selected else "ABSTAINED",
        "reason": ("TERMINAL_TRANSITION_ONLY" if terminal_only else "BOUNDED_PATH_SELECTED"
                   if selected else "NO_PROVEN_FIRST_TRANSITION"),
        "causal_claim": False,
        "probabilistic_claim": False,
        "weights": deepcopy(_WEIGHTS),
        "selected_path": selected.model_dump(mode="json") if selected else None,
        "paths": [path.model_dump(mode="json") for path in paths],
    }


def compare_preplay_prediction(preplay: dict, assessment: dict, *, actual_action: str | None) -> dict:
    """把第一步预测与真实结果对齐；只生成关联证据，不回写权威业务状态。"""
    selected = preplay.get("selected_path") if isinstance(preplay.get("selected_path"), dict) else {}
    predictions = selected.get("predictions") if isinstance(selected.get("predictions"), list) else []
    first = predictions[0] if predictions else {}
    expected_action = first.get("action")
    effects = list(first.get("effect_hypotheses") or [])
    action_match = bool(expected_action) and expected_action == actual_action
    rows = []
    information = assessment.get("information_gain") if isinstance(assessment.get("information_gain"), dict) else {}
    signals = set(assessment.get("signals") or [])
    for effect in effects:
        if not action_match:
            status, reason = "UNKNOWN", "ACTION_DIVERGED"
        elif effect == "INFORMATION_GAIN":
            value = information.get("status")
            status = "SUPPORT" if value == "NEW_INFORMATION" else "COUNTER" if value == "NO_NEW_INFORMATION" else "UNKNOWN"
            reason = str(value or "OUTCOME_UNKNOWN")
        elif effect == "OBJECTIVE_PROGRESS":
            verified = information.get("newly_verified_objective_ids") or []
            status = "SUPPORT" if verified else (
                "COUNTER" if assessment.get("progress_status") == "NO_PROGRESS" else "UNKNOWN")
            reason = "OBJECTIVES_VERIFIED" if verified else str(assessment.get("progress_reason") or "OUTCOME_UNKNOWN")
        elif effect == "GOAL_DEFINITION":
            status = "SUPPORT" if "GOAL_DEFINED" in signals else (
                "COUNTER" if assessment.get("progress_status") == "NO_PROGRESS" else "UNKNOWN")
            reason = "GOAL_DEFINED" if status == "SUPPORT" else str(assessment.get("progress_reason") or "OUTCOME_UNKNOWN")
        elif effect == "EXTERNAL_INPUT":
            status = "SUPPORT" if assessment.get("progress_status") == "DEFERRED" else "UNKNOWN"
            reason = str(assessment.get("progress_reason") or "OUTCOME_UNKNOWN")
        elif effect == "TASK_TERMINATION":
            status, reason = "UNKNOWN", "TERMINAL_RESULT_REQUIRES_VERIFICATION_EVENT"
        else:
            status, reason = "UNKNOWN", "UNMODELED_EFFECT"
        rows.append({"effect": effect, "status": status, "reason": reason})
    statuses = {row["status"] for row in rows}
    status = ("DIVERGED" if expected_action and not action_match else "COUNTER" if "COUNTER" in statuses
              else "SUPPORT" if "SUPPORT" in statuses else "UNKNOWN")
    basis = {
        "preplay_plan_id": preplay.get("plan_id"),
        "expected_action": expected_action,
        "actual_action": actual_action,
        "effects": rows,
    }
    return {
        **basis,
        "comparison_id": fingerprint(basis),
        "status": status,
        "action_match": action_match,
        "causal_claim": False,
    }
