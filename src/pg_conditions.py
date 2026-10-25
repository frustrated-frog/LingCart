"""PG 程序边的受限条件协议、三值求值与有界路径排序。"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TruthValue = Literal["TRUE", "FALSE", "UNKNOWN"]
FactName = Literal[
    "goal.defined",
    "goal.all_verified",
    "goal.unresolved_kinds",
    "goal.unresolved_count",
    "goal.evidence_count",
    "trajectory.latest_action",
    "trajectory.latest_outcome",
    "trajectory.actions",
    "plan.pending_tools",
    "runtime.feedback_code",
    "budget.model_calls_remaining",
    "budget.tool_calls_remaining",
]
ConditionOperator = Literal[
    "EQ", "NE", "IN", "CONTAINS", "NOT_CONTAINS", "EMPTY", "NOT_EMPTY", "GTE", "LTE", "EXISTS",
]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


_FACT_KINDS = {
    "goal.defined": "boolean",
    "goal.all_verified": "boolean",
    "goal.unresolved_kinds": "collection",
    "goal.unresolved_count": "integer",
    "goal.evidence_count": "integer",
    "trajectory.latest_action": "text",
    "trajectory.latest_outcome": "text",
    "trajectory.actions": "collection",
    "plan.pending_tools": "collection",
    "runtime.feedback_code": "text",
    "budget.model_calls_remaining": "integer",
    "budget.tool_calls_remaining": "integer",
}
_ALLOWED_OPERATORS = {
    "boolean": {"EQ", "NE"},
    "text": {"EQ", "NE", "IN", "EXISTS"},
    "collection": {"CONTAINS", "NOT_CONTAINS", "EMPTY", "NOT_EMPTY"},
    "integer": {"EQ", "GTE", "LTE"},
}


class ConditionClause(_FrozenModel):
    fact: FactName
    operator: ConditionOperator
    value: str | bool | int | tuple[str, ...] | None = None

    @model_validator(mode="after")
    def valid_operation(self):
        kind = _FACT_KINDS[self.fact]
        if self.operator not in _ALLOWED_OPERATORS[kind]:
            raise ValueError(f"事实 {self.fact} 不支持操作符 {self.operator}")
        if self.operator in {"EMPTY", "NOT_EMPTY", "EXISTS"}:
            if self.value is not None:
                raise ValueError(f"操作符 {self.operator} 不接受 value")
            return self
        if kind == "boolean" and type(self.value) is not bool:
            raise ValueError("布尔条件必须使用布尔值")
        if kind == "integer" and (type(self.value) is not int or self.value < 0):
            raise ValueError("整数条件必须使用非负整数")
        if self.operator == "IN" and (not isinstance(self.value, tuple) or not self.value):
            raise ValueError("IN 条件必须使用非空字符串数组")
        if kind in {"text", "collection"} and self.operator != "IN" and (
            not isinstance(self.value, str) or not self.value
        ):
            raise ValueError("文本或集合条件必须使用非空字符串")
        return self


class EdgeCondition(_FrozenModel):
    """有限的合取/析取条件；不接受代码、路径或任意表达式。"""

    all_of: tuple[ConditionClause, ...] = Field(default=(), max_length=8)
    any_of: tuple[ConditionClause, ...] = Field(default=(), max_length=8)

    @model_validator(mode="after")
    def nonempty_and_unique(self):
        clauses = self.all_of + self.any_of
        if not clauses:
            raise ValueError("程序边条件至少包含一个受限子句")
        identities = {(item.fact, item.operator, repr(item.value)) for item in clauses}
        if len(identities) != len(clauses):
            raise ValueError("程序边条件不能包含重复子句")
        return self


_UNKNOWN = object()
_FACT_SOURCES = {
    "goal.defined": "trajectory_state.goal_defined",
    "goal.all_verified": "trajectory_state.all_objectives_verified",
    "goal.unresolved_kinds": "trajectory_state.unresolved_objective_kinds",
    "goal.unresolved_count": "trajectory_state.objectives",
    "goal.evidence_count": "trajectory_state.objectives",
    "trajectory.latest_action": "trajectory_state.recent_actions[-1].action",
    "trajectory.latest_outcome": "trajectory_state.recent_actions[-1].outcome",
    "trajectory.actions": "trajectory_state.recent_actions",
    "plan.pending_tools": "trajectory_state.pending_steps",
    "runtime.feedback_code": "trajectory_state.feedback_code",
    "budget.model_calls_remaining": "trajectory_state.budget.model_calls_remaining",
    "budget.tool_calls_remaining": "trajectory_state.budget.tool_calls_remaining",
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _facts(trajectory: dict) -> dict[str, Any]:
    actions = trajectory.get("recent_actions")
    latest = actions[-1] if isinstance(actions, list) and actions else None
    objectives = trajectory.get("objectives")
    pending = trajectory.get("pending_steps")
    budget = trajectory.get("budget")
    facts: dict[str, Any] = {
        "goal.defined": trajectory.get("goal_defined", _UNKNOWN),
        "goal.all_verified": trajectory.get("all_objectives_verified", _UNKNOWN),
        "goal.unresolved_kinds": tuple(trajectory.get("unresolved_objective_kinds") or ())
            if "unresolved_objective_kinds" in trajectory else _UNKNOWN,
        "goal.unresolved_count": sum(item.get("status") != "VERIFIED" for item in objectives)
            if isinstance(objectives, list) else _UNKNOWN,
        "goal.evidence_count": sum(len(item.get("evidence_refs") or ()) for item in objectives)
            if isinstance(objectives, list) else _UNKNOWN,
        "trajectory.latest_action": latest.get("action") if isinstance(latest, dict) else None,
        "trajectory.latest_outcome": latest.get("outcome") if isinstance(latest, dict) else None,
        "trajectory.actions": tuple(item.get("action") for item in actions if isinstance(item, dict) and item.get("action"))
            if isinstance(actions, list) else _UNKNOWN,
        "plan.pending_tools": tuple(item.get("tool") for item in pending if isinstance(item, dict) and item.get("tool"))
            if isinstance(pending, list) else _UNKNOWN,
        "runtime.feedback_code": trajectory.get("feedback_code") if "feedback_code" in trajectory else _UNKNOWN,
        "budget.model_calls_remaining": budget.get("model_calls_remaining", _UNKNOWN)
            if isinstance(budget, dict) else _UNKNOWN,
        "budget.tool_calls_remaining": budget.get("tool_calls_remaining", _UNKNOWN)
            if isinstance(budget, dict) else _UNKNOWN,
    }
    return facts


def _evaluate_clause(clause: ConditionClause, facts: dict[str, Any]) -> dict:
    actual = facts.get(clause.fact, _UNKNOWN)
    if actual is _UNKNOWN or (clause.fact == "trajectory.latest_outcome" and actual == "UNKNOWN"):
        status: TruthValue = "UNKNOWN"
        rendered_actual = None
    else:
        rendered_actual = list(actual) if isinstance(actual, tuple) else actual
        expected = clause.value
        if clause.operator == "EQ":
            passed = actual == expected
        elif clause.operator == "NE":
            passed = actual != expected
        elif clause.operator == "IN":
            passed = actual in (expected or ())
        elif clause.operator == "CONTAINS":
            passed = expected in actual
        elif clause.operator == "NOT_CONTAINS":
            passed = expected not in actual
        elif clause.operator == "EMPTY":
            passed = len(actual) == 0
        elif clause.operator == "NOT_EMPTY":
            passed = len(actual) > 0
        elif clause.operator == "GTE":
            passed = actual >= expected
        elif clause.operator == "LTE":
            passed = actual <= expected
        elif clause.operator == "EXISTS":
            passed = actual is not None
        else:  # pragma: no cover - Pydantic 已封闭操作符集合
            raise AssertionError(clause.operator)
        status = "TRUE" if passed else "FALSE"
    return {
        "fact": clause.fact,
        "operator": clause.operator,
        "expected": list(clause.value) if isinstance(clause.value, tuple) else clause.value,
        "actual": rendered_actual,
        "status": status,
        "evidence_source": _FACT_SOURCES[clause.fact],
    }


def _and(values: list[TruthValue]) -> TruthValue:
    if "FALSE" in values:
        return "FALSE"
    if "UNKNOWN" in values:
        return "UNKNOWN"
    return "TRUE"


def _or(values: list[TruthValue]) -> TruthValue:
    if "TRUE" in values:
        return "TRUE"
    if "UNKNOWN" in values:
        return "UNKNOWN"
    return "FALSE"


def evaluate_edge_condition(condition: EdgeCondition | None, trajectory: dict) -> dict:
    if condition is None:
        return {
            "schema_version": 1,
            "status": "UNKNOWN",
            "reason": "UNTYPED_CONDITION",
            "condition_sha256": None,
            "all_of": [],
            "any_of": [],
            "evidence_bindings": [],
        }
    facts = _facts(trajectory)
    all_rows = [_evaluate_clause(item, facts) for item in condition.all_of]
    any_rows = [_evaluate_clause(item, facts) for item in condition.any_of]
    groups: list[TruthValue] = []
    if all_rows:
        groups.append(_and([item["status"] for item in all_rows]))
    if any_rows:
        groups.append(_or([item["status"] for item in any_rows]))
    status = _and(groups)
    rows = all_rows + any_rows
    return {
        "schema_version": 1,
        "status": status,
        "reason": "CONDITION_SATISFIED" if status == "TRUE" else (
            "CONDITION_NOT_SATISFIED" if status == "FALSE" else "CONDITION_EVIDENCE_UNKNOWN"
        ),
        "condition_sha256": _fingerprint(condition.model_dump(mode="json")),
        "all_of": all_rows,
        "any_of": any_rows,
        "evidence_bindings": [
            {"fact": item["fact"], "actual": item["actual"], "source": item["evidence_source"]}
            for item in rows
        ],
    }


_ACTION_COST = {
    "search_products": 1.0,
    "discover_products": 1.0,
    "compare_products": 1.0,
    "inspect": 0.5,
    "answer_faq": 1.0,
}


def select_applicable_subgraph(graph, selection: dict, trajectory: dict, *, max_edges: int) -> dict:
    """只选择当前证据证明为 TRUE 的一跳边；未来路径必须在后续状态重新求值。"""
    if not 1 <= max_edges <= 16:
        raise ValueError("适用边上限必须位于 1 到 16 之间")
    candidate_scores = {item.get("node_id"): item.get("score", 0.0)
                        for item in selection.get("candidates") or []}
    unresolved = set(trajectory.get("unresolved_objective_kinds") or [])
    history = trajectory.get("edge_feedback") if isinstance(trajectory.get("edge_feedback"), dict) else {}
    direct = [dict(edge) for edge in selection.get("edges") or [] if edge.get("retrieval_depth", 1) == 1]
    evaluations = []
    applicable = []
    for edge in direct:
        graph_edge = next((item for item in graph.edges if item.id == edge.get("id")), None)
        evaluation = evaluate_edge_condition(graph_edge.applicability if graph_edge else None, trajectory)
        feedback = history.get(edge.get("id"), {}) if isinstance(history.get(edge.get("id")), dict) else {}
        score = (
            1.0
            + float(candidate_scores.get(edge.get("target"), 0.0))
            + (0.8 if edge.get("target") in unresolved else 0.0)
            + 0.15 * int(feedback.get("progressed", 0))
            - 0.2 * int(feedback.get("no_progress", 0))
            - 0.25 * int(feedback.get("failed", 0))
            - 0.1 * _ACTION_COST.get(str(edge.get("target")), 0.25)
        )
        row = {
            "edge_id": edge.get("id"),
            "source": edge.get("source"),
            "target": edge.get("target"),
            "status": evaluation["status"],
            "reason": evaluation["reason"],
            "score": round(score, 4) if evaluation["status"] == "TRUE" else None,
            "evaluation": evaluation,
        }
        evaluations.append(row)
        if evaluation["status"] == "TRUE":
            applicable.append((score, edge, evaluation))
    applicable.sort(key=lambda item: (-item[0], str(item[1].get("id"))))
    selected = []
    for score, edge, evaluation in applicable[:max_edges]:
        selected.append({**edge, "applicability": evaluation, "path_score": round(score, 4)})

    def continuations(node_id: str, remaining: int, visited: frozenset[str]):
        if remaining <= 0:
            return [([], [], 0.0)]
        outgoing = sorted((item for item in graph.edges
                           if item.source == node_id and item.id not in visited), key=lambda item: item.id)
        if not outgoing:
            return [([], [], 0.0)]
        rows = []
        for following in outgoing:
            future_eval = evaluate_edge_condition(following.applicability, trajectory)
            feedback = history.get(following.id, {}) if isinstance(history.get(following.id), dict) else {}
            future_score = (
                0.2 if future_eval["status"] == "TRUE" else -0.05 if future_eval["status"] == "FALSE" else 0.0
            )
            future_score += 0.25 if following.target in unresolved else 0.0
            future_score += 0.08 * int(feedback.get("progressed", 0))
            future_score -= 0.1 * int(feedback.get("no_progress", 0))
            future_score -= 0.12 * int(feedback.get("failed", 0))
            future_score -= 0.05 * _ACTION_COST.get(following.target, 0.25)
            for edge_ids, node_ids, tail_score in continuations(
                    following.target, remaining - 1, visited | {following.id}):
                rows.append(([following.id, *edge_ids], [following.target, *node_ids],
                             future_score + tail_score))
        return rows

    ranked_paths = []
    for score, edge, _ in applicable[:max_edges]:
        for future_edges, future_nodes, future_score in continuations(
                edge["target"], int(selection.get("hops") or 1) - 1, frozenset({edge["id"]})):
            edge_ids = [edge["id"], *future_edges]
            node_ids = [edge["source"], edge["target"], *future_nodes]
            ranked_paths.append((score + future_score, edge_ids, node_ids))
    ranked_paths.sort(key=lambda item: (-item[0], item[1]))
    paths = [{
        "path_id": _fingerprint({"anchor": selection.get("anchor"), "edge_ids": edge_ids})[:16],
        "score": round(score, 4),
        "edge_ids": edge_ids,
        "node_ids": node_ids,
        "currently_applicable_prefix": 1,
        "future_edges_require_reevaluation": edge_ids[1:],
    } for score, edge_ids, node_ids in ranked_paths[:max_edges]]

    rejected = [item["edge_id"] for item in evaluations if item["status"] != "TRUE"]
    result = dict(selection)
    result.update(
        algorithm="trajectory-state-conditioned-v4",
        edges=selected,
        candidate_edge_ids=[item.get("id") for item in direct],
        paths=paths,
        applicability={
            "strategy": "typed-evidence-v1",
            "status": "APPLICABLE" if selected else "ABSTAINED",
            "reason": "APPLICABLE_EDGE_FOUND" if selected else "NO_PROVEN_APPLICABLE_EDGE",
            "evaluations": evaluations,
            "selected_edge_ids": [item["id"] for item in selected],
            "false_edge_ids": [item["edge_id"] for item in evaluations if item["status"] == "FALSE"],
            "unknown_edge_ids": [item["edge_id"] for item in evaluations if item["status"] == "UNKNOWN"],
        },
        excluded_edge_ids=sorted(set(selection.get("excluded_edge_ids") or []) | set(rejected)),
    )
    return result
