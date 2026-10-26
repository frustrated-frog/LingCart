"""把持久化政策义务编译为可审计的候选行动图并确定性求解。"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.after_sales.execution import CaseExecutionState
from src.after_sales.service import _checksum


OBLIGATION_GRAPH_VERSION = "after-sales-obligation-graph-v1"
OBLIGATION_ACTION_SOLVER_VERSION = "obligation-action-solver-v1"
OBLIGATION_ACTION_PLANNER_VERSION = "policy-aware-obligation-planner-v1"
OBLIGATION_GRAPH_VERIFIER_VERSION = "policy-obligation-graph-verifier-v1"
AUTONOMOUS_OPERATION = "continue_after_sales_case"


class ObligationNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    kind: Literal["REQUIREMENT", "RESOLUTION"]
    status: str
    requirement_id: str | None = None
    evidence_type: str | None = None


class ObligationEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_node_id: str
    target_node_id: str
    relation: Literal["BLOCKS"] = "BLOCKS"


class ObligationActionCandidate(BaseModel):
    """A bounded action candidate with ordinal, not probabilistic, cost evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    action_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    action_kind: str
    mode: Literal["AUTONOMOUS", "ASK_USER", "WAIT", "APPROVAL", "BLOCKED", "INVALID"]
    goal_id: str | None = None
    requirement_id: str | None = None
    operation: str | None = None
    requires_approval: bool = False
    question: str | None = None
    admissible: bool
    reason_code: str
    information_gain_upper_bound: int = Field(ge=0)
    unlocks: tuple[str, ...] = ()
    user_interaction_cost: int = Field(ge=0)
    tool_call_cost: int = Field(ge=0)
    delay_cost: int = Field(ge=0)
    risk_rank: int = Field(ge=0)


class ObligationGraph(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    graph_version: Literal["after-sales-obligation-graph-v1"] = OBLIGATION_GRAPH_VERSION
    verifier_version: Literal["policy-obligation-graph-verifier-v1"] = OBLIGATION_GRAPH_VERIFIER_VERSION
    graph_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["READY", "INVALID"]
    reason_code: str
    nodes: tuple[ObligationNode, ...] = ()
    edges: tuple[ObligationEdge, ...] = ()
    candidates: tuple[ObligationActionCandidate, ...] = ()
    validation_issues: tuple[str, ...] = ()


class CandidateSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    solver_version: Literal["obligation-action-solver-v1"] = OBLIGATION_ACTION_SOLVER_VERSION
    selection_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_candidate_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    eligible_candidate_ids: tuple[str, ...]
    dominated_candidate_ids: tuple[str, ...]
    reason_code: str
    ranking_policy: tuple[str, ...] = (
        "safe_action_class",
        "information_gain_upper_bound_desc",
        "user_interaction_cost",
        "tool_call_cost",
        "delay_cost",
        "risk_rank",
        "stable_target_identity",
    )


class ObligationActionDirective(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    planner_version: Literal["policy-aware-obligation-planner-v1"] = OBLIGATION_ACTION_PLANNER_VERSION
    verifier_version: Literal["policy-obligation-graph-verifier-v1"] = OBLIGATION_GRAPH_VERIFIER_VERSION
    directive_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    candidate_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    status: Literal["EXECUTE", "WAIT_USER_INPUT", "WAIT_APPROVAL", "BLOCKED", "COMPLETE", "ABSTAINED"]
    reason_code: str
    action_kind: str | None = None
    goal_id: str | None = None
    requirement_id: str | None = None
    operation: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    question: str | None = None
    selection: CandidateSelection | None = None


def _node_id(requirement_id: str) -> str:
    return f"requirement:{requirement_id}"


def _expected_goal_id(evidence_type: str) -> str:
    return f"evidence:{evidence_type}"


def _action_sort_key(action: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(action.get(key) or "") for key in (
        "kind", "goal_id", "requirement_id", "operation", "claim_id", "question",
    ))


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _requires_no_approval(action: dict[str, Any]) -> bool:
    """Treat omitted/false as safe and reject truthy lookalikes such as 1 or "false"."""

    return "requires_approval" not in action or action["requires_approval"] is False


def _has_no_operation(action: dict[str, Any]) -> bool:
    return "operation" not in action or action["operation"] is None


def _potential_unlocks(state: CaseExecutionState, requirement_id: str) -> tuple[str, ...]:
    unresolved = {
        obligation.requirement_id
        for obligation in state.obligations
        if obligation.status != "SATISFIED"
    }
    unlocks = [_node_id(requirement_id)]
    if unresolved <= {requirement_id}:
        unlocks.append("resolution")
    return tuple(unlocks)


def _candidate(
    state: CaseExecutionState,
    action: dict[str, Any],
    *,
    mode: str,
    admissible: bool,
    reason_code: str,
    information_gain_upper_bound: int = 0,
    unlocks: tuple[str, ...] = (),
    user_interaction_cost: int = 0,
    tool_call_cost: int = 0,
    delay_cost: int = 0,
    risk_rank: int = 0,
    resolved_requirement_id: str | None = None,
) -> ObligationActionCandidate:
    payload = {
        "action_fingerprint": _checksum(action),
        "action_kind": str(action.get("kind") or ""),
        "mode": mode,
        "goal_id": _optional_text(action.get("goal_id")),
        "requirement_id": resolved_requirement_id or _optional_text(action.get("requirement_id")),
        "operation": _optional_text(action.get("operation")),
        "requires_approval": action.get("requires_approval") is True,
        "question": _optional_text(action.get("question")),
        "admissible": admissible,
        "reason_code": reason_code,
        "information_gain_upper_bound": information_gain_upper_bound,
        "unlocks": unlocks,
        "user_interaction_cost": user_interaction_cost,
        "tool_call_cost": tool_call_cost,
        "delay_cost": delay_cost,
        "risk_rank": risk_rank,
    }
    return ObligationActionCandidate(
        candidate_id=_checksum({"state_id": state.state_id, **payload}),
        **payload,
    )


def _candidate_from_action(
    state: CaseExecutionState,
    action: dict[str, Any],
    requirements: dict[str, Any],
) -> ObligationActionCandidate:
    kind = str(action.get("kind") or "")
    requirement_id = str(action.get("requirement_id") or "")
    requirement = requirements.get(requirement_id)
    if kind == "REQUIRES_HUMAN_CAPABILITY" and not requirement_id:
        # Older persisted plans bound unsupported evidence actions only by their stable goal id.
        # Accept that legacy shape only when it resolves to exactly one current obligation.
        matches = [
            row for row in requirements.values()
            if _expected_goal_id(row.evidence_type) == action.get("goal_id")
        ]
        if len(matches) == 1:
            requirement = matches[0]
            requirement_id = requirement.requirement_id
    expected_goal_id = _expected_goal_id(requirement.evidence_type) if requirement else None
    targets_requirement = bool(
        requirement
        and requirement.status != "SATISFIED"
        and action.get("goal_id") == expected_goal_id
    )

    if kind == "COLLECT_BUSINESS_EVIDENCE":
        admissible = bool(
            targets_requirement
            and requirement.validation_condition == "AUTHORITATIVE_ORDER"
            and "ORDER_RECORD" in requirement.accepted_sources
            and action.get("operation") == AUTONOMOUS_OPERATION
            and _requires_no_approval(action)
        )
        unlocks = _potential_unlocks(state, requirement_id) if admissible else ()
        return _candidate(
            state,
            action,
            mode="AUTONOMOUS" if admissible else "INVALID",
            admissible=admissible,
            reason_code=(
                "AUTHORITATIVE_EVIDENCE_AVAILABLE"
                if admissible else "POLICY_OBLIGATION_NOT_AUTOMATABLE"
            ),
            information_gain_upper_bound=len(unlocks),
            unlocks=unlocks,
            tool_call_cost=1,
        )

    if kind == "REQUEST_CUSTOMER_EVIDENCE":
        admissible = bool(
            targets_requirement
            and requirement.validation_condition == "CUSTOMER_ATTESTATION"
            and "CUSTOMER_STATEMENT" in requirement.accepted_sources
            and requirement.readiness is None
            and _has_no_operation(action)
            and _requires_no_approval(action)
            and isinstance(action.get("question"), str)
            and action["question"].strip()
        )
        unlocks = _potential_unlocks(state, requirement_id) if admissible else ()
        return _candidate(
            state,
            action,
            mode="ASK_USER" if admissible else "INVALID",
            admissible=admissible,
            reason_code=(
                "CUSTOMER_EVIDENCE_REQUIRED"
                if admissible else "CUSTOMER_EVIDENCE_BINDING_INVALID"
            ),
            information_gain_upper_bound=len(unlocks),
            unlocks=unlocks,
            user_interaction_cost=1,
        )

    if kind == "WAIT_FOR_CUSTOMER_EVIDENCE":
        readiness = requirement.readiness if requirement else None
        admissible = bool(
            targets_requirement
            and readiness is not None
            and readiness.disposition == "DEFERRED"
            and readiness.requirement_id == requirement.requirement_id
            and readiness.evidence_type == requirement.evidence_type
            and readiness.policy_version_id == state.policy_version_id
            and readiness.policy_checksum == state.policy_checksum
            and action.get("claim_id") == readiness.claim_id
            and _has_no_operation(action)
            and _requires_no_approval(action)
            and isinstance(action.get("question"), str)
            and action["question"].strip()
        )
        return _candidate(
            state,
            action,
            mode="WAIT" if admissible else "INVALID",
            admissible=admissible,
            reason_code=(
                "CUSTOMER_EVIDENCE_DEFERRED"
                if admissible else "EVIDENCE_READINESS_BINDING_INVALID"
            ),
            delay_cost=1,
        )

    if kind == "REQUEST_CUSTOMER_CONFIRMATION":
        all_satisfied = all(row.status == "SATISFIED" for row in state.obligations)
        admissible = bool(
            all_satisfied
            and state.policy_outcome == "ELIGIBLE"
            and action.get("goal_id") == "resolution"
            and not action.get("requirement_id")
            and action.get("operation") == "submit_after_sales_case"
            and action.get("requires_approval") is True
        )
        return _candidate(
            state,
            action,
            mode="APPROVAL" if admissible else "INVALID",
            admissible=admissible,
            reason_code=("EXPLICIT_APPROVAL_REQUIRED" if admissible else "APPROVAL_ACTION_INVALID"),
            information_gain_upper_bound=1 if admissible else 0,
            unlocks=("resolution",) if admissible else (),
            user_interaction_cost=1,
            risk_rank=2,
        )

    if kind == "REQUIRES_HUMAN_CAPABILITY":
        requirement_valid = bool(
            targets_requirement
            and requirement.validation_condition not in {"AUTHORITATIVE_ORDER", "CUSTOMER_ATTESTATION"}
        )
        resolution_valid = bool(
            not requirement_id
            and action.get("goal_id") == "resolution"
            and state.policy_outcome == "MANUAL_REVIEW"
        )
        admissible = bool(
            (requirement_valid or resolution_valid)
            and _has_no_operation(action)
            and _requires_no_approval(action)
        )
        return _candidate(
            state,
            action,
            mode="BLOCKED" if admissible else "INVALID",
            admissible=admissible,
            reason_code=("UNSUPPORTED_CAPABILITY" if admissible else "HUMAN_CAPABILITY_BINDING_INVALID"),
            delay_cost=2,
            resolved_requirement_id=requirement_id or None,
        )

    return _candidate(
        state,
        action,
        mode="INVALID",
        admissible=False,
        reason_code="UNSUPPORTED_PLAN_ACTION_KIND",
    )


def build_obligation_graph(state: CaseExecutionState) -> ObligationGraph:
    """Compile current obligations and every persisted action without inventing alternatives."""

    issues: list[str] = []
    requirement_ids = [row.requirement_id for row in state.obligations]
    if len(requirement_ids) != len(set(requirement_ids)):
        issues.append("OBLIGATION_GRAPH_DUPLICATE_REQUIREMENT")
    requirements = {row.requirement_id: row for row in state.obligations}
    ordered_requirements = [requirements[key] for key in sorted(requirements)]
    nodes = tuple([
        *(
            ObligationNode(
                node_id=_node_id(row.requirement_id),
                kind="REQUIREMENT",
                status=row.status,
                requirement_id=row.requirement_id,
                evidence_type=row.evidence_type,
            )
            for row in ordered_requirements
        ),
        ObligationNode(
            node_id="resolution",
            kind="RESOLUTION",
            status="SATISFIED" if state.status == "TERMINAL" else "PENDING",
        ),
    ])
    edges = tuple(
        ObligationEdge(source_node_id=_node_id(row.requirement_id), target_node_id="resolution")
        for row in ordered_requirements
    )
    actions = sorted((dict(row) for row in state.next_actions), key=_action_sort_key)
    candidates = tuple(_candidate_from_action(state, action, requirements) for action in actions)
    candidate_ids = [row.candidate_id for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        issues.append("OBLIGATION_GRAPH_DUPLICATE_ACTION")
    issues.extend(row.reason_code for row in candidates if not row.admissible)

    represented_requirements = {
        row.requirement_id
        for row in candidates
        if row.admissible and row.requirement_id
    }
    unresolved_requirements = {
        row.requirement_id for row in state.obligations if row.status != "SATISFIED"
    }
    expected_case_status = {
        "NEEDS_EVIDENCE": "AWAITING_EVIDENCE",
        "ELIGIBLE": "AWAITING_CONFIRMATION",
        "INELIGIBLE": "REJECTED",
        "MANUAL_REVIEW": "MANUAL_REVIEW",
    }
    if state.policy_outcome not in expected_case_status:
        issues.append("POLICY_OUTCOME_UNSUPPORTED")
    elif state.status == "READY" and state.case_status != expected_case_status[state.policy_outcome]:
        issues.append("POLICY_OUTCOME_CASE_STATUS_MISMATCH")
    if (
        state.policy_outcome == "NEEDS_EVIDENCE" and not unresolved_requirements
    ) or (
        state.policy_outcome == "ELIGIBLE" and unresolved_requirements
    ):
        issues.append("POLICY_OUTCOME_OBLIGATION_MISMATCH")
    if state.policy_outcome == "NEEDS_EVIDENCE":
        missing = sorted(
            requirement_id
            for requirement_id in unresolved_requirements
            if requirement_id not in represented_requirements
        )
        if missing:
            issues.append("OBLIGATION_ACTION_MISSING:" + ",".join(missing))
    if state.policy_outcome == "ELIGIBLE" and not any(
        row.admissible and row.mode == "APPROVAL" for row in candidates
    ):
        issues.append("OBLIGATION_APPROVAL_ACTION_MISSING")
    if state.policy_outcome == "MANUAL_REVIEW" and not any(
        row.admissible and row.mode == "BLOCKED" for row in candidates
    ):
        issues.append("OBLIGATION_CAPABILITY_BOUNDARY_MISSING")

    unique_issues = tuple(dict.fromkeys(issues))
    status = "INVALID" if unique_issues else "READY"
    reason_code = unique_issues[0] if unique_issues else "OBLIGATION_GRAPH_VERIFIED"
    basis = {
        "graph_version": OBLIGATION_GRAPH_VERSION,
        "verifier_version": OBLIGATION_GRAPH_VERIFIER_VERSION,
        "state_id": state.state_id,
        "status": status,
        "reason_code": reason_code,
        "nodes": [row.model_dump(mode="json") for row in nodes],
        "edges": [row.model_dump(mode="json") for row in edges],
        "candidates": [row.model_dump(mode="json") for row in candidates],
        "validation_issues": unique_issues,
    }
    return ObligationGraph(graph_id=_checksum(basis), **basis)


_MODE_RANK = {
    "AUTONOMOUS": 0,
    "APPROVAL": 1,
    "ASK_USER": 2,
    "WAIT": 3,
    "BLOCKED": 4,
}


def _candidate_rank(candidate: ObligationActionCandidate) -> tuple[Any, ...]:
    return (
        _MODE_RANK[candidate.mode],
        -candidate.information_gain_upper_bound,
        candidate.user_interaction_cost,
        candidate.tool_call_cost,
        candidate.delay_cost,
        candidate.risk_rank,
        candidate.requirement_id or candidate.goal_id or "",
        candidate.candidate_id,
    )


def solve_obligation_graph(graph: ObligationGraph) -> CandidateSelection | None:
    if graph.status != "READY":
        return None
    eligible = sorted((row for row in graph.candidates if row.admissible), key=_candidate_rank)
    if not eligible:
        return None
    selected = eligible[0]
    reasons = {
        "AUTONOMOUS": "TRUSTED_AUTONOMOUS_ACTION_DOMINATES",
        "APPROVAL": "POLICY_BOUND_APPROVAL_READY",
        "ASK_USER": "HIGHEST_INFORMATION_GAIN_QUESTION",
        "WAIT": "DEFERRED_OBLIGATION_PRESERVED",
        "BLOCKED": "CAPABILITY_BOUNDARY_PRESERVED",
    }
    payload = {
        "graph_id": graph.graph_id,
        "selected_candidate_id": selected.candidate_id,
        "eligible_candidate_ids": tuple(row.candidate_id for row in eligible),
        "dominated_candidate_ids": tuple(row.candidate_id for row in eligible[1:]),
        "reason_code": reasons[selected.mode],
    }
    return CandidateSelection(selection_id=_checksum(payload), **payload)


def _directive(
    state: CaseExecutionState,
    *,
    status: str,
    reason_code: str,
    graph: ObligationGraph | None = None,
    candidate: ObligationActionCandidate | None = None,
    selection: CandidateSelection | None = None,
    operation: str | None = None,
    arguments: dict[str, Any] | None = None,
    requires_approval: bool = False,
    question: str | None = None,
) -> ObligationActionDirective:
    payload = {
        "state_id": state.state_id,
        "graph_id": graph.graph_id if graph else None,
        "candidate_id": candidate.candidate_id if candidate else None,
        "status": status,
        "reason_code": reason_code,
        "action_kind": candidate.action_kind if candidate else None,
        "goal_id": candidate.goal_id if candidate else None,
        "requirement_id": candidate.requirement_id if candidate else None,
        "operation": operation,
        "arguments": arguments or {},
        "requires_approval": requires_approval,
        "question": question,
        "selection": selection,
    }
    identity = {
        **payload,
        "selection": selection.model_dump(mode="json") if selection else None,
    }
    return ObligationActionDirective(directive_id=_checksum(identity), **payload)


def decide_next_obligation_action(
    state: CaseExecutionState,
    graph: ObligationGraph | None = None,
) -> ObligationActionDirective:
    """Solve one bounded action while keeping policy and authorization authoritative."""

    if state.status == "TERMINAL":
        return _directive(state, status="COMPLETE", reason_code="CASE_TERMINAL")
    if state.status != "READY":
        return _directive(state, status="ABSTAINED", reason_code="CASE_STATE_NOT_EXECUTABLE")
    graph = graph or build_obligation_graph(state)
    if graph.state_id != state.state_id:
        return _directive(
            state,
            graph=graph,
            status="ABSTAINED",
            reason_code="OBLIGATION_GRAPH_STATE_MISMATCH",
        )
    if graph.status != "READY":
        return _directive(state, graph=graph, status="ABSTAINED", reason_code=graph.reason_code)
    selection = solve_obligation_graph(graph)
    if selection is None:
        if not graph.candidates and state.policy_outcome == "INELIGIBLE":
            return _directive(
                state,
                graph=graph,
                status="COMPLETE",
                reason_code="NO_FURTHER_AUTONOMOUS_ACTION",
            )
        return _directive(state, graph=graph, status="ABSTAINED", reason_code="NO_SUPPORTED_NEXT_ACTION")
    candidates = {row.candidate_id: row for row in graph.candidates}
    candidate = candidates[selection.selected_candidate_id]
    if candidate.mode == "AUTONOMOUS":
        return _directive(
            state,
            graph=graph,
            candidate=candidate,
            selection=selection,
            status="EXECUTE",
            reason_code=candidate.reason_code,
            operation=AUTONOMOUS_OPERATION,
            arguments={
                "case_id": state.case_id,
                "expected_revision": state.case_revision,
                "requirement_id": candidate.requirement_id,
            },
        )
    if candidate.mode == "ASK_USER":
        return _directive(
            state,
            graph=graph,
            candidate=candidate,
            selection=selection,
            status="WAIT_USER_INPUT",
            reason_code=candidate.reason_code,
            question=candidate.question,
        )
    if candidate.mode == "WAIT":
        return _directive(
            state,
            graph=graph,
            candidate=candidate,
            selection=selection,
            status="WAIT_USER_INPUT",
            reason_code=candidate.reason_code,
            question=candidate.question,
        )
    if candidate.mode == "APPROVAL":
        return _directive(
            state,
            graph=graph,
            candidate=candidate,
            selection=selection,
            status="WAIT_APPROVAL",
            reason_code=candidate.reason_code,
            requires_approval=True,
        )
    return _directive(
        state,
        graph=graph,
        candidate=candidate,
        selection=selection,
        status="BLOCKED",
        reason_code=candidate.reason_code,
    )
