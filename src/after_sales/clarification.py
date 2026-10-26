"""用可审计的反事实分支选择最有价值的单义务澄清问题。"""
from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from src.after_sales.execution import CaseExecutionState
from src.after_sales.obligation_graph import (
    ACTIVE_CLARIFICATION_SOLVER_VERSION,
    CandidateSelection,
    ObligationActionCandidate,
    ObligationGraph,
    build_candidate_selection,
    solve_obligation_graph,
)
from src.after_sales.service import _checksum


CLARIFICATION_PORTFOLIO_VERSION = "active-clarification-portfolio-v1"
CLARIFICATION_VERIFIER_VERSION = "active-clarification-verifier-v1"
CLARIFICATION_SELECTION_REASON = "ACTIVE_CLARIFICATION_MAX_STRUCTURAL_VALUE"
MAX_REPORTED_NEEDS = 8


class ClarificationNeed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    need_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source: Literal["SEMANTIC_GAP", "POLICY_REQUIREMENT"]
    text: str = Field(min_length=1, max_length=500)


class ClarificationAnswerBranch(BaseModel):
    """A possible successor class, never a probability or user prediction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    branch_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome: Literal[
        "SUFFICIENT_EVIDENCE",
        "PARTIAL_EVIDENCE",
        "EXPLICIT_DEFERRAL",
        "UNRESOLVED_OR_CONFLICT",
    ]
    next_mode: Literal["REASSESS", "WAIT", "ASK_USER"]
    resolves_unknown_upper_bound: int = Field(ge=0)
    policy_unlock_upper_bound: int = Field(ge=0)
    requires_semantic_verification: bool = True


class ClarificationQuestionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    obligation_candidate_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    requirement_id: str
    question: str = Field(min_length=1, max_length=500)
    semantic_status: Literal[
        "INCOMPLETE", "IRRELEVANT", "CONFLICT", "UNKNOWN", "NOT_ASSESSED"
    ]
    assessment_id: str | None = Field(default=None, max_length=64)
    semantic_basis_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    needs: tuple[ClarificationNeed, ...]
    answer_branches: tuple[ClarificationAnswerBranch, ...]
    reported_need_count: int = Field(ge=0)
    covered_need_count: int = Field(ge=0)
    information_gain_upper_bound: int = Field(ge=1)
    policy_unlock_upper_bound: int = Field(ge=1)
    interaction_turn_cost: Literal[1] = 1
    response_burden: int = Field(ge=1)
    specificity_rank: int = Field(ge=0)


class ClarificationPortfolio(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    portfolio_version: Literal["active-clarification-portfolio-v1"] = CLARIFICATION_PORTFOLIO_VERSION
    verifier_version: Literal["active-clarification-verifier-v1"] = CLARIFICATION_VERIFIER_VERSION
    portfolio_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_decision_id: str | None = None
    status: Literal["READY", "NOT_REQUIRED", "INVALID"]
    reason_code: str
    questions: tuple[ClarificationQuestionCandidate, ...] = ()
    validation_issues: tuple[str, ...] = ()


class ActiveClarificationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    solver_version: Literal["active-clarification-solver-v1"] = ACTIVE_CLARIFICATION_SOLVER_VERSION
    decision_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    portfolio_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["SELECTED", "BYPASSED", "NOT_REQUIRED", "INVALID"]
    reason_code: str
    selected_question_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    selected_obligation_candidate_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    frontier_question_ids: tuple[str, ...] = ()
    dominated_question_ids: tuple[str, ...] = ()
    candidate_selection: CandidateSelection | None = None
    ranking_policy: tuple[str, ...] = (
        "safe_non_user_action_first",
        "pareto_frontier",
        "policy_unlock_upper_bound_desc",
        "information_gain_upper_bound_desc",
        "semantic_specificity",
        "response_burden",
        "stable_requirement_identity",
    )


def _record(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _question_rank(question: ClarificationQuestionCandidate) -> tuple[Any, ...]:
    return (
        -question.policy_unlock_upper_bound,
        -question.information_gain_upper_bound,
        question.specificity_rank,
        question.response_burden,
        question.requirement_id,
        question.question_id,
    )


def _dominates(
    left: ClarificationQuestionCandidate,
    right: ClarificationQuestionCandidate,
) -> bool:
    weakly_better = (
        left.policy_unlock_upper_bound >= right.policy_unlock_upper_bound
        and left.information_gain_upper_bound >= right.information_gain_upper_bound
        and left.specificity_rank <= right.specificity_rank
        and left.response_burden <= right.response_burden
        and left.interaction_turn_cost <= right.interaction_turn_cost
    )
    strictly_better = (
        left.policy_unlock_upper_bound > right.policy_unlock_upper_bound
        or left.information_gain_upper_bound > right.information_gain_upper_bound
        or left.specificity_rank < right.specificity_rank
        or left.response_burden < right.response_burden
        or left.interaction_turn_cost < right.interaction_turn_cost
    )
    return weakly_better and strictly_better


def _needs_for_requirement(
    requirement_id: str,
    description: object,
    semantic: dict[str, Any],
    question: object,
) -> tuple[tuple[ClarificationNeed, ...], str, int, int, int, tuple[str, ...]]:
    issues: list[str] = []
    description_text = description.strip() if isinstance(description, str) else ""
    if len(description_text) > 500:
        issues.append(f"CLARIFICATION_DESCRIPTION_INVALID:{requirement_id}")
        description_text = "政策要求的材料"
    status = str(semantic.get("status") or "NOT_ASSESSED")
    if status not in {"INCOMPLETE", "IRRELEVANT", "CONFLICT", "UNKNOWN", "NOT_ASSESSED"}:
        issues.append(f"CLARIFICATION_SEMANTIC_STATUS_INVALID:{requirement_id}")
        status = "NOT_ASSESSED"
    raw_missing = semantic.get("missing_information") if semantic else None
    if raw_missing is not None and not isinstance(raw_missing, list):
        issues.append(f"CLARIFICATION_NEEDS_INVALID:{requirement_id}")
        raw_missing = []
    if isinstance(raw_missing, list) and len(raw_missing) > MAX_REPORTED_NEEDS:
        issues.append(f"CLARIFICATION_NEEDS_LIMIT_EXCEEDED:{requirement_id}")
        raw_missing = raw_missing[:MAX_REPORTED_NEEDS]
    missing: list[str] = []
    for item in raw_missing or []:
        if not isinstance(item, str) or not item.strip() or len(item.strip()) > 500:
            issues.append(f"CLARIFICATION_NEED_INVALID:{requirement_id}")
            continue
        if item.strip() not in missing:
            missing.append(item.strip())
    covered_missing = [
        item for item in missing
        if isinstance(question, str) and item in question
    ]
    use_semantic_gaps = bool(covered_missing and not semantic.get("error_code"))
    texts = covered_missing if use_semantic_gaps else [description_text]
    if not description_text or any(not text for text in texts):
        issues.append(f"CLARIFICATION_DESCRIPTION_INVALID:{requirement_id}")
        texts = ["政策要求的材料"]
    source = "SEMANTIC_GAP" if use_semantic_gaps else "POLICY_REQUIREMENT"
    needs = tuple(
        ClarificationNeed(
            need_id=_checksum({
                "requirement_id": requirement_id,
                "source": source,
                "text": text,
            }),
            source=source,
            text=text,
        )
        for text in texts
    )
    specificity = {
        "INCOMPLETE": 0,
        "CONFLICT": 1,
        "IRRELEVANT": 2,
        "UNKNOWN": 3,
        "NOT_ASSESSED": 4,
    }[status]
    if not use_semantic_gaps:
        specificity = max(specificity, 4)
    return needs, status, specificity, len(missing), len(covered_missing), tuple(issues)


def _answer_branches(
    question_id: str,
    *,
    need_count: int,
    policy_unlock_upper_bound: int,
) -> tuple[ClarificationAnswerBranch, ...]:
    rows = (
        ("SUFFICIENT_EVIDENCE", "REASSESS", need_count, policy_unlock_upper_bound),
        ("PARTIAL_EVIDENCE", "REASSESS", max(need_count - 1, 0), 0),
        ("EXPLICIT_DEFERRAL", "WAIT", 0, 0),
        ("UNRESOLVED_OR_CONFLICT", "ASK_USER", 0, 0),
    )
    return tuple(
        ClarificationAnswerBranch(
            branch_id=_checksum({
                "question_id": question_id,
                "outcome": outcome,
                "next_mode": next_mode,
                "resolves_unknown_upper_bound": resolved,
                "policy_unlock_upper_bound": unlocked,
            }),
            outcome=outcome,
            next_mode=next_mode,
            resolves_unknown_upper_bound=resolved,
            policy_unlock_upper_bound=unlocked,
        )
        for outcome, next_mode, resolved, unlocked in rows
    )


def _question_candidate(
    candidate: ObligationActionCandidate,
    requirement: dict[str, Any],
) -> tuple[ClarificationQuestionCandidate | None, tuple[str, ...]]:
    issues: list[str] = []
    requirement_id = candidate.requirement_id or ""
    raw_semantic = requirement.get("semantic_assessment")
    if raw_semantic is not None and not isinstance(raw_semantic, dict):
        issues.append(f"CLARIFICATION_SEMANTIC_ASSESSMENT_INVALID:{requirement_id}")
    semantic = _record(raw_semantic)
    needs, semantic_status, specificity, reported_need_count, covered_need_count, need_issues = _needs_for_requirement(
        requirement_id,
        requirement.get("description"),
        semantic,
        candidate.question,
    )
    issues.extend(need_issues)
    semantic_question = semantic.get("question") if semantic else None
    expected_question = semantic_question or requirement.get("description")
    if (
        not isinstance(expected_question, str)
        or not expected_question.strip()
        or len(expected_question) > 500
        or candidate.question != expected_question
    ):
        issues.append(f"CLARIFICATION_QUESTION_BINDING_INVALID:{requirement_id}")
    assessment_id = semantic.get("assessment_id")
    if assessment_id is not None and (
        not isinstance(assessment_id, str)
        or not assessment_id
        or len(assessment_id) > 64
    ):
        issues.append(f"CLARIFICATION_ASSESSMENT_ID_INVALID:{requirement_id}")
    if not candidate.question or len(candidate.question) > 500 or not candidate.unlocks:
        issues.append(f"CLARIFICATION_CANDIDATE_INVALID:{requirement_id}")
    if issues:
        return None, tuple(issues)
    semantic_basis = _checksum(semantic) if semantic else None
    question_basis = {
        "obligation_candidate_id": candidate.candidate_id,
        "requirement_id": requirement_id,
        "question": candidate.question,
        "semantic_status": semantic_status,
        "assessment_id": semantic.get("assessment_id"),
        "semantic_basis_sha256": semantic_basis,
        "need_ids": [need.need_id for need in needs],
        "policy_unlock_upper_bound": len(candidate.unlocks),
    }
    question_id = _checksum(question_basis)
    return ClarificationQuestionCandidate(
        question_id=question_id,
        obligation_candidate_id=candidate.candidate_id,
        requirement_id=requirement_id,
        question=candidate.question,
        semantic_status=semantic_status,
        assessment_id=assessment_id,
        semantic_basis_sha256=semantic_basis,
        needs=needs,
        answer_branches=_answer_branches(
            question_id,
            need_count=len(needs),
            policy_unlock_upper_bound=len(candidate.unlocks),
        ),
        reported_need_count=reported_need_count,
        covered_need_count=covered_need_count,
        information_gain_upper_bound=len(needs),
        policy_unlock_upper_bound=len(candidate.unlocks),
        response_burden=len(needs),
        specificity_rank=specificity,
    ), ()


def build_clarification_portfolio(
    state: CaseExecutionState,
    graph: ObligationGraph,
    result: Mapping[str, Any],
) -> ClarificationPortfolio:
    """Compile question candidates from immutable policy facts without inventing questions."""

    issues: list[str] = []
    if graph.state_id != state.state_id:
        issues.append("CLARIFICATION_GRAPH_STATE_MISMATCH")
    if graph.status != "READY":
        issues.append("CLARIFICATION_OBLIGATION_GRAPH_INVALID")
    case = _record(result.get("case"))
    decision = _record(result.get("policy_decision") or case.get("latest_decision"))
    if (
        decision.get("decision_id") != state.policy_decision_id
        or decision.get("policy_version_id") != state.policy_version_id
        or decision.get("policy_checksum") != state.policy_checksum
        or decision.get("outcome") != state.policy_outcome
    ):
        issues.append("CLARIFICATION_POLICY_IDENTITY_MISMATCH")
    raw_requirements = _record(decision.get("facts")).get("evidence_requirements")
    if raw_requirements is None:
        raw_requirements = []
    elif not isinstance(raw_requirements, list):
        issues.append("CLARIFICATION_REQUIREMENT_SET_INVALID")
        raw_requirements = []
    requirements_list = []
    for row in raw_requirements:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("requirement_id"), str)
            or not row["requirement_id"]
        ):
            issues.append("CLARIFICATION_REQUIREMENT_INVALID")
            continue
        requirements_list.append(row)
    requirement_ids = [str(row["requirement_id"]) for row in requirements_list]
    if len(requirement_ids) != len(set(requirement_ids)):
        issues.append("CLARIFICATION_DUPLICATE_REQUIREMENT")
    requirements = {str(row["requirement_id"]): row for row in requirements_list}
    state_requirements = {row.requirement_id: row for row in state.obligations}
    questions: list[ClarificationQuestionCandidate] = []
    for candidate in sorted(
        (row for row in graph.candidates if row.admissible and row.mode == "ASK_USER"),
        key=lambda row: (row.requirement_id or "", row.candidate_id),
    ):
        requirement_id = candidate.requirement_id or ""
        requirement = requirements.get(requirement_id)
        projected = state_requirements.get(requirement_id)
        accepted_sources = requirement.get("accepted_sources") if requirement else None
        if (
            requirement is None
            or projected is None
            or requirement.get("evidence_type") != projected.evidence_type
            or requirement.get("description") != projected.description
            or requirement.get("status") != projected.status
            or requirement.get("validation_condition") != projected.validation_condition
            or not isinstance(accepted_sources, list)
            or not all(isinstance(item, str) for item in accepted_sources)
            or tuple(accepted_sources) != projected.accepted_sources
        ):
            issues.append(f"CLARIFICATION_REQUIREMENT_BINDING_INVALID:{requirement_id}")
            continue
        question, question_issues = _question_candidate(candidate, requirement)
        issues.extend(question_issues)
        if question is not None:
            questions.append(question)
    questions.sort(key=lambda row: (row.requirement_id, row.question_id))
    question_ids = [row.question_id for row in questions]
    if len(question_ids) != len(set(question_ids)):
        issues.append("CLARIFICATION_DUPLICATE_QUESTION")
    expected_questions = sum(
        1 for row in graph.candidates if row.admissible and row.mode == "ASK_USER"
    )
    if len(questions) != expected_questions:
        issues.append("CLARIFICATION_QUESTION_SET_INCOMPLETE")
    unique_issues = tuple(dict.fromkeys(issues))
    status = "INVALID" if unique_issues else "READY" if questions else "NOT_REQUIRED"
    reason_code = (
        unique_issues[0]
        if unique_issues else
        "CLARIFICATION_PORTFOLIO_VERIFIED" if questions else
        "CLARIFICATION_NOT_REQUIRED"
    )
    basis = {
        "portfolio_version": CLARIFICATION_PORTFOLIO_VERSION,
        "verifier_version": CLARIFICATION_VERIFIER_VERSION,
        "state_id": state.state_id,
        "graph_id": graph.graph_id,
        "policy_decision_id": state.policy_decision_id,
        "status": status,
        "reason_code": reason_code,
        "questions": [row.model_dump(mode="json") for row in questions],
        "validation_issues": unique_issues,
    }
    return ClarificationPortfolio(portfolio_id=_checksum(basis), **basis)


def _decision(
    graph: ObligationGraph,
    portfolio: ClarificationPortfolio,
    *,
    status: str,
    reason_code: str,
    frontier_question_ids: tuple[str, ...] = (),
    dominated_question_ids: tuple[str, ...] = (),
    selected_question: ClarificationQuestionCandidate | None = None,
    candidate_selection: CandidateSelection | None = None,
) -> ActiveClarificationDecision:
    payload = {
        "graph_id": graph.graph_id,
        "portfolio_id": portfolio.portfolio_id,
        "status": status,
        "reason_code": reason_code,
        "selected_question_id": selected_question.question_id if selected_question else None,
        "selected_obligation_candidate_id": (
            selected_question.obligation_candidate_id if selected_question else None
        ),
        "frontier_question_ids": frontier_question_ids,
        "dominated_question_ids": dominated_question_ids,
        "candidate_selection": (
            candidate_selection.model_dump(mode="json") if candidate_selection else None
        ),
    }
    return ActiveClarificationDecision(decision_id=_checksum({
        "solver_version": ACTIVE_CLARIFICATION_SOLVER_VERSION,
        **payload,
    }), **payload)


def solve_active_clarification(
    graph: ObligationGraph,
    portfolio: ClarificationPortfolio,
) -> ActiveClarificationDecision:
    """Prefer safe non-user actions, otherwise select one Pareto-frontier question."""

    if graph.status != "READY":
        return _decision(
            graph,
            portfolio,
            status="INVALID",
            reason_code="CLARIFICATION_OBLIGATION_GRAPH_INVALID",
        )
    fatal_portfolio_issues = {
        "CLARIFICATION_GRAPH_STATE_MISMATCH",
        "CLARIFICATION_OBLIGATION_GRAPH_INVALID",
        "CLARIFICATION_POLICY_IDENTITY_MISMATCH",
    }
    if any(issue in fatal_portfolio_issues for issue in portfolio.validation_issues):
        return _decision(
            graph,
            portfolio,
            status="INVALID",
            reason_code=portfolio.reason_code,
        )
    base = solve_obligation_graph(graph)
    if base is None:
        return _decision(
            graph,
            portfolio,
            status="NOT_REQUIRED",
            reason_code="CLARIFICATION_NOT_REQUIRED",
        )
    candidates = {row.candidate_id: row for row in graph.candidates}
    base_candidate = candidates[base.selected_candidate_id]
    if base_candidate.mode != "ASK_USER":
        return _decision(
            graph,
            portfolio,
            status="BYPASSED",
            reason_code=base.reason_code,
            candidate_selection=base,
        )
    if portfolio.status != "READY":
        return _decision(
            graph,
            portfolio,
            status="INVALID",
            reason_code=portfolio.reason_code,
        )

    frontier = tuple(
        question for question in portfolio.questions
        if not any(
            _dominates(other, question)
            for other in portfolio.questions
            if other.question_id != question.question_id
        )
    )
    ordered_frontier = sorted(frontier, key=_question_rank)
    selected = ordered_frontier[0]
    ordered_questions = sorted(portfolio.questions, key=_question_rank)
    ordered_candidate_ids = [row.obligation_candidate_id for row in ordered_questions]
    ordered_candidates = [candidates[candidate_id] for candidate_id in ordered_candidate_ids]
    ordered_candidates.extend(
        candidates[candidate_id]
        for candidate_id in base.eligible_candidate_ids
        if candidate_id not in set(ordered_candidate_ids)
    )
    selection = build_candidate_selection(
        graph,
        ordered_candidates,
        reason_code=CLARIFICATION_SELECTION_REASON,
        solver_version=ACTIVE_CLARIFICATION_SOLVER_VERSION,
        ranking_policy=ActiveClarificationDecision.model_fields["ranking_policy"].default,
    )
    frontier_ids = tuple(row.question_id for row in ordered_frontier)
    dominated_ids = tuple(
        row.question_id for row in ordered_questions if row.question_id not in set(frontier_ids)
    )
    return _decision(
        graph,
        portfolio,
        status="SELECTED",
        reason_code=CLARIFICATION_SELECTION_REASON,
        frontier_question_ids=frontier_ids,
        dominated_question_ids=dominated_ids,
        selected_question=selected,
        candidate_selection=selection,
    )
