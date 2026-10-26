"""持久化主动澄清问题，并把下一轮原文安全绑定回政策义务。"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from src.after_sales.service import _checksum
from src.backend.db import (
    AfterSalesCase,
    AfterSalesCasePlan,
    AfterSalesPendingClarification,
    PolicyDecision,
    PolicyVersion,
)
from src.errors import AgentError


CLARIFICATION_BINDING_VERSION = "after-sales-clarification-binding-v1"


class PendingClarificationContract(BaseModel):
    """Trusted runtime output. It routes an answer but carries no authorization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    binding_version: Literal["after-sales-clarification-binding-v1"] = CLARIFICATION_BINDING_VERSION
    clarification_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_id: str = Field(min_length=1, max_length=64)
    requirement_id: str = Field(min_length=1, max_length=160)
    question_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    question: str = Field(min_length=1, max_length=500)
    case_revision: int = Field(gt=0)
    policy_version_id: str = Field(min_length=1, max_length=96)
    policy_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_decision_id: str = Field(min_length=1, max_length=64)
    state_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    portfolio_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    clarification_decision_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


def _question_basis(*, question_id: str, requirement_id: str, question: str) -> dict:
    return {
        "question_id": question_id,
        "requirement_id": requirement_id,
        "question": question,
    }


def _binding_basis(payload: dict) -> dict:
    return {
        key: payload[key]
        for key in (
            "binding_version",
            "case_id",
            "requirement_id",
            "question_id",
            "question",
            "case_revision",
            "policy_version_id",
            "policy_checksum",
            "policy_decision_id",
            "state_id",
            "graph_id",
            "portfolio_id",
            "clarification_decision_id",
            "selection_id",
            "question_checksum",
        )
    }


def build_pending_clarification(
    *,
    case_id: str,
    requirement_id: str,
    question_id: str,
    question: str,
    case_revision: int,
    policy_version_id: str,
    policy_checksum: str,
    policy_decision_id: str,
    state_id: str,
    graph_id: str,
    portfolio_id: str,
    clarification_decision_id: str,
    selection_id: str,
) -> PendingClarificationContract:
    question_checksum = _checksum(_question_basis(
        question_id=question_id,
        requirement_id=requirement_id,
        question=question,
    ))
    payload = {
        "schema_version": 1,
        "binding_version": CLARIFICATION_BINDING_VERSION,
        "case_id": case_id,
        "requirement_id": requirement_id,
        "question_id": question_id,
        "question": question,
        "case_revision": case_revision,
        "policy_version_id": policy_version_id,
        "policy_checksum": policy_checksum,
        "policy_decision_id": policy_decision_id,
        "state_id": state_id,
        "graph_id": graph_id,
        "portfolio_id": portfolio_id,
        "clarification_decision_id": clarification_decision_id,
        "selection_id": selection_id,
        "question_checksum": question_checksum,
    }
    payload["binding_checksum"] = _checksum(_binding_basis(payload))
    payload["clarification_id"] = _checksum({
        "kind": "AFTER_SALES_PENDING_CLARIFICATION",
        "binding_checksum": payload["binding_checksum"],
    })
    return PendingClarificationContract.model_validate(payload, strict=True)


def _latest_decision(session, case_id: str) -> PolicyDecision | None:
    return session.scalar(
        select(PolicyDecision)
        .where(PolicyDecision.case_id == case_id)
        .order_by(PolicyDecision.case_revision.desc(), PolicyDecision.decided_at.desc())
        .limit(1)
    )


def _bound_plan(session, row: AfterSalesPendingClarification) -> AfterSalesCasePlan | None:
    return session.scalar(
        select(AfterSalesCasePlan).where(
            AfterSalesCasePlan.case_id == row.case_id,
            AfterSalesCasePlan.case_revision == row.case_revision,
            AfterSalesCasePlan.decision_id == row.policy_decision_id,
        )
    )


def _row_payload(row: AfterSalesPendingClarification) -> dict:
    return {
        "schema_version": 1,
        "binding_version": CLARIFICATION_BINDING_VERSION,
        "clarification_id": row.clarification_id,
        "case_id": row.case_id,
        "requirement_id": row.requirement_id,
        "question_id": row.question_id,
        "question": row.question,
        "case_revision": row.case_revision,
        "policy_version_id": row.policy_version_id,
        "policy_checksum": row.policy_checksum,
        "policy_decision_id": row.policy_decision_id,
        "state_id": row.state_id,
        "graph_id": row.graph_id,
        "portfolio_id": row.portfolio_id,
        "clarification_decision_id": row.clarification_decision_id,
        "selection_id": row.selection_id,
        "question_checksum": row.question_checksum,
        "binding_checksum": row.binding_checksum,
    }


def _contract_is_intact(payload: dict) -> bool:
    try:
        contract = PendingClarificationContract.model_validate(payload, strict=True)
    except (TypeError, ValueError):
        return False
    expected_question = _checksum(_question_basis(
        question_id=contract.question_id,
        requirement_id=contract.requirement_id,
        question=contract.question,
    ))
    expected_binding = _checksum(_binding_basis(contract.model_dump(mode="json")))
    expected_id = _checksum({
        "kind": "AFTER_SALES_PENDING_CLARIFICATION",
        "binding_checksum": expected_binding,
    })
    return (
        contract.question_checksum == expected_question
        and contract.binding_checksum == expected_binding
        and contract.clarification_id == expected_id
    )


def is_current_open_clarification(session, row: AfterSalesPendingClarification) -> bool:
    if row.status != "OPEN" or not _contract_is_intact(_row_payload(row)):
        return False
    case = session.get(AfterSalesCase, row.case_id)
    decision = _latest_decision(session, row.case_id)
    policy = session.get(PolicyVersion, row.policy_version_id)
    plan = _bound_plan(session, row)
    if not case or not decision or not policy or not plan:
        return False
    if (
        case.origin_session_id != row.session_id
        or case.revision != row.case_revision
        or decision.decision_id != row.policy_decision_id
        or decision.case_revision != row.case_revision
        or decision.policy_version_id != row.policy_version_id
        or policy.checksum != row.policy_checksum
    ):
        return False
    requirements = decision.facts.get("evidence_requirements", []) if isinstance(decision.facts, dict) else []
    requirement = next(
        (item for item in requirements if isinstance(item, dict) and item.get("requirement_id") == row.requirement_id),
        None,
    )
    if not requirement or requirement.get("status") == "SATISFIED":
        return False
    actions = plan.payload.get("next_actions", []) if isinstance(plan.payload, dict) else []
    return any(
        isinstance(action, dict)
        and action.get("kind") == "REQUEST_CUSTOMER_EVIDENCE"
        and action.get("requirement_id") == row.requirement_id
        and action.get("question") == row.question
        for action in actions
    )


def clarification_projection(row: AfterSalesPendingClarification) -> dict:
    return {
        **_row_payload(row),
        "status": row.status,
        "source_run_id": row.source_run_id,
        "created_at": row.created_at.isoformat(),
        "answer_run_id": row.answer_run_id,
        "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
    }


def clarification_context_projection(row: AfterSalesPendingClarification) -> dict:
    """Bounded dialogue context; the private row id is retained for runtime normalization only."""

    return {
        "clarification_id": row.clarification_id,
        "status": row.status,
        "question_id": row.question_id,
        "requirement_id": row.requirement_id,
        "question": row.question,
        "case_revision": row.case_revision,
        "policy_version_id": row.policy_version_id,
        "policy_checksum": row.policy_checksum,
        "policy_decision_id": row.policy_decision_id,
        "question_checksum": row.question_checksum,
    }


def persist_open_clarification(
    session,
    *,
    payload: dict,
    session_id: str,
    source_run_id: str,
    now: datetime,
) -> dict:
    try:
        contract = PendingClarificationContract.model_validate(payload, strict=True)
    except (TypeError, ValueError) as exc:
        raise AgentError("CLARIFICATION_BINDING_INVALID", "待澄清问题契约无效", retryable=False) from exc
    if not _contract_is_intact(contract.model_dump(mode="json")):
        raise AgentError("CLARIFICATION_BINDING_INVALID", "待澄清问题内容指纹不一致", retryable=False)
    case = session.get(AfterSalesCase, contract.case_id)
    if case is None or case.origin_session_id != session_id:
        raise AgentError("CLARIFICATION_CONTEXT_FORBIDDEN", "待澄清问题不属于当前会话", retryable=False)
    row = session.get(AfterSalesPendingClarification, contract.clarification_id)
    if row is None:
        row = AfterSalesPendingClarification(
            **{
                key: value
                for key, value in contract.model_dump(mode="json").items()
                if key not in {"schema_version", "binding_version"}
            },
            session_id=session_id,
            source_run_id=source_run_id,
            status="OPEN",
            created_at=now,
        )
        session.add(row)
        session.flush()
    elif row.session_id != session_id or _row_payload(row) != contract.model_dump(mode="json"):
        raise AgentError("CLARIFICATION_IDEMPOTENCY_CONFLICT", "待澄清问题编号已绑定不同内容", retryable=False)
    if not is_current_open_clarification(session, row):
        raise AgentError("CLARIFICATION_BINDING_STALE", "待澄清问题已不匹配当前案件与政策", retryable=False)
    others = session.scalars(select(AfterSalesPendingClarification).where(
        AfterSalesPendingClarification.session_id == session_id,
        AfterSalesPendingClarification.case_id == contract.case_id,
        AfterSalesPendingClarification.status == "OPEN",
        AfterSalesPendingClarification.clarification_id != contract.clarification_id,
    )).all()
    for other in others:
        other.status = "SUPERSEDED"
        other.resolved_at = now
    return clarification_projection(row)


def current_open_for_case(session, *, session_id: str, case_id: str) -> dict | None:
    rows = session.scalars(
        select(AfterSalesPendingClarification).where(
            AfterSalesPendingClarification.session_id == session_id,
            AfterSalesPendingClarification.case_id == case_id,
            AfterSalesPendingClarification.status == "OPEN",
        ).order_by(
            AfterSalesPendingClarification.created_at.desc(),
            AfterSalesPendingClarification.clarification_id.desc(),
        )
    ).all()
    row = next((item for item in rows if is_current_open_clarification(session, item)), None)
    return clarification_context_projection(row) if row else None


def reconcile_session_clarifications(session, *, session_id: str, now: datetime) -> dict | None:
    rows = session.scalars(
        select(AfterSalesPendingClarification).where(
            AfterSalesPendingClarification.session_id == session_id,
            AfterSalesPendingClarification.status == "OPEN",
        ).order_by(
            AfterSalesPendingClarification.created_at.desc(),
            AfterSalesPendingClarification.clarification_id.desc(),
        )
    ).all()
    current = None
    for row in rows:
        if is_current_open_clarification(session, row):
            current = current or row
        else:
            row.status = "SUPERSEDED"
            row.resolved_at = now
    return clarification_projection(current) if current else None


def require_answer_binding(
    session,
    *,
    clarification_id: str,
    session_id: str,
    case_id: str,
    requirement_id: str | None,
    case_revision: int,
    answer_quote: str | None,
) -> AfterSalesPendingClarification:
    row = session.get(AfterSalesPendingClarification, clarification_id)
    if (
        row is None
        or row.session_id != session_id
        or row.case_id != case_id
        or row.requirement_id != requirement_id
        or row.case_revision != case_revision
        or not answer_quote
        or not is_current_open_clarification(session, row)
    ):
        raise AgentError("CLARIFICATION_BINDING_STALE", "回答引用的问题已失效，请基于当前案件重新确认", retryable=False)
    return row


def bind_answer(
    row: AfterSalesPendingClarification,
    *,
    answer_run_id: str,
    answer_quote: str,
    resolved_at: datetime,
) -> dict:
    row.status = "ANSWERED"
    row.answer_run_id = answer_run_id
    row.answer_quote = answer_quote
    row.resolved_at = resolved_at
    return {
        "binding_version": CLARIFICATION_BINDING_VERSION,
        "clarification_id": row.clarification_id,
        "question_id": row.question_id,
        "case_id": row.case_id,
        "requirement_id": row.requirement_id,
        "asked_case_revision": row.case_revision,
        "answer_run_id": answer_run_id,
        "answer_source_ref": f"agent_runs:{answer_run_id}:request",
        "status": "ANSWERED",
        "authorization_granted": False,
    }
