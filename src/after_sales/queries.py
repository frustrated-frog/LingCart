"""Read-only projections for customer and engineering after-sales views."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.after_sales.contracts import (
    CaseDetail,
    CaseListItem,
    CaseRefundReference,
    CaseRunReference,
    CaseStatus,
    PolicyOutcome,
    PolicyVersionProjection,
)
from src.after_sales.service import AfterSalesCaseService, AfterSalesDomainError
from src.after_sales.policy import AfterSalesPolicyEngine
from src.backend.db import (
    AfterSalesCase,
    AfterSalesCaseRefund,
    AfterSalesCaseRun,
    PolicyVersion,
    Refund,
)


def _current_projection(session, case):
    decision = case.latest_decision
    if not decision or case.status in {"SUBMITTED", "RESOLVED", "REJECTED", "CANCELLED"}:
        return case
    reason = None
    try:
        current = AfterSalesPolicyEngine(session).evaluate(case.case_id)
        if current.input_hash != decision['input_hash']:
            reason = 'POLICY_INPUT_CHANGED'
    except AfterSalesDomainError as exc:
        reason = exc.code
    if reason or not decision.get('valid', False):
        return case.model_copy(update={
            'latest_decision': {**decision, 'valid': False, 'stale_reason': reason or 'EVIDENCE_CHANGED'},
            'plans': tuple({**plan, 'valid': False} for plan in case.plans),
        })
    return case


def list_cases(
    session: Session,
    *,
    order_id: int | None = None,
    origin_session_id: str | None = None,
    status: CaseStatus | None = None,
    limit: int = 50,
) -> tuple[CaseListItem, ...]:
    statement = select(AfterSalesCase)
    if order_id is not None:
        statement = statement.where(AfterSalesCase.order_id == order_id)
    if origin_session_id is not None:
        statement = statement.where(AfterSalesCase.origin_session_id == origin_session_id)
    if status is not None:
        statement = statement.where(AfterSalesCase.status == status.value)
    rows = session.scalars(
        statement.order_by(AfterSalesCase.updated_at.desc(), AfterSalesCase.case_id).limit(limit)
    ).all()
    service = AfterSalesCaseService(session)
    links = {
        link.case_id: link
        for link in session.scalars(
            select(AfterSalesCaseRefund).where(
                AfterSalesCaseRefund.case_id.in_([row.case_id for row in rows])
            )
        ).all()
    } if rows else {}
    items: list[CaseListItem] = []
    for row in rows:
        projection = _current_projection(session, service.get_case(row.case_id))
        decision = projection.latest_decision or {}
        outcome = decision.get("outcome")
        items.append(
            CaseListItem(
                case_id=projection.case_id,
                case_type=projection.case_type,
                order_id=projection.order_id,
                order_item_id=projection.order_item_id,
                requested_resolution=projection.requested_resolution,
                reason_text=projection.reason_text,
                status=projection.status,
                revision=projection.revision,
                origin_session_id=projection.origin_session_id,
                opened_at=projection.opened_at,
                updated_at=projection.updated_at,
                policy_outcome=PolicyOutcome(outcome) if outcome and decision.get('valid', True) else None,
                policy_reason_codes=tuple(decision.get("reason_codes") or ()),
                policy_version_id=decision.get("policy_version_id"),
                refund_id=links[row.case_id].refund_id if row.case_id in links else None,
            )
        )
    return tuple(items)


def get_case_detail(session: Session, case_id: str) -> CaseDetail:
    service = AfterSalesCaseService(session)
    case = _current_projection(session, service.get_case(case_id))
    run_rows = session.scalars(
        select(AfterSalesCaseRun)
        .where(AfterSalesCaseRun.case_id == case_id)
        .order_by(AfterSalesCaseRun.linked_at, AfterSalesCaseRun.link_id)
    ).all()
    link = session.scalar(
        select(AfterSalesCaseRefund).where(AfterSalesCaseRefund.case_id == case_id)
    )
    refund = session.get(Refund, link.refund_id) if link else None
    decision = case.latest_decision or {}
    policy = session.get(PolicyVersion, decision.get("policy_version_id")) if decision else None
    return CaseDetail(
        case=case,
        runs=tuple(
            CaseRunReference(
                run_id=row.run_id,
                session_id=row.session_id,
                role=row.role,
                linked_at=row.linked_at,
            )
            for row in run_rows
        ),
        refund=(
            CaseRefundReference(
                refund_id=link.refund_id,
                status=refund.status if refund else "UNKNOWN",
                reason=refund.reason if refund else None,
                policy_decision_id=link.policy_decision_id,
                submitted_run_id=link.submitted_run_id,
                linked_at=link.linked_at,
            )
            if link
            else None
        ),
        policy=(
            PolicyVersionProjection(
                policy_version_id=policy.policy_version_id,
                policy_id=policy.policy_id,
                version=policy.version,
                status=policy.status,
                effective_from=policy.effective_from,
                effective_until=policy.effective_until,
                rules=policy.rules,
                source_ref=policy.source_ref,
                checksum=policy.checksum,
            )
            if policy
            else None
        ),
    )
