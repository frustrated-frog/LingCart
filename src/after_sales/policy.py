"""Deterministic policy evaluation over authoritative business facts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from src.after_sales.contracts import PolicyEvaluation, PolicyOutcome
from src.after_sales.service import AfterSalesDomainError
from src.backend.db import (
    AfterSalesCase,
    AfterSalesEvidence,
    Order,
    OrderItem,
    Payment,
    PolicyVersion,
    Refund,
    ShipmentEvent,
)


EVALUATOR_VERSION = "after-sales-policy-v1"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _checksum(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


class AfterSalesPolicyEngine:
    """Produces a verdict; it never mutates a case or executes a side effect."""

    def __init__(self, session: Session):
        self.session = session

    def evaluate(self, case_id: str, *, as_of: datetime | None = None) -> PolicyEvaluation:
        evaluated_at = _naive_utc(as_of or _utcnow())
        case = self.session.get(AfterSalesCase, case_id)
        if case is None:
            raise AfterSalesDomainError("CASE_NOT_FOUND", "没有找到售后案件")
        policy = self._applicable_policy(case.case_type, evaluated_at)
        rules = self._validated_rules(policy)

        order = self.session.get(Order, case.order_id)
        item = self.session.get(OrderItem, case.order_item_id) if case.order_item_id else None
        payment = self.session.scalar(select(Payment).where(Payment.order_id == case.order_id))
        refund = self.session.scalar(select(Refund).where(Refund.order_id == case.order_id))
        shipment_events = self.session.scalars(
            select(ShipmentEvent)
            .where(ShipmentEvent.order_id == case.order_id)
            .order_by(ShipmentEvent.occurred_at, ShipmentEvent.event_id)
        ).all()
        active_evidence = self.session.scalars(
            select(AfterSalesEvidence)
            .where(
                AfterSalesEvidence.case_id == case_id,
                AfterSalesEvidence.status == "ACTIVE",
                AfterSalesEvidence.observed_at <= evaluated_at,
            )
            .order_by(AfterSalesEvidence.observed_at, AfterSalesEvidence.evidence_id)
        ).all()
        delivered_event = next(
            (
                event
                for event in reversed(shipment_events)
                if any(token in event.description.casefold() for token in ("签收", "delivered"))
            ),
            None,
        )
        amount_minor = item.unit_price_minor * item.quantity if item else None
        facts = {
            "case_id": case.case_id,
            "case_type": case.case_type,
            "requested_resolution": case.requested_resolution,
            "reason_code": case.reason_code,
            "reason_text_hash": _checksum(case.reason_text),
            "order_id": case.order_id,
            "order_item_id": case.order_item_id,
            "order_status": order.status if order else None,
            "payment_status": payment.status if payment else None,
            "refund_status": refund.status if refund else None,
            "delivery_date": delivered_event.occurred_at.date().isoformat() if delivered_event else None,
            "amount_minor": amount_minor,
            "currency": item.currency if item else None,
            "active_evidence_types": sorted({evidence.evidence_type for evidence in active_evidence}),
            "as_of_date": evaluated_at.date().isoformat(),
        }
        evidence_refs = self._evidence_refs(
            order=order,
            item=item,
            payment=payment,
            refund=refund,
            delivered_event=delivered_event,
            active_evidence=active_evidence,
        )
        outcome, reason_codes, allowed_actions = self._decide(facts, rules, delivered_event, evaluated_at)
        input_hash = _checksum(
            {
                "policy_version_id": policy.policy_version_id,
                "policy_checksum": policy.checksum,
                "facts": facts,
                "evidence_refs": evidence_refs,
                "evaluator_version": EVALUATOR_VERSION,
            }
        )
        return PolicyEvaluation(
            policy_version_id=policy.policy_version_id,
            policy_checksum=policy.checksum,
            outcome=outcome,
            reason_codes=reason_codes,
            facts=facts,
            evidence_refs=evidence_refs,
            allowed_actions=allowed_actions,
            evaluator_version=EVALUATOR_VERSION,
            evaluated_at=evaluated_at,
            input_hash=input_hash,
        )

    def _applicable_policy(self, case_type: str, as_of: datetime) -> PolicyVersion:
        candidates = self.session.scalars(
            select(PolicyVersion)
            .where(
                PolicyVersion.policy_type == "AFTER_SALES",
                PolicyVersion.status == "ACTIVE",
                PolicyVersion.effective_from <= as_of,
                or_(PolicyVersion.effective_until.is_(None), PolicyVersion.effective_until > as_of),
            )
            .order_by(PolicyVersion.effective_from.desc(), PolicyVersion.policy_version_id)
        ).all()
        matched = [
            policy
            for policy in candidates
            if case_type in (policy.rules or {}).get("supported_case_types", [])
        ]
        if not matched:
            raise AfterSalesDomainError("POLICY_NOT_FOUND", f"当前没有适用于 {case_type} 的有效政策")
        if len(matched) > 1:
            raise AfterSalesDomainError(
                "POLICY_VERSION_AMBIGUOUS",
                "同一售后诉求同时匹配多个有效政策版本，拒绝猜测适用政策",
            )
        return matched[0]

    @staticmethod
    def _validated_rules(policy: PolicyVersion) -> dict[str, Any]:
        rules = policy.rules
        if not isinstance(rules, dict) or _checksum(rules) != policy.checksum:
            raise AfterSalesDomainError("POLICY_INTEGRITY_ERROR", "政策内容与保存的指纹不一致")
        required = {
            "supported_case_types": list,
            "eligible_order_statuses": list,
            "eligible_payment_statuses": list,
            "blocking_refund_statuses": list,
            "return_window_days": int,
            "manual_review_amount_minor": int,
            "required_evidence_by_reason": dict,
        }
        if any(not isinstance(rules.get(key), value_type) for key, value_type in required.items()):
            raise AfterSalesDomainError("POLICY_SCHEMA_INVALID", "政策规则结构不完整或类型无效")
        if rules["return_window_days"] < 0 or rules["manual_review_amount_minor"] < 0:
            raise AfterSalesDomainError("POLICY_SCHEMA_INVALID", "政策期限或金额阈值不能为负数")
        return rules

    @staticmethod
    def _decide(
        facts: dict[str, Any],
        rules: dict[str, Any],
        delivered_event: ShipmentEvent | None,
        evaluated_at: datetime,
    ) -> tuple[PolicyOutcome, tuple[str, ...], tuple[str, ...]]:
        if facts["order_status"] is None or facts["payment_status"] is None or facts["amount_minor"] is None:
            return (
                PolicyOutcome.MANUAL_REVIEW,
                ("AUTHORITATIVE_BUSINESS_DATA_MISSING",),
                ("ADD_EVIDENCE", "ESCALATE_TO_HUMAN"),
            )
        if facts["order_status"] not in rules["eligible_order_statuses"]:
            return (
                PolicyOutcome.INELIGIBLE,
                ("ORDER_STATUS_NOT_ELIGIBLE",),
                ("VIEW_POLICY", "ESCALATE_TO_HUMAN", "CLOSE_CASE"),
            )
        if facts["payment_status"] not in rules["eligible_payment_statuses"]:
            return (
                PolicyOutcome.INELIGIBLE,
                ("PAYMENT_STATUS_NOT_ELIGIBLE",),
                ("VIEW_POLICY", "ESCALATE_TO_HUMAN", "CLOSE_CASE"),
            )
        if facts["refund_status"] in rules["blocking_refund_statuses"]:
            return (
                PolicyOutcome.INELIGIBLE,
                ("REFUND_ALREADY_EXISTS",),
                ("VIEW_EXISTING_REFUND", "ESCALATE_TO_HUMAN", "CLOSE_CASE"),
            )
        if delivered_event is None:
            return (
                PolicyOutcome.MANUAL_REVIEW,
                ("DELIVERY_EVIDENCE_MISSING",),
                ("ADD_EVIDENCE", "ESCALATE_TO_HUMAN"),
            )
        elapsed_days = (evaluated_at.date() - delivered_event.occurred_at.date()).days
        if elapsed_days < 0:
            return (
                PolicyOutcome.MANUAL_REVIEW,
                ("DELIVERY_TIMESTAMP_IN_FUTURE",),
                ("ESCALATE_TO_HUMAN",),
            )
        if elapsed_days > rules["return_window_days"]:
            return (
                PolicyOutcome.INELIGIBLE,
                ("RETURN_WINDOW_EXPIRED",),
                ("VIEW_POLICY", "ESCALATE_TO_HUMAN", "CLOSE_CASE"),
            )
        if not facts["reason_code"]:
            return (
                PolicyOutcome.NEEDS_EVIDENCE,
                ("AFTER_SALES_REASON_REQUIRED",),
                ("CLARIFY_REASON", "CANCEL_CASE", "ESCALATE_TO_HUMAN"),
            )
        if facts["reason_code"] not in rules["required_evidence_by_reason"]:
            return (
                PolicyOutcome.MANUAL_REVIEW,
                ("AFTER_SALES_REASON_NOT_SUPPORTED",),
                ("ESCALATE_TO_HUMAN", "CANCEL_CASE"),
            )
        required_evidence = set(
            rules["required_evidence_by_reason"][facts["reason_code"]]
        )
        missing = sorted(required_evidence - set(facts["active_evidence_types"]))
        if missing:
            return (
                PolicyOutcome.NEEDS_EVIDENCE,
                tuple(f"MISSING_EVIDENCE:{evidence_type}" for evidence_type in missing),
                ("ADD_EVIDENCE", "CANCEL_CASE", "ESCALATE_TO_HUMAN"),
            )
        if facts["amount_minor"] >= rules["manual_review_amount_minor"]:
            return (
                PolicyOutcome.MANUAL_REVIEW,
                ("AMOUNT_REQUIRES_MANUAL_REVIEW",),
                ("WAIT_FOR_HUMAN", "ADD_EVIDENCE"),
            )
        return (
            PolicyOutcome.ELIGIBLE,
            ("POLICY_REQUIREMENTS_SATISFIED",),
            ("REQUEST_CUSTOMER_CONFIRMATION", "ESCALATE_TO_HUMAN"),
        )

    @staticmethod
    def _evidence_refs(
        *,
        order: Order | None,
        item: OrderItem | None,
        payment: Payment | None,
        refund: Refund | None,
        delivered_event: ShipmentEvent | None,
        active_evidence: list[AfterSalesEvidence],
    ) -> tuple[str, ...]:
        refs: list[str] = []
        if order:
            refs.append(f"orders:{order.order_id}")
        if item:
            refs.append(f"order_items:{item.item_id}")
        if payment:
            refs.append(f"payments:{payment.payment_id}")
        if refund:
            refs.append(f"refunds:{refund.refund_id}")
        if delivered_event:
            refs.append(f"shipment_events:{delivered_event.event_id}")
        refs.extend(f"after_sales_evidence:{evidence.evidence_id}" for evidence in active_evidence)
        return tuple(refs)
