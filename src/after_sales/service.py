"""Transactional case ledger with optimistic concurrency and idempotent writes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from src.after_sales.contracts import (
    ActorType,
    CaseCreateCommand,
    CaseProjection,
    CaseStatus,
    CaseType,
    EvidenceCommand,
    PolicyEvaluation,
    PolicyOutcome,
    RequestedResolution,
)
from src.backend.db import (
    AfterSalesCase,
    AfterSalesCaseEvent,
    AfterSalesCaseRun,
    AfterSalesEvidence,
    AgentRun,
    Order,
    OrderItem,
    PolicyDecision,
    PolicyVersion,
)


class AfterSalesDomainError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


_TERMINAL_STATUSES = {CaseStatus.RESOLVED, CaseStatus.REJECTED, CaseStatus.CANCELLED}
_TRANSITIONS = {
    CaseStatus.OPEN: {
        CaseStatus.AWAITING_EVIDENCE,
        CaseStatus.AWAITING_CONFIRMATION,
        CaseStatus.MANUAL_REVIEW,
        CaseStatus.REJECTED,
        CaseStatus.CANCELLED,
    },
    CaseStatus.AWAITING_EVIDENCE: {
        CaseStatus.OPEN,
        CaseStatus.AWAITING_CONFIRMATION,
        CaseStatus.MANUAL_REVIEW,
        CaseStatus.REJECTED,
        CaseStatus.CANCELLED,
    },
    CaseStatus.AWAITING_CONFIRMATION: {
        CaseStatus.OPEN,
        CaseStatus.AWAITING_EVIDENCE,
        CaseStatus.SUBMITTED,
        CaseStatus.MANUAL_REVIEW,
        CaseStatus.REJECTED,
        CaseStatus.CANCELLED,
    },
    CaseStatus.SUBMITTED: {
        CaseStatus.MANUAL_REVIEW,
        CaseStatus.RESOLVED,
        CaseStatus.REJECTED,
    },
    CaseStatus.MANUAL_REVIEW: {
        CaseStatus.AWAITING_EVIDENCE,
        CaseStatus.AWAITING_CONFIRMATION,
        CaseStatus.RESOLVED,
        CaseStatus.REJECTED,
        CaseStatus.CANCELLED,
    },
    CaseStatus.RESOLVED: set(),
    CaseStatus.REJECTED: set(),
    CaseStatus.CANCELLED: set(),
}
_RESOLUTION_BY_CASE_TYPE = {
    CaseType.RETURN: {RequestedResolution.RETURN_AND_REFUND, RequestedResolution.REFUND},
    CaseType.REFUND: {RequestedResolution.REFUND},
    CaseType.EXCHANGE: {RequestedResolution.REPLACEMENT},
    CaseType.REPAIR: {RequestedResolution.REPAIR},
    CaseType.DELIVERY_ISSUE: {
        RequestedResolution.INFORMATION,
        RequestedResolution.REFUND,
        RequestedResolution.REPLACEMENT,
    },
}
_STATUS_BY_OUTCOME = {
    PolicyOutcome.ELIGIBLE: CaseStatus.AWAITING_CONFIRMATION,
    PolicyOutcome.NEEDS_EVIDENCE: CaseStatus.AWAITING_EVIDENCE,
    PolicyOutcome.INELIGIBLE: CaseStatus.REJECTED,
    PolicyOutcome.MANUAL_REVIEW: CaseStatus.MANUAL_REVIEW,
}


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


class AfterSalesCaseService:
    """Mutates a case only inside the caller-owned database transaction."""

    def __init__(self, session: Session):
        self.session = session

    def find_active_case(
        self,
        *,
        order_id: int,
        origin_session_id: str | None,
        case_type: CaseType,
        requested_resolution: RequestedResolution,
        reason_text: str,
        order_item_id: str | None = None,
    ) -> CaseProjection | None:
        """Find one exact, non-terminal business case for a later Agent run.

        A conversation identifier is mandatory.  Reusing all anonymous cases
        would cross customer boundaries, while choosing among several matches
        would hide a domain conflict.
        """

        if origin_session_id is None:
            return None
        statement = select(AfterSalesCase).where(
            AfterSalesCase.order_id == order_id,
            AfterSalesCase.origin_session_id == origin_session_id,
            AfterSalesCase.case_type == case_type.value,
            AfterSalesCase.requested_resolution == requested_resolution.value,
            AfterSalesCase.reason_text == reason_text,
            AfterSalesCase.status.not_in([status.value for status in _TERMINAL_STATUSES]),
        )
        if order_item_id is not None:
            statement = statement.where(AfterSalesCase.order_item_id == order_item_id)
        matches = self.session.scalars(
            statement.order_by(AfterSalesCase.updated_at.desc(), AfterSalesCase.case_id).limit(2)
        ).all()
        if len(matches) > 1:
            raise AfterSalesDomainError(
                "CASE_CONTEXT_AMBIGUOUS",
                "当前会话存在多个相同诉求的进行中案件，请明确案件编号",
            )
        return self.get_case(matches[0].case_id) if matches else None

    def open_case(
        self,
        command: CaseCreateCommand,
        *,
        actor_type: ActorType = ActorType.CUSTOMER,
        actor_id: str | None = None,
        now: datetime | None = None,
    ) -> CaseProjection:
        existing = self.session.scalar(
            select(AfterSalesCase).where(AfterSalesCase.request_key == command.request_key)
        )
        if existing is not None:
            if self._case_identity(existing) != self._command_identity(
                command,
                resolved_order_item_id=existing.order_item_id if command.order_item_id is None else command.order_item_id,
            ):
                raise AfterSalesDomainError(
                    "CASE_REQUEST_CONFLICT",
                    "同一案件请求编号不能用于不同的售后诉求",
                )
            return self.get_case(existing.case_id)

        order = self.session.get(Order, command.order_id)
        if order is None:
            raise AfterSalesDomainError("ORDER_NOT_FOUND", f"没有找到订单 {command.order_id}")
        order_item_id = self._resolve_order_item(command.order_id, command.order_item_id)
        allowed_resolutions = _RESOLUTION_BY_CASE_TYPE[command.case_type]
        if command.requested_resolution not in allowed_resolutions:
            raise AfterSalesDomainError(
                "CASE_RESOLUTION_MISMATCH",
                f"{command.case_type.value} 案件不支持 {command.requested_resolution.value} 处理目标",
            )

        opened_at = _naive_utc(now or _utcnow())
        case = AfterSalesCase(
            case_id=uuid.uuid4().hex,
            request_key=command.request_key,
            case_type=command.case_type.value,
            order_id=command.order_id,
            order_item_id=order_item_id,
            requested_resolution=command.requested_resolution.value,
            reason_code=command.reason_code,
            reason_text=command.reason_text,
            status=CaseStatus.OPEN.value,
            revision=1,
            origin_session_id=command.origin_session_id,
            context=command.context,
            opened_at=opened_at,
            updated_at=opened_at,
        )
        self.session.add(case)
        self.session.flush()
        self._append_event(
            case,
            event_type="CASE_OPENED",
            idempotency_key=f"open:{command.request_key}",
            actor_type=actor_type,
            actor_id=actor_id,
            payload={
                "case_type": case.case_type,
                "order_id": case.order_id,
                "order_item_id": case.order_item_id,
                "requested_resolution": case.requested_resolution,
                "reason_code": case.reason_code,
            },
            occurred_at=opened_at,
            revision=1,
        )
        self.session.flush()
        return self.get_case(case.case_id)

    def add_evidence(
        self,
        case_id: str,
        command: EvidenceCommand,
        *,
        idempotency_key: str,
        expected_revision: int,
        actor_type: ActorType = ActorType.CUSTOMER,
        actor_id: str | None = None,
        now: datetime | None = None,
    ) -> CaseProjection:
        case = self._case(case_id)
        previous = self._event_by_key(case_id, idempotency_key)
        content = {
            "evidence_type": command.evidence_type,
            "source_type": command.source_type,
            "source_ref": command.source_ref,
            "payload": command.payload,
            "observed_at": _naive_utc(command.observed_at).isoformat(),
        }
        evidence_checksum = _checksum(content)
        if previous is not None:
            self._assert_replay(previous, "EVIDENCE_ADDED", {**content, "checksum": evidence_checksum})
            return self.get_case(case_id)
        self._assert_mutable(case, expected_revision)

        duplicate = self.session.scalar(
            select(AfterSalesEvidence).where(
                AfterSalesEvidence.case_id == case_id,
                AfterSalesEvidence.source_ref == command.source_ref,
                AfterSalesEvidence.checksum == evidence_checksum,
            )
        )
        if duplicate is None:
            created_at = _naive_utc(now or _utcnow())
            duplicate = AfterSalesEvidence(
                evidence_id=uuid.uuid4().hex,
                case_id=case_id,
                evidence_type=command.evidence_type,
                source_type=command.source_type,
                source_ref=command.source_ref,
                payload=command.payload,
                checksum=evidence_checksum,
                status="ACTIVE",
                observed_at=_naive_utc(command.observed_at),
                created_at=created_at,
            )
            self.session.add(duplicate)
            self.session.flush()
        self._advance(
            case,
            target_status=CaseStatus(case.status),
            event_type="EVIDENCE_ADDED",
            idempotency_key=idempotency_key,
            actor_type=actor_type,
            actor_id=actor_id,
            payload={**content, "checksum": evidence_checksum},
            occurred_at=_naive_utc(now or _utcnow()),
            allow_same_status=True,
        )
        return self.get_case(case_id)

    def attach_run(
        self,
        case_id: str,
        run_id: str,
        *,
        session_id: str,
        role: str,
        idempotency_key: str,
        expected_revision: int,
        now: datetime | None = None,
    ) -> CaseProjection:
        case = self._case(case_id)
        previous = self._event_by_key(case_id, idempotency_key)
        payload = {"run_id": run_id, "session_id": session_id, "role": role}
        if previous is not None:
            self._assert_replay(previous, "RUN_LINKED", payload)
            return self.get_case(case_id)
        self._assert_mutable(case, expected_revision)
        run = self.session.get(AgentRun, run_id)
        if run is None or run.session_id != session_id:
            raise AfterSalesDomainError("RUN_NOT_FOUND", "运行不存在或不属于指定会话")
        link = self.session.scalar(
            select(AfterSalesCaseRun).where(
                AfterSalesCaseRun.case_id == case_id,
                AfterSalesCaseRun.run_id == run_id,
            )
        )
        if link is None:
            link = AfterSalesCaseRun(
                link_id=uuid.uuid4().hex,
                case_id=case_id,
                run_id=run_id,
                session_id=session_id,
                role=role,
                linked_at=_naive_utc(now or _utcnow()),
            )
            self.session.add(link)
            self.session.flush()
        elif link.role != role or link.session_id != session_id:
            raise AfterSalesDomainError("CASE_RUN_CONFLICT", "同一运行不能以不同身份重复绑定案件")
        self._advance(
            case,
            target_status=CaseStatus(case.status),
            event_type="RUN_LINKED",
            idempotency_key=idempotency_key,
            actor_type=ActorType.SYSTEM,
            actor_id=run_id,
            payload=payload,
            occurred_at=_naive_utc(now or _utcnow()),
            allow_same_status=True,
        )
        return self.get_case(case_id)

    def record_policy_decision(
        self,
        case_id: str,
        evaluation: PolicyEvaluation,
        *,
        idempotency_key: str,
        expected_revision: int,
    ) -> CaseProjection:
        case = self._case(case_id)
        previous_event = self._event_by_key(case_id, idempotency_key)
        existing = self.session.scalar(
            select(PolicyDecision).where(
                PolicyDecision.case_id == case_id,
                PolicyDecision.input_hash == evaluation.input_hash,
            )
        )
        if previous_event is not None:
            if existing is None:
                raise AfterSalesDomainError("CASE_EVENT_CORRUPT", "裁决事件缺少对应的不可变裁决记录")
            self._assert_replay(previous_event, "POLICY_DECIDED", self._decision_event_payload(existing))
            return self.get_case(case_id)
        self._assert_mutable(case, expected_revision)
        if existing is not None:
            raise AfterSalesDomainError(
                "POLICY_DECISION_IDEMPOTENCY_REQUIRED",
                "相同政策输入已存在裁决，重放时必须使用原幂等键",
            )
        policy = self.session.get(PolicyVersion, evaluation.policy_version_id)
        if policy is None or policy.checksum != evaluation.policy_checksum:
            raise AfterSalesDomainError("POLICY_VERSION_MISMATCH", "政策版本不存在或内容指纹不匹配")
        latest = self.session.scalar(
            select(PolicyDecision)
            .where(PolicyDecision.case_id == case_id)
            .order_by(PolicyDecision.case_revision.desc())
            .limit(1)
        )
        decision = PolicyDecision(
            decision_id=uuid.uuid4().hex,
            case_id=case_id,
            policy_version_id=evaluation.policy_version_id,
            case_revision=case.revision + 1,
            input_hash=evaluation.input_hash,
            outcome=evaluation.outcome.value,
            reason_codes=list(evaluation.reason_codes),
            facts=evaluation.facts,
            evidence_refs=list(evaluation.evidence_refs),
            allowed_actions=list(evaluation.allowed_actions),
            evaluator_version=evaluation.evaluator_version,
            decided_at=_naive_utc(evaluation.evaluated_at),
            supersedes_decision_id=latest.decision_id if latest else None,
        )
        self.session.add(decision)
        self.session.flush()
        self._advance(
            case,
            target_status=_STATUS_BY_OUTCOME[evaluation.outcome],
            event_type="POLICY_DECIDED",
            idempotency_key=idempotency_key,
            actor_type=ActorType.SYSTEM,
            actor_id=evaluation.evaluator_version,
            payload=self._decision_event_payload(decision),
            occurred_at=_naive_utc(evaluation.evaluated_at),
            allow_same_status=True,
        )
        return self.get_case(case_id)

    def transition(
        self,
        case_id: str,
        target_status: CaseStatus,
        *,
        event_type: str,
        idempotency_key: str,
        expected_revision: int,
        actor_type: ActorType,
        actor_id: str | None = None,
        payload: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> CaseProjection:
        case = self._case(case_id)
        event_payload = {"from_status": case.status, "to_status": target_status.value, **(payload or {})}
        previous = self._event_by_key(case_id, idempotency_key)
        if previous is not None:
            self._assert_replay(previous, event_type, event_payload)
            return self.get_case(case_id)
        self._assert_mutable(case, expected_revision)
        self._advance(
            case,
            target_status=target_status,
            event_type=event_type,
            idempotency_key=idempotency_key,
            actor_type=actor_type,
            actor_id=actor_id,
            payload=event_payload,
            occurred_at=_naive_utc(now or _utcnow()),
        )
        return self.get_case(case_id)

    def get_case(self, case_id: str) -> CaseProjection:
        case = self._case(case_id)
        evidence = self.session.scalars(
            select(AfterSalesEvidence)
            .where(AfterSalesEvidence.case_id == case_id)
            .order_by(AfterSalesEvidence.created_at, AfterSalesEvidence.evidence_id)
        ).all()
        events = self.session.scalars(
            select(AfterSalesCaseEvent)
            .where(AfterSalesCaseEvent.case_id == case_id)
            .order_by(AfterSalesCaseEvent.sequence)
        ).all()
        decision = self.session.scalar(
            select(PolicyDecision)
            .where(PolicyDecision.case_id == case_id)
            .order_by(PolicyDecision.case_revision.desc())
            .limit(1)
        )
        return CaseProjection(
            case_id=case.case_id,
            request_key=case.request_key,
            case_type=case.case_type,
            order_id=case.order_id,
            order_item_id=case.order_item_id,
            requested_resolution=case.requested_resolution,
            reason_code=case.reason_code,
            reason_text=case.reason_text,
            status=case.status,
            revision=case.revision,
            origin_session_id=case.origin_session_id,
            context=case.context or {},
            opened_at=case.opened_at,
            updated_at=case.updated_at,
            closed_at=case.closed_at,
            latest_decision=self._decision_dict(decision) if decision else None,
            evidence=tuple(
                {
                    "evidence_id": item.evidence_id,
                    "evidence_type": item.evidence_type,
                    "source_type": item.source_type,
                    "source_ref": item.source_ref,
                    "payload": item.payload,
                    "checksum": item.checksum,
                    "status": item.status,
                    "observed_at": item.observed_at,
                    "created_at": item.created_at,
                }
                for item in evidence
            ),
            events=tuple(
                {
                    "event_id": item.event_id,
                    "sequence": item.sequence,
                    "case_revision": item.case_revision,
                    "event_type": item.event_type,
                    "actor_type": item.actor_type,
                    "actor_id": item.actor_id,
                    "idempotency_key": item.idempotency_key,
                    "payload": item.payload,
                    "occurred_at": item.occurred_at,
                }
                for item in events
            ),
        )

    def _resolve_order_item(self, order_id: int, requested_item_id: str | None) -> str:
        items = self.session.scalars(
            select(OrderItem).where(OrderItem.order_id == order_id).order_by(OrderItem.line_number)
        ).all()
        if requested_item_id is not None:
            item = self.session.get(OrderItem, requested_item_id)
            if item is None or item.order_id != order_id:
                raise AfterSalesDomainError("ORDER_ITEM_NOT_FOUND", "售后商品不属于指定订单")
            return item.item_id
        if len(items) == 1:
            return items[0].item_id
        if not items:
            raise AfterSalesDomainError("ORDER_ITEM_NOT_FOUND", "订单缺少可建立售后案件的行项目")
        raise AfterSalesDomainError("ORDER_ITEM_REQUIRED", "订单包含多个商品，请明确售后商品")

    def _advance(
        self,
        case: AfterSalesCase,
        *,
        target_status: CaseStatus,
        event_type: str,
        idempotency_key: str,
        actor_type: ActorType,
        actor_id: str | None,
        payload: dict[str, Any],
        occurred_at: datetime,
        allow_same_status: bool = False,
    ) -> None:
        current = CaseStatus(case.status)
        if target_status == current:
            if not allow_same_status:
                raise AfterSalesDomainError("CASE_TRANSITION_INVALID", "案件状态没有发生变化")
        elif target_status not in _TRANSITIONS[current]:
            raise AfterSalesDomainError(
                "CASE_TRANSITION_INVALID",
                f"案件不能从 {current.value} 转为 {target_status.value}",
            )
        previous_revision = case.revision
        next_revision = previous_revision + 1
        closed_at = occurred_at if target_status in _TERMINAL_STATUSES else None
        result = self.session.execute(
            update(AfterSalesCase)
            .where(
                AfterSalesCase.case_id == case.case_id,
                AfterSalesCase.revision == previous_revision,
            )
            .values(
                status=target_status.value,
                revision=next_revision,
                updated_at=occurred_at,
                closed_at=closed_at,
            )
        )
        if result.rowcount != 1:
            raise AfterSalesDomainError(
                "CASE_REVISION_CONFLICT",
                "案件已被其他执行更新，请读取最新版本后重试",
                retryable=True,
            )
        self._append_event(
            case,
            event_type=event_type,
            idempotency_key=idempotency_key,
            actor_type=actor_type,
            actor_id=actor_id,
            payload=payload,
            occurred_at=occurred_at,
            revision=next_revision,
        )
        self.session.flush()
        self.session.expire(case)

    def _append_event(
        self,
        case: AfterSalesCase,
        *,
        event_type: str,
        idempotency_key: str,
        actor_type: ActorType,
        actor_id: str | None,
        payload: dict[str, Any],
        occurred_at: datetime,
        revision: int,
    ) -> None:
        self.session.add(
            AfterSalesCaseEvent(
                event_id=uuid.uuid4().hex,
                case_id=case.case_id,
                sequence=revision,
                case_revision=revision,
                event_type=event_type,
                actor_type=actor_type.value,
                actor_id=actor_id,
                idempotency_key=idempotency_key,
                payload=payload,
                occurred_at=occurred_at,
            )
        )

    def _assert_mutable(self, case: AfterSalesCase, expected_revision: int) -> None:
        if CaseStatus(case.status).terminal:
            raise AfterSalesDomainError("CASE_TERMINAL", "已终止案件不能继续修改")
        if case.revision != expected_revision:
            raise AfterSalesDomainError(
                "CASE_REVISION_CONFLICT",
                f"案件版本已变更，当前版本为 {case.revision}",
                retryable=True,
            )

    def _case(self, case_id: str) -> AfterSalesCase:
        case = self.session.get(AfterSalesCase, case_id)
        if case is None:
            raise AfterSalesDomainError("CASE_NOT_FOUND", "没有找到售后案件")
        return case

    def _event_by_key(self, case_id: str, key: str) -> AfterSalesCaseEvent | None:
        return self.session.scalar(
            select(AfterSalesCaseEvent).where(
                AfterSalesCaseEvent.case_id == case_id,
                AfterSalesCaseEvent.idempotency_key == key,
            )
        )

    @staticmethod
    def _assert_replay(event: AfterSalesCaseEvent, event_type: str, payload: dict[str, Any]) -> None:
        if event.event_type != event_type or _canonical(event.payload) != _canonical(payload):
            raise AfterSalesDomainError(
                "CASE_IDEMPOTENCY_CONFLICT",
                "同一幂等键不能用于不同的案件变更",
            )

    @staticmethod
    def _case_identity(case: AfterSalesCase) -> str:
        return _canonical(
            {
                "case_type": case.case_type,
                "order_id": case.order_id,
                "order_item_id": case.order_item_id,
                "requested_resolution": case.requested_resolution,
                "reason_code": case.reason_code,
                "reason_text": case.reason_text,
                "origin_session_id": case.origin_session_id,
                "context": case.context or {},
            }
        )

    @staticmethod
    def _command_identity(
        command: CaseCreateCommand,
        *,
        resolved_order_item_id: str | None = None,
    ) -> str:
        return _canonical(
            {
                "case_type": command.case_type.value,
                "order_id": command.order_id,
                "order_item_id": resolved_order_item_id or command.order_item_id,
                "requested_resolution": command.requested_resolution.value,
                "reason_code": command.reason_code,
                "reason_text": command.reason_text,
                "origin_session_id": command.origin_session_id,
                "context": command.context,
            }
        )

    @staticmethod
    def _decision_event_payload(decision: PolicyDecision) -> dict[str, Any]:
        return {
            "decision_id": decision.decision_id,
            "policy_version_id": decision.policy_version_id,
            "case_revision": decision.case_revision,
            "input_hash": decision.input_hash,
            "outcome": decision.outcome,
            "reason_codes": decision.reason_codes,
            "allowed_actions": decision.allowed_actions,
            "supersedes_decision_id": decision.supersedes_decision_id,
        }

    @staticmethod
    def _decision_dict(decision: PolicyDecision) -> dict[str, Any]:
        return {
            **AfterSalesCaseService._decision_event_payload(decision),
            "facts": decision.facts,
            "evidence_refs": decision.evidence_refs,
            "evaluator_version": decision.evaluator_version,
            "decided_at": decision.decided_at,
        }
