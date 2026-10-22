"""Stable contracts for after-sales cases and deterministic policy decisions."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CaseType(StrEnum):
    RETURN = "RETURN"
    REFUND = "REFUND"
    EXCHANGE = "EXCHANGE"
    REPAIR = "REPAIR"
    DELIVERY_ISSUE = "DELIVERY_ISSUE"


class RequestedResolution(StrEnum):
    REFUND = "REFUND"
    RETURN_AND_REFUND = "RETURN_AND_REFUND"
    REPLACEMENT = "REPLACEMENT"
    REPAIR = "REPAIR"
    INFORMATION = "INFORMATION"


class CaseStatus(StrEnum):
    OPEN = "OPEN"
    AWAITING_EVIDENCE = "AWAITING_EVIDENCE"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    SUBMITTED = "SUBMITTED"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    RESOLVED = "RESOLVED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self in {self.RESOLVED, self.REJECTED, self.CANCELLED}


class PolicyOutcome(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    NEEDS_EVIDENCE = "NEEDS_EVIDENCE"
    INELIGIBLE = "INELIGIBLE"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class ActorType(StrEnum):
    CUSTOMER = "CUSTOMER"
    AGENT = "AGENT"
    SYSTEM = "SYSTEM"
    HUMAN = "HUMAN"


class CaseCreateCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_key: str = Field(min_length=1, max_length=128)
    case_type: CaseType
    order_id: int = Field(gt=0)
    order_item_id: str | None = Field(default=None, min_length=1, max_length=64)
    requested_resolution: RequestedResolution
    reason_code: str | None = Field(default=None, min_length=1, max_length=40)
    reason_text: str = Field(min_length=1, max_length=2000)
    origin_session_id: str | None = Field(default=None, min_length=1, max_length=64)
    context: dict[str, Any] = Field(default_factory=dict)


class EvidenceCommand(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_type: str = Field(min_length=1, max_length=48)
    source_type: str = Field(min_length=1, max_length=32)
    source_ref: str = Field(min_length=1, max_length=200)
    payload: dict[str, Any]
    observed_at: datetime


class PolicyEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version_id: str
    policy_checksum: str
    outcome: PolicyOutcome
    reason_codes: tuple[str, ...]
    facts: dict[str, Any]
    evidence_refs: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    evaluator_version: str
    evaluated_at: datetime
    input_hash: str


class EvidenceRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: str
    policy_version_id: str
    policy_checksum: str
    target_case_id: str
    evidence_type: str
    description: str
    accepted_sources: tuple[str, ...]
    validation_condition: str
    max_age_days: int | None = None
    status: str
    evidence_refs: tuple[str, ...] = ()
    rejected_evidence: dict[str, str] = Field(default_factory=dict)


class CaseProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    request_key: str
    case_type: CaseType
    order_id: int
    order_item_id: str | None
    requested_resolution: RequestedResolution
    reason_code: str | None
    reason_text: str
    status: CaseStatus
    revision: int
    origin_session_id: str | None
    context: dict[str, Any]
    opened_at: datetime
    updated_at: datetime
    closed_at: datetime | None
    latest_decision: dict[str, Any] | None = None
    evidence: tuple[dict[str, Any], ...] = ()
    events: tuple[dict[str, Any], ...] = ()


class CaseRunReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    session_id: str
    role: str
    linked_at: datetime


class CaseRefundReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    refund_id: int
    status: str
    reason: str | None
    policy_decision_id: str
    submitted_run_id: str | None
    linked_at: datetime


class PolicyVersionProjection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_version_id: str
    policy_id: str
    version: str
    status: str
    effective_from: datetime
    effective_until: datetime | None
    rules: dict[str, Any]
    source_ref: str
    checksum: str


class CaseListItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    case_type: CaseType
    order_id: int
    order_item_id: str | None
    requested_resolution: RequestedResolution
    reason_text: str
    status: CaseStatus
    revision: int
    origin_session_id: str | None
    opened_at: datetime
    updated_at: datetime
    policy_outcome: PolicyOutcome | None
    policy_reason_codes: tuple[str, ...] = ()
    policy_version_id: str | None = None
    refund_id: int | None = None


class CaseDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case: CaseProjection
    runs: tuple[CaseRunReference, ...] = ()
    refund: CaseRefundReference | None = None
    policy: PolicyVersionProjection | None = None
