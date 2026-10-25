"""Durable evidence-readiness claims kept separate from evidence and authorization."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable, Mapping

from sqlalchemy import select

from src.after_sales.contracts import EvidenceReadinessClaim, EvidenceReadinessState
from src.backend.db import AfterSalesCaseEvent


READINESS_PROJECTOR_VERSION = "evidence-readiness-v1"
READINESS_CATALOG_VERSION = "after-sales-evidence-readiness/2026-01"
READINESS_CATALOG_SOURCE = "LINGCART_AFTER_SALES_POLICY"


class EvidenceReadinessValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def claim_fingerprint(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical(dict(payload)).encode("utf-8")).hexdigest()


def readiness_type_descriptions() -> dict[str, str]:
    """Expose only customer-controlled evidence types from the versioned demo policy catalog."""

    from src.backend.seed import DEFAULT_AFTER_SALES_POLICY_RULES

    return {
        evidence_type: str(rule["description"])
        for evidence_type, rule in DEFAULT_AFTER_SALES_POLICY_RULES["evidence_acceptance"].items()
        if rule.get("condition") == "CUSTOMER_ATTESTATION"
        and "CUSTOMER_STATEMENT" in rule.get("accepted_sources", [])
    }


def readiness_prompt_catalog() -> dict[str, object]:
    values = readiness_type_descriptions()
    identity = {
        "version": READINESS_CATALOG_VERSION,
        "source": READINESS_CATALOG_SOURCE,
        "values": values,
    }
    return {**identity, "fingerprint": claim_fingerprint(identity)}


def validate_readiness_claims(
    claims: Iterable[EvidenceReadinessClaim | Mapping[str, object]],
    *,
    request: str,
    reason_code: str | None,
    rules: Mapping[str, object],
) -> tuple[EvidenceReadinessClaim, ...]:
    """Validate grounding and policy applicability without treating a claim as evidence."""

    normalized = tuple(
        claim if isinstance(claim, EvidenceReadinessClaim)
        else EvidenceReadinessClaim.model_validate(claim, strict=True)
        for claim in claims
    )
    identities = [(claim.evidence_type, claim.disposition, claim.source_quote) for claim in normalized]
    if len(identities) != len(set(identities)):
        raise EvidenceReadinessValidationError(
            "EVIDENCE_READINESS_DUPLICATE",
            "同一证据就绪度声明不能重复记录",
        )
    required_by_reason = rules.get("required_evidence_by_reason")
    acceptance = rules.get("evidence_acceptance")
    if not isinstance(required_by_reason, Mapping) or not isinstance(acceptance, Mapping):
        raise EvidenceReadinessValidationError(
            "EVIDENCE_READINESS_POLICY_INVALID",
            "适用政策没有可核验的证据要求",
        )
    required = set(required_by_reason.get(reason_code, [])) if reason_code else set()
    for claim in normalized:
        if claim.source_quote not in request:
            raise EvidenceReadinessValidationError(
                "UNGROUNDED_EVIDENCE_READINESS",
                "证据就绪度必须逐字引用本轮用户原文",
            )
        if claim.evidence_type not in required:
            raise EvidenceReadinessValidationError(
                "EVIDENCE_READINESS_NOT_APPLICABLE",
                "证据就绪度声明不属于当前原因的政策义务",
            )
        rule = acceptance.get(claim.evidence_type)
        if (
            not isinstance(rule, Mapping)
            or rule.get("condition") != "CUSTOMER_ATTESTATION"
            or "CUSTOMER_STATEMENT" not in rule.get("accepted_sources", [])
        ):
            raise EvidenceReadinessValidationError(
                "EVIDENCE_READINESS_NOT_CUSTOMER_CONTROLLED",
                "只有用户可补充的声明类证据才能记录就绪度",
            )
    return normalized


def readiness_event_payload(
    claim: EvidenceReadinessClaim,
    *,
    source_ref: str,
    policy_version_id: str,
    policy_checksum: str,
) -> dict[str, object]:
    basis = {
        "schema_version": 1,
        "projector_version": READINESS_PROJECTOR_VERSION,
        "requirement_id": f"{policy_version_id}:{claim.evidence_type}",
        "evidence_type": claim.evidence_type,
        "disposition": claim.disposition,
        "source_quote": claim.source_quote,
        "source_ref": source_ref,
        "policy_version_id": policy_version_id,
        "policy_checksum": policy_checksum,
    }
    return {**basis, "claim_id": claim_fingerprint(basis)}


def project_evidence_readiness(
    session,
    *,
    case_id: str,
    policy_version_id: str,
    policy_checksum: str,
    requirements: Iterable[Mapping[str, object]],
) -> dict[str, EvidenceReadinessState]:
    """Project current deferrals; later evidence for the same type consumes the deferral."""

    requirement_by_type = {
        str(row.get("evidence_type")): str(row.get("requirement_id"))
        for row in requirements
        if row.get("evidence_type") and row.get("requirement_id")
    }
    current: dict[str, EvidenceReadinessState] = {}
    events = session.scalars(
        select(AfterSalesCaseEvent)
        .where(AfterSalesCaseEvent.case_id == case_id)
        .order_by(AfterSalesCaseEvent.sequence)
    ).all()
    for event in events:
        payload = event.payload if isinstance(event.payload, dict) else {}
        evidence_type = str(payload.get("evidence_type") or "")
        if event.event_type == "EVIDENCE_ADDED":
            current.pop(evidence_type, None)
            continue
        if event.event_type != "EVIDENCE_READINESS_RECORDED":
            continue
        if (
            payload.get("policy_version_id") != policy_version_id
            or payload.get("policy_checksum") != policy_checksum
            or evidence_type not in requirement_by_type
        ):
            continue
        basis = {key: payload.get(key) for key in (
            "schema_version",
            "projector_version",
            "requirement_id",
            "evidence_type",
            "disposition",
            "source_quote",
            "source_ref",
            "policy_version_id",
            "policy_checksum",
        )}
        if (
            payload.get("schema_version") != 1
            or payload.get("projector_version") != READINESS_PROJECTOR_VERSION
            or payload.get("requirement_id") != requirement_by_type[evidence_type]
            or payload.get("claim_id") != claim_fingerprint(basis)
        ):
            from src.after_sales.service import AfterSalesDomainError

            raise AfterSalesDomainError(
                "EVIDENCE_READINESS_CORRUPT",
                "证据就绪度事件与政策义务绑定不一致",
            )
        current[evidence_type] = EvidenceReadinessState(
            **payload,
            case_revision=event.case_revision,
            recorded_at=event.occurred_at.isoformat(),
        )
    return {
        requirement_by_type[evidence_type]: state
        for evidence_type, state in current.items()
    }
