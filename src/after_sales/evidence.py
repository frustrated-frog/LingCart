"""证据来源核验与政策充分性：来源可信和陈述真实是不同的判断。"""
from __future__ import annotations

from datetime import datetime
from sqlalchemy import select

from src.after_sales.contracts import EvidenceRequirement
from src.after_sales.service import _checksum
from src.backend.db import AfterSalesEvidence, AfterSalesEvidenceValidation, AgentRun, Order

VALIDATOR_VERSION = "evidence-provenance-v1"


def evidence_content(evidence):
    return {
        "evidence_type": evidence.evidence_type,
        "source_type": evidence.source_type,
        "source_ref": evidence.source_ref,
        "payload": evidence.payload,
        "observed_at": evidence.observed_at.isoformat(),
    }


def verify_source(session, case, evidence) -> tuple[str, str]:
    if _checksum(evidence_content(evidence)) != evidence.checksum:
        return "REJECTED", "EVIDENCE_INTEGRITY_ERROR"
    if evidence.source_type == "CUSTOMER_STATEMENT":
        prefix, separator, run_id = evidence.source_ref.partition(":")
        run = session.get(AgentRun, run_id.removesuffix(":request")) if prefix == "agent_runs" and separator else None
        statement = evidence.payload.get("statement")
        if (run is not None and case.origin_session_id is not None
                and run.session_id == case.origin_session_id and isinstance(statement, str)
                and statement.strip() and statement in run.request):
            return "VERIFIED", "CUSTOMER_SOURCE_MATCHED"
        return "UNVERIFIED", "CUSTOMER_SOURCE_NOT_VERIFIABLE"
    if evidence.source_type == "ORDER_RECORD":
        order = session.get(Order, case.order_id)
        expected = {"order_id": case.order_id, "status": order.status if order else None}
        if order and evidence.source_ref == f"orders:{case.order_id}" and evidence.payload == expected:
            return "VERIFIED", "AUTHORITATIVE_RECORD_MATCHED"
        return "REJECTED", "AUTHORITATIVE_RECORD_CHANGED"
    return "UNVERIFIED", "SOURCE_VALIDATOR_UNAVAILABLE"


def record_validation(session, case, evidence, now):
    status, reason = verify_source(session, case, evidence)
    session.add(AfterSalesEvidenceValidation(
        evidence_id=evidence.evidence_id, checksum=evidence.checksum, status=status,
        reason_code=reason, validator_version=VALIDATOR_VERSION,
        case_revision=case.revision + 1, validated_at=now,
    ))


def evaluate_requirements(session, case, policy, rules, as_of: datetime) -> list[dict]:
    evidence = session.scalars(select(AfterSalesEvidence).where(
        AfterSalesEvidence.case_id == case.case_id).order_by(AfterSalesEvidence.evidence_id)).all()
    from src.after_sales.semantic import current_assessment
    assessment = current_assessment(session, case, policy, evidence, as_of)
    semantic_rows = {r['requirement_id']: r for r in assessment.payload['result']['requirements']} if assessment else {}
    requirements = []
    for evidence_type in rules["required_evidence_by_reason"].get(case.reason_code, []):
        rule = rules.get("evidence_acceptance", {}).get(evidence_type, {})
        sources = rule.get("accepted_sources", [])
        condition = rule.get("condition", "UNSUPPORTED")
        age = rule.get("max_age_days")
        semantic = semantic_rows.get(f"{policy.policy_version_id}:{evidence_type}")
        accepted, rejected = [], {}
        for item in evidence:
            if item.evidence_type != evidence_type:
                continue
            validation = session.get(AfterSalesEvidenceValidation, item.evidence_id)
            source_status, reason = verify_source(session, case, item)
            if item.status != "ACTIVE":
                reason = item.status
            elif item.observed_at > as_of:
                reason = "OBSERVATION_IN_FUTURE"
            elif age is not None and (as_of - item.observed_at).total_seconds() > age * 86400:
                reason = "EXPIRED"
            elif not validation or validation.status != "VERIFIED" or validation.checksum != item.checksum:
                reason = "SOURCE_NOT_VERIFIED"
            elif source_status != "VERIFIED":
                pass
            elif item.source_type not in sources:
                reason = "SOURCE_NOT_ACCEPTED"
            elif condition == "CUSTOMER_ATTESTATION" and item.source_type == "CUSTOMER_STATEMENT":
                cited = {c['evidence_id'] for c in (semantic or {}).get('citations', [])}
                if semantic and semantic['status'] == 'SUFFICIENT' and item.evidence_id in cited:
                    accepted.append(item.evidence_id)
                    continue
                reason = f"SEMANTIC_{semantic['status']}" if semantic else "SEMANTIC_NOT_ASSESSED"
            elif condition == "AUTHORITATIVE_ORDER" and item.source_type == "ORDER_RECORD":
                accepted.append(item.evidence_id)
                continue
            else:
                reason = "VALIDATION_CONDITION_UNSUPPORTED"
            rejected[item.evidence_id] = reason
        requirements.append(EvidenceRequirement(
            requirement_id=f"{policy.policy_version_id}:{evidence_type}",
            policy_version_id=policy.policy_version_id, policy_checksum=policy.checksum,
            target_case_id=case.case_id, evidence_type=evidence_type,
            description=rule.get("description", f"需要可核验的 {evidence_type} 材料"),
            accepted_sources=tuple(sources), validation_condition=condition, max_age_days=age,
            status="SATISFIED" if accepted else "UNSATISFIED",
            evidence_refs=tuple(accepted), rejected_evidence=rejected,
            semantic_assessment=({**semantic, "assessment_id": assessment.assessment_id,
                                  "verifier_version": assessment.payload['verifier_version'],
                                  "error_code": assessment.payload['result'].get('error_code')}
                                 if semantic else None),
        ).model_dump(mode="json"))
    return requirements
