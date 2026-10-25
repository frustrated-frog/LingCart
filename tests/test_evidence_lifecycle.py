"""证据不能靠类型、伪造来源或陈旧裁决绕过政策。"""
from datetime import datetime, timedelta
import pytest
from sqlalchemy import select

from tests.test_after_sales import session, _command, _evidence
from src.after_sales import AfterSalesCaseService, AfterSalesPolicyEngine, AfterSalesDomainError
from src.after_sales.service import _checksum
from src.backend.db import (AfterSalesEvidence, AfterSalesEvidenceValidation, Conversation,
                            PendingAction, PolicyVersion)
from src.backend.seed import DEFAULT_AFTER_SALES_POLICY_VERSION_ID

NOW = datetime(2026, 6, 20, 12)


def assess(session, case):
    from tests.evidence_fixtures import record_assessment_fixture
    record_assessment_fixture(session, case)
    evaluation = AfterSalesPolicyEngine(session).evaluate(case.case_id, as_of=NOW)
    return AfterSalesCaseService(session).record_policy_decision(
        case.case_id, evaluation, idempotency_key=f"policy:{evaluation.input_hash}", expected_revision=case.revision)


@pytest.mark.parametrize('change', [
    {"source_ref": "agent_runs:invented:request"},
    {"payload": {"statement": "模型伪造的陈述"}},
    {"source_type": "CUSTOMER_UPLOAD"},
])
def test_type_match_with_unverified_source_is_insufficient(session, change):
    service = AfterSalesCaseService(session)
    case = service.open_case(_command())
    case = service.add_evidence(case.case_id, _evidence().model_copy(update=change),
                               idempotency_key="e", expected_revision=case.revision)
    result = assess(session, case)
    assert result.latest_decision["outcome"] == "NEEDS_EVIDENCE"
    requirement = result.latest_decision["facts"]["evidence_requirements"][0]
    assert requirement["status"] == "UNSATISFIED"
    assert requirement["rejected_evidence"]
    assert case.evidence[0]["validation"]["status"] == "UNVERIFIED"


def test_revoke_invalidates_decision_and_approval_atomically_and_replays(session):
    service = AfterSalesCaseService(session)
    case = service.open_case(_command())
    case = service.add_evidence(case.case_id, _evidence(), idempotency_key="e", expected_revision=case.revision)
    case = assess(session, case)
    pending = {"case_id": case.case_id, "action_id": "approval"}
    session.get(Conversation, "conversation-1").state = {"pending_action": pending}
    session.add(PendingAction(action_id="approval", session_id="conversation-1", source_run_id="evidence-run",
                              status="OPEN", payload=pending))
    session.commit()
    changed = service.revoke_evidence(case.case_id, case.evidence[0]["evidence_id"],
                                     idempotency_key="revoke", expected_revision=case.revision)
    assert changed.latest_decision["valid"] is False
    assert session.get(PendingAction, "approval").status == "CLOSED"
    assert session.get(Conversation, "conversation-1").state["pending_action"] is None
    session.rollback()
    assert session.get(PendingAction, "approval").status == "OPEN"
    changed = service.revoke_evidence(case.case_id, case.evidence[0]["evidence_id"],
                                     idempotency_key="revoke", expected_revision=case.revision)
    replay = service.revoke_evidence(case.case_id, case.evidence[0]["evidence_id"],
                                    idempotency_key="revoke", expected_revision=case.revision)
    assert replay.revision == changed.revision
    result = assess(session, changed)
    assert result.latest_decision["outcome"] == "NEEDS_EVIDENCE"
    assert result.latest_decision["supersedes_decision_id"] == case.latest_decision["decision_id"]


@pytest.mark.parametrize('mode', ['expired', 'tampered', 'legacy', 'wrong_session'])
def test_evidence_lifetime_integrity_and_legacy_migration_fail_closed(session, mode):
    service = AfterSalesCaseService(session)
    case = service.open_case(_command())
    command = _evidence()
    if mode == 'expired':
        command = command.model_copy(update={"observed_at": NOW - timedelta(days=31)})
    case = service.add_evidence(case.case_id, command, idempotency_key="e", expected_revision=case.revision)
    item = session.get(AfterSalesEvidence, case.evidence[0]["evidence_id"])
    if mode == 'tampered':
        item.payload = {"statement": "伪造内容"}
    if mode == 'legacy':
        session.delete(session.get(AfterSalesEvidenceValidation, item.evidence_id))
    if mode == 'wrong_session':
        from src.backend.db import AgentRun
        session.get(AgentRun, "evidence-run").session_id = "another-session"
    session.flush()
    result = assess(session, case)
    assert result.latest_decision['outcome'] == 'NEEDS_EVIDENCE'


def test_policy_must_explicitly_accept_verified_source(session):
    service = AfterSalesCaseService(session)
    case = service.open_case(_command())
    case = service.add_evidence(case.case_id, _evidence(), idempotency_key='e', expected_revision=case.revision)
    policy = session.get(PolicyVersion, DEFAULT_AFTER_SALES_POLICY_VERSION_ID)
    policy.rules = {k: v for k, v in policy.rules.items() if k != 'evidence_acceptance'}
    policy.checksum = _checksum(policy.rules)
    session.flush()
    assert assess(session, case).latest_decision['outcome'] == 'NEEDS_EVIDENCE'


def test_revoke_rejects_foreign_evidence_and_stale_revision(session):
    service = AfterSalesCaseService(session)
    first = service.open_case(_command())
    first = service.add_evidence(first.case_id, _evidence(), idempotency_key='e', expected_revision=first.revision)
    other = service.open_case(_command(request_key='other'))
    with pytest.raises(AfterSalesDomainError, match='证据不属于'):
        service.revoke_evidence(other.case_id, first.evidence[0]['evidence_id'], idempotency_key='r', expected_revision=1)
    with pytest.raises(AfterSalesDomainError) as exc:
        service.revoke_evidence(first.case_id, first.evidence[0]['evidence_id'], idempotency_key='r', expected_revision=1)
    assert exc.value.code == 'CASE_REVISION_CONFLICT'


def test_stale_evaluation_cannot_be_written_after_evidence_changes(session):
    service = AfterSalesCaseService(session)
    case = service.open_case(_command())
    old = AfterSalesPolicyEngine(session).evaluate(case.case_id, as_of=NOW)
    case = service.add_evidence(case.case_id, _evidence(), idempotency_key='e', expected_revision=case.revision)
    with pytest.raises(AfterSalesDomainError) as exc:
        service.record_policy_decision(case.case_id, old, idempotency_key='old', expected_revision=case.revision)
    assert exc.value.code == 'POLICY_EVALUATION_STALE'


def test_authoritative_evidence_is_rechecked_when_business_record_changes(session):
    from src.after_sales import EvidenceCommand
    from src.backend.db import Order
    service = AfterSalesCaseService(session)
    policy = session.get(PolicyVersion, DEFAULT_AFTER_SALES_POLICY_VERSION_ID)
    policy.rules = {**policy.rules,
                    'required_evidence_by_reason': {'DAMAGED': ['ORDER_FACT']},
                    'evidence_acceptance': {'ORDER_FACT': {'accepted_sources': ['ORDER_RECORD'],
                        'condition': 'AUTHORITATIVE_ORDER', 'description': '需要权威订单状态'}}}
    policy.checksum = _checksum(policy.rules)
    session.flush()
    case = service.open_case(_command())
    case = service.add_evidence(case.case_id, EvidenceCommand(
        evidence_type='ORDER_FACT', source_type='ORDER_RECORD', source_ref='orders:1001',
        payload={'order_id': 1001, 'status': 'delivered'}, observed_at=NOW),
        idempotency_key='order-fact', expected_revision=case.revision)
    result = AfterSalesPolicyEngine(session).evaluate(case.case_id, as_of=NOW)
    assert result.facts['evidence_requirements'][0]['status'] == 'SATISFIED'
    session.get(Order, 1001).status = 'shipped'
    session.flush()
    changed = AfterSalesPolicyEngine(session).evaluate(case.case_id, as_of=NOW)
    assert changed.facts['evidence_requirements'][0]['status'] == 'UNSATISFIED'
    assert 'AUTHORITATIVE_RECORD_CHANGED' in changed.facts['evidence_requirements'][0]['rejected_evidence'].values()
