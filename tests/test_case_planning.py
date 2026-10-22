"""案件计划只修订失效依赖，持续保留其他目标与历史证据。"""
from datetime import datetime
from sqlalchemy import select

from tests.test_after_sales import session, _command, _evidence
from tests.test_evidence_lifecycle import assess, NOW
from src.after_sales import AfterSalesCaseService, AfterSalesPolicyEngine
from src.after_sales.queries import get_case_detail
from src.backend.db import AfterSalesCasePlan, AfterSalesEvidence, PolicyDecision, PolicyVersion
from src.backend.seed import (DEFAULT_AFTER_SALES_POLICY_VERSION_ID, DEFAULT_AFTER_SALES_POLICY_ID,
                              LEGACY_AFTER_SALES_POLICY_RULES, seed, _json_checksum)


def test_two_requirements_repair_only_changed_goal_and_reuse_unaffected_goal(session):
    service = AfterSalesCaseService(session)
    case = assess(session, service.open_case(_command(reason_code='CHANGED_MIND')))
    initial = case.plans[-1]
    assert len(initial['next_actions']) == 2
    case = service.add_evidence(case.case_id, _evidence().model_copy(update={'evidence_type': 'ITEM_CONDITION'}),
                                idempotency_key='condition', expected_revision=case.revision)
    assert case.plans[-1]['valid'] is False
    case = assess(session, case)
    second = case.plans[-1]
    assert second['supersedes_plan_id'] == initial['plan_id']
    assert second['retained_goal_ids'] == ['evidence:ORIGINAL_PACKAGING']
    assert len(second['next_actions']) == 1
    case = service.add_evidence(case.case_id, _evidence().model_copy(update={'evidence_type': 'ORIGINAL_PACKAGING'}),
                                idempotency_key='packaging', expected_revision=case.revision)
    case = assess(session, case)
    third = case.plans[-1]
    assert third['retained_goal_ids'] == ['evidence:ITEM_CONDITION']
    assert third['next_actions'][0]['kind'] == 'REQUEST_CUSTOMER_CONFIRMATION'
    assert third['next_actions'][0]['requires_approval'] is True
    assert len(case.plans) == 3
    assert case.plans[0]['goals'] == initial['goals']
    repeat = assess(session, case)
    assert len(repeat.plans) == 3
    assert repeat.plans[-1]['plan_id'] == third['plan_id']


def test_revoke_repairs_dependencies_but_never_discards_old_evidence_or_decisions(session):
    service = AfterSalesCaseService(session)
    case = service.open_case(_command())
    case = service.add_evidence(case.case_id, _evidence(), idempotency_key='e', expected_revision=case.revision)
    case = assess(session, case)
    original = case.plans[-1]
    case = service.revoke_evidence(case.case_id, case.evidence[0]['evidence_id'],
                                   idempotency_key='r', expected_revision=case.revision)
    case = assess(session, case)
    assert case.plans[-1]['invalidated_decision_id'] == original['decision_id']
    assert case.plans[-1]['next_actions'][0]['kind'] == 'REQUEST_CUSTOMER_EVIDENCE'
    assert session.scalar(select(AfterSalesEvidence)).status == 'REVOKED'
    assert len(session.scalars(select(PolicyDecision)).all()) == 2
    assert len(session.scalars(select(AfterSalesCasePlan)).all()) == 2


def test_detail_marks_time_expired_decision_and_plan_stale_without_mutating_ledger(session):
    service = AfterSalesCaseService(session)
    case = service.open_case(_command())
    case = service.add_evidence(case.case_id, _evidence(), idempotency_key='e', expected_revision=case.revision)
    case = assess(session, case)
    detail = get_case_detail(session, case.case_id)
    assert detail.case.latest_decision['valid'] is False
    assert detail.case.latest_decision['stale_reason'] == 'POLICY_INPUT_CHANGED'
    assert detail.case.plans[-1]['valid'] is False
    assert detail.case.revision == case.revision
    assert len(detail.case.events) == len(case.events)


def test_seed_upgrade_preserves_legacy_policy_contents(session):
    session.delete(session.get(PolicyVersion, DEFAULT_AFTER_SALES_POLICY_VERSION_ID))
    legacy_id = f'{DEFAULT_AFTER_SALES_POLICY_ID}:2026-01'
    session.add(PolicyVersion(policy_version_id=legacy_id, policy_id=DEFAULT_AFTER_SALES_POLICY_ID,
                              version='2026-01', policy_type='AFTER_SALES', status='ACTIVE',
                              effective_from=datetime(2026, 1, 1), rules=LEGACY_AFTER_SALES_POLICY_RULES,
                              source_ref='DEMO_POLICY:standard-return-refund:2026-01',
                              checksum=_json_checksum(LEGACY_AFTER_SALES_POLICY_RULES), created_at=datetime(2026, 1, 1)))
    session.commit()
    seed(session)
    assert session.get(PolicyVersion, legacy_id).rules == LEGACY_AFTER_SALES_POLICY_RULES
    assert session.get(PolicyVersion, legacy_id).status == 'RETIRED'
    assert session.get(PolicyVersion, DEFAULT_AFTER_SALES_POLICY_VERSION_ID).status == 'ACTIVE'
