"""案件计划只修订失效依赖，持续保留其他目标与历史证据。"""
from datetime import datetime
from sqlalchemy import select

from tests.test_after_sales import session, _command, _evidence
from tests.test_evidence_lifecycle import assess, NOW
from src.after_sales import AfterSalesCaseService, AfterSalesPolicyEngine, EvidenceReadinessClaim
from src.after_sales.queries import get_case_detail
from src.backend.db import AgentRun, AfterSalesCasePlan, AfterSalesEvidence, PolicyDecision, PolicyVersion
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


def test_deferred_readiness_is_a_temporal_plan_state_consumed_by_later_evidence(session):
    service = AfterSalesCaseService(session)
    policy = session.get(PolicyVersion, DEFAULT_AFTER_SALES_POLICY_VERSION_ID)
    case = service.open_case(_command(reason_code='CHANGED_MIND', reason_text='商品没有质量问题但不再需要'))
    session.get(AgentRun, 'readiness-run').request = '原包装情况稍后才能确认'
    case = service.record_evidence_readiness(
        case.case_id,
        EvidenceReadinessClaim(
            evidence_type='ORIGINAL_PACKAGING',
            disposition='DEFERRED',
            source_quote='原包装情况稍后才能确认',
        ),
        policy_version_id=policy.policy_version_id,
        policy_checksum=policy.checksum,
        source_ref='agent_runs:readiness-run:request',
        idempotency_key='readiness:packaging',
        expected_revision=case.revision,
    )
    evaluation = AfterSalesPolicyEngine(session).evaluate(case.case_id, as_of=NOW)
    case = service.record_policy_decision(
        case.case_id,
        evaluation,
        idempotency_key=f'policy:{evaluation.input_hash}',
        expected_revision=case.revision,
    )

    initial = case.plans[-1]
    deferred = next(action for action in initial['next_actions']
                    if action['requirement_id'].endswith(':ORIGINAL_PACKAGING'))
    goal = next(goal for goal in initial['goals']
                if goal.get('requirement_id', '').endswith(':ORIGINAL_PACKAGING'))
    assert deferred['kind'] == 'WAIT_FOR_CUSTOMER_EVIDENCE'
    assert goal['readiness']['disposition'] == 'DEFERRED'

    case = service.add_evidence(
        case.case_id,
        _evidence().model_copy(update={'evidence_type': 'ORIGINAL_PACKAGING'}),
        idempotency_key='packaging-attempt',
        expected_revision=case.revision,
    )
    reevaluation = AfterSalesPolicyEngine(session).evaluate(case.case_id, as_of=NOW)
    case = service.record_policy_decision(
        case.case_id,
        reevaluation,
        idempotency_key=f'policy:{reevaluation.input_hash}',
        expected_revision=case.revision,
    )

    revised = case.plans[-1]
    packaging = next(action for action in revised['next_actions']
                     if action['requirement_id'].endswith(':ORIGINAL_PACKAGING'))
    revised_goal = next(goal for goal in revised['goals']
                        if goal.get('requirement_id', '').endswith(':ORIGINAL_PACKAGING'))
    assert packaging['kind'] == 'REQUEST_CUSTOMER_EVIDENCE'
    assert 'readiness' not in revised_goal


def test_late_readiness_claim_creates_a_new_decision_and_plan_identity(session):
    service = AfterSalesCaseService(session)
    policy = session.get(PolicyVersion, DEFAULT_AFTER_SALES_POLICY_VERSION_ID)
    case = service.open_case(_command(reason_code='CHANGED_MIND', reason_text='商品没有质量问题但不再需要'))
    first_evaluation = AfterSalesPolicyEngine(session).evaluate(case.case_id, as_of=NOW)
    case = service.record_policy_decision(
        case.case_id,
        first_evaluation,
        idempotency_key=f'policy:{first_evaluation.input_hash}',
        expected_revision=case.revision,
    )
    first_plan_id = case.plans[-1]['plan_id']
    session.get(AgentRun, 'readiness-run').request = '原包装情况稍后才能确认'

    case = service.record_evidence_readiness(
        case.case_id,
        EvidenceReadinessClaim(
            evidence_type='ORIGINAL_PACKAGING',
            disposition='DEFERRED',
            source_quote='原包装情况稍后才能确认',
        ),
        policy_version_id=policy.policy_version_id,
        policy_checksum=policy.checksum,
        source_ref='agent_runs:readiness-run:request',
        idempotency_key='readiness:late-packaging',
        expected_revision=case.revision,
    )
    assert case.latest_decision['valid'] is False
    second_evaluation = AfterSalesPolicyEngine(session).evaluate(case.case_id, as_of=NOW)
    assert second_evaluation.input_hash != first_evaluation.input_hash
    case = service.record_policy_decision(
        case.case_id,
        second_evaluation,
        idempotency_key=f'policy:{second_evaluation.input_hash}',
        expected_revision=case.revision,
    )

    assert case.plans[-1]['plan_id'] != first_plan_id
    assert case.plans[-1]['supersedes_plan_id'] == first_plan_id
    packaging = next(action for action in case.plans[-1]['next_actions']
                     if action['requirement_id'].endswith(':ORIGINAL_PACKAGING'))
    assert packaging['kind'] == 'WAIT_FOR_CUSTOMER_EVIDENCE'


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
