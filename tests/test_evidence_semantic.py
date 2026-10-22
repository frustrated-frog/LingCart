"""语义核验契约与真实持久化链路；模型替身只验证控制逻辑。"""
import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select

from src import llm, tools
from src.after_sales.semantic import assess, validate_response
from src.backend.db import AfterSalesCase, AfterSalesEvidence, AfterSalesSemanticAssessment, PendingAction, Refund
from src.durable import run_conversation
from src.runtime import RunRuntime, BudgetLimits, runtime_scope
from tests.test_evidence_runtime import database, start_case, DETAIL, REVOKE
from tests.evidence_fixtures import SemanticModel, response_for


def inputs():
    return {'latest_request': '不是商品坏了，是包装破了，商品我还没有检查。',
            'requested_withdrawal': None,
            'requirements': [{'requirement_id': 'p:DAMAGE', 'evidence_type': 'DAMAGE', 'description': '说明商品损坏情况'}],
            'evidence': [{'evidence_id': 'new', 'evidence_type': 'DAMAGE',
                          'statement': '包装破了', 'source_text': '不是商品坏了，是包装破了，商品我还没有检查。'}]}


@pytest.mark.parametrize('mode', ['foreign', 'quote', 'missing', 'duplicate', 'withdrawal', 'unasked', 'wrong_type'])
def test_response_requires_grounded_complete_protocol(mode):
    data = inputs()
    response = response_for(data)
    row = response['requirements'][0]
    if mode == 'foreign':
        row['citations'][0]['evidence_id'] = 'another-case'
    elif mode == 'quote':
        row['citations'][0]['quote'] = '商品确实损坏了'
    elif mode == 'missing':
        response['requirements'] = []
    elif mode == 'duplicate':
        response['requirements'] *= 2
    elif mode == 'withdrawal':
        data['requested_withdrawal'] = 'new'
        response.update(withdrawal_confirmed=True, withdrawal_quote='我授权删除')
    elif mode == 'unasked':
        response.update(withdrawal_confirmed=True, withdrawal_quote=data['latest_request'])
    else:
        data['evidence'][0]['evidence_type'] = 'OTHER'
    with pytest.raises(ValueError):
        validate_response(json.dumps(response), data)


@pytest.mark.parametrize('status', ['SUFFICIENT', 'INCOMPLETE', 'IRRELEVANT', 'CONFLICT', 'UNKNOWN'])
def test_valid_statuses_preserve_quotes_and_full_source_context(status):
    runtime = RunRuntime('semantic-test', BudgetLimits())
    model = SemanticModel(status)
    with runtime_scope(runtime):
        result = assess(inputs(), model=model, runtime=runtime)
    assert result['requirements'][0]['status'] == status
    assert model.calls == 1
    event = next(e for e in runtime.events if e['type'] == 'model_call')
    assert event['purpose'] == 'evidence_verification'
    assert inputs()['latest_request'] in next(iter(runtime.models.values()))['input']


def test_protocol_failure_and_timeout_abstain_but_budget_is_not_swallowed():
    from src.errors import AgentError
    runtime = RunRuntime('semantic-test', BudgetLimits(read_attempts=1))
    class Invalid:
        def invoke(self, _):
            return AIMessage(content='{"requirements": []}')
    with runtime_scope(runtime):
        assert assess(inputs(), model=Invalid(), runtime=runtime)['error_code'] == 'SEMANTIC_PROTOCOL_INVALID'
    class TimedOut:
        def model_call(self, *_a, **_kw):
            raise AgentError('MODEL_TIMEOUT', 'timeout', retryable=False)
        limits = BudgetLimits()
    assert assess(inputs(), model=Invalid(), runtime=TimedOut())['error_code'] == 'SEMANTIC_MODEL_UNAVAILABLE'
    runtime = RunRuntime('budget', BudgetLimits(max_model_calls=1))
    runtime.counters['model_calls'] = 1
    with runtime_scope(runtime), pytest.raises(AgentError, match='预算'):
        assess(inputs(), model=Invalid(), runtime=runtime)


def test_oversized_input_is_not_silently_truncated_or_sent_to_model():
    data = inputs()
    data['latest_request'] *= 2000
    model = SemanticModel()
    runtime = RunRuntime('oversized', BudgetLimits())
    result = assess(data, model=model, runtime=runtime)
    assert result['error_code'] == 'SEMANTIC_CONTEXT_LIMIT'
    assert model.calls == 0


@pytest.mark.parametrize('status', ['INCOMPLETE', 'IRRELEVANT', 'CONFLICT', 'UNKNOWN'])
def test_non_sufficient_material_is_persisted_but_never_approved(database, monkeypatch, status):
    case_id = start_case(monkeypatch)
    monkeypatch.setattr(llm, 'get_llm', lambda **_: SemanticModel(status))
    result = run_conversation(DETAIL, 'evidence-session', request_id='semantic')
    assert not result.get('pending_action'), result
    with tools._session() as session:
        from src.after_sales.service import AfterSalesCaseService
        case = AfterSalesCaseService(session).get_case(case_id)
        assert len(case.evidence) == 2
        assert case.status == 'AWAITING_EVIDENCE'
        assert case.latest_decision['facts']['evidence_requirements'][0]['semantic_assessment']['status'] == status
        assert case.plans[-1]['next_actions'][0]['question'] == '请确认商品本身是否损坏？'
        assert session.scalar(select(Refund).where(Refund.order_id == 1001)).status == 'none'
        latest = session.scalar(select(AfterSalesSemanticAssessment).order_by(AfterSalesSemanticAssessment.created_at.desc()))
        assert latest.payload['input']['latest_request'] == DETAIL
    assert '请确认商品本身是否损坏' in result['reply']


def test_conflict_closes_old_approval_and_explicit_withdrawal_resolves(database, monkeypatch):
    case_id = start_case(monkeypatch)
    first = run_conversation(DETAIL, 'evidence-session', request_id='sufficient')
    action_id = first['pending_action']['action_id']
    monkeypatch.setattr(llm, 'get_llm', lambda **_: SemanticModel('CONFLICT'))
    conflict = run_conversation(DETAIL, 'evidence-session', request_id='conflict')
    assert conflict.get('pending_action') is None
    with tools._session() as session:
        assert session.get(PendingAction, action_id).status == 'CLOSED'
        assert session.get(AfterSalesCase, case_id).status == 'AWAITING_EVIDENCE'
        assert len(session.scalars(select(AfterSalesEvidence).where(AfterSalesEvidence.status == 'ACTIVE')).all()) == 3
    monkeypatch.setattr(llm, 'get_llm', lambda **_: SemanticModel('SUFFICIENT'))
    corrected = run_conversation(REVOKE, 'evidence-session', request_id='correction')
    assert corrected['pending_action']['action_id'] != action_id
    with tools._session() as session:
        assert len(session.scalars(select(AfterSalesSemanticAssessment)).all()) == 4
        assert session.scalar(select(Refund).where(Refund.order_id == 1001)).status == 'none'


def test_unconfirmed_withdrawal_is_rejected_without_mutating_evidence(database, monkeypatch):
    case_id = start_case(monkeypatch)
    run_conversation(DETAIL, 'evidence-session', request_id='first')
    with tools._session() as session:
        revision = session.get(AfterSalesCase, case_id).revision
    monkeypatch.setattr(llm, 'get_llm', lambda **_: SemanticModel(withdrawal=False))
    result = run_conversation(REVOKE, 'evidence-session', request_id='ambiguous')
    assert any(e.get('code') == 'EVIDENCE_WITHDRAWAL_UNCONFIRMED' for e in result['events'])
    with tools._session() as session:
        assert session.get(AfterSalesCase, case_id).revision == revision
        assert all(e.status == 'ACTIVE' for e in session.scalars(select(AfterSalesEvidence)))


def test_policy_change_during_verification_rejects_atomic_write(database, monkeypatch):
    case_id = start_case(monkeypatch)
    class PolicyChanging(SemanticModel):
        def invoke(self, prompt):
            from src.backend.db import PolicyVersion
            from src.after_sales.service import _checksum
            with tools._session() as session:
                policy = session.scalar(select(PolicyVersion).where(PolicyVersion.status == 'ACTIVE'))
                policy.rules = {**policy.rules, 'return_window_days': 999}
                policy.checksum = _checksum(policy.rules)
                session.commit()
            return super().invoke(prompt)
    monkeypatch.setattr(llm, 'get_llm', lambda **_: PolicyChanging())
    result = run_conversation(DETAIL, 'evidence-session', request_id='policy-change')
    assert any(e.get('code') == 'SEMANTIC_ASSESSMENT_STALE' for e in result['events']), result
    with tools._session() as session:
        assert len(session.scalars(select(AfterSalesEvidence)).all()) == 1
        assert len(session.scalars(select(AfterSalesSemanticAssessment)).all()) == 1


def test_original_source_context_change_invalidates_prior_sufficiency(database, monkeypatch):
    from src.backend.db import AgentRun
    from src.after_sales.queries import get_case_detail
    case_id = start_case(monkeypatch)
    result = run_conversation(DETAIL, 'evidence-session', request_id='sufficient')
    with tools._session() as session:
        run = session.get(AgentRun, result['run_id'])
        run.request = '以下是错误的描述：' + run.request
        session.commit()
        detail = get_case_detail(session, case_id)
        assert detail.case.latest_decision['valid'] is False
    rejected = run_conversation('确认', 'evidence-session', request_id='confirm-stale')
    assert rejected['status'] != 'SUCCEEDED'
    with tools._session() as session:
        assert session.scalar(select(Refund).where(Refund.order_id == 1001)).status == 'none'


def test_model_unavailable_closes_old_approval_without_reusing_old_semantics(database, monkeypatch):
    case_id = start_case(monkeypatch)
    first = run_conversation(DETAIL, 'evidence-session', request_id='sufficient')
    class Invalid(SemanticModel):
        def invoke(self, _):
            return AIMessage(content='invalid')
    monkeypatch.setattr(llm, 'get_llm', lambda **_: Invalid())
    result = run_conversation(DETAIL, 'evidence-session', request_id='unavailable')
    assert result.get('pending_action') is None
    with tools._session() as session:
        assert session.get(PendingAction, first['pending_action']['action_id']).status == 'CLOSED'
        assert session.get(AfterSalesCase, case_id).status == 'AWAITING_EVIDENCE'


def test_semantic_commit_failure_rolls_back_material_assessment_and_approval(database, monkeypatch):
    from src.after_sales.service import AfterSalesCaseService, AfterSalesDomainError
    case_id = start_case(monkeypatch)
    first = run_conversation(DETAIL, 'evidence-session', request_id='sufficient')
    with tools._session() as session:
        revision = session.get(AfterSalesCase, case_id).revision
    def fail(*_a, **_kw):
        raise AfterSalesDomainError('INJECTED_POLICY_FAILURE', '测试回滚')
    monkeypatch.setattr(AfterSalesCaseService, 'record_policy_decision', fail)
    run_conversation(DETAIL, 'evidence-session', request_id='rollback')
    with tools._session() as session:
        assert session.get(AfterSalesCase, case_id).revision == revision
        assert len(session.scalars(select(AfterSalesEvidence)).all()) == 2
        assert len(session.scalars(select(AfterSalesSemanticAssessment)).all()) == 2
        assert session.get(PendingAction, first['pending_action']['action_id']).status == 'OPEN'


def test_reassessment_with_same_timestamp_uses_case_revision_not_random_id(database, monkeypatch):
    from src.after_sales.policy import AfterSalesPolicyEngine
    case_id = start_case(monkeypatch)
    run_conversation(DETAIL, 'evidence-session', request_id='sufficient')
    monkeypatch.setattr(llm, 'get_llm', lambda **_: SemanticModel('UNKNOWN'))
    result = run_conversation('请重新核验材料', 'evidence-session', request_id='reassess')
    assert result.get('pending_action') is None
    with tools._session() as session:
        rows = session.scalars(select(AfterSalesSemanticAssessment).where(
            AfterSalesSemanticAssessment.case_id == case_id).order_by(AfterSalesSemanticAssessment.case_revision)).all()
        assert rows[-1].basis_hash == rows[-2].basis_hash
        rows[-1].created_at = rows[-2].created_at
        rows[-2].assessment_id = 'z-older'
        rows[-1].assessment_id = 'a-newer'
        session.flush()
        evaluation = AfterSalesPolicyEngine(session).evaluate(case_id)
        assert evaluation.outcome == 'NEEDS_EVIDENCE'
        assert evaluation.facts['evidence_requirements'][0]['semantic_assessment']['status'] == 'UNKNOWN'
