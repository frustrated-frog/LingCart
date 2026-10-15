"""Lossless, bounded evidence context and run-local paging."""
from copy import deepcopy
import json

import pytest

from src.context_budget import ContextBudgetError, _text_length
from src.context_engine import ContextEngine
from src.errors import AgentError
from src.task_plan import TaskPlan, TaskStep


def fixture():
    state = dict(request='不要主动降噪，预算500', goal={'hard_constraints': ['完整保留否定范围']},
                 goal_revision=1, next_steps=['查证佩戴体验'], feedback=None,
                 objective_ledger=[dict(objective_id='shopping', status='PENDING', evidence_refs=[])],
                 working_memory={'summary': '待查证', 'evidence_refs': ['step-1'], 'open_questions': ['是否舒适？']})
    products = [dict(product_id=f'P{i:04}', name=f'商品{i}', price=100 + i, stock=2,
                     specs={'降噪': '无', '详情': '完整规格' * 300}) for i in range(10)]
    plan = TaskPlan(['PRODUCT_SEARCH'], [TaskStep('step-1', 'shopping', 'search_products', {},
                    status='SUCCEEDED', result={'products': products, 'observation': {'timestamp': 'original-time'}})])
    return state, plan


def test_context_keeps_goal_and_indexes_without_truncating_original_records():
    state, plan = fixture()
    original = deepcopy(plan.payload())
    engine = ContextEngine(state['request'])
    messages, manifest = engine.decision_messages('system', state, plan, [], max_chars=1500)
    payload = json.loads(messages[-1].content)
    assert payload['goal'] == state['goal']
    assert payload['request'] == state['request']
    assert payload['working_memory'] == state['working_memory']
    assert payload['objective_ledger'] == state['objective_ledger']
    assert manifest['archived_refs'] == ['step-1']
    assert 'products' not in payload['observations'][0]['result']
    assert state['evidence_access'] == {}
    assert state['read_evidence_refs'] == []
    assert plan.payload() == original
    assert sum(_text_length(m) for m in messages) <= 1500


def test_page_returns_full_records_and_provenance_without_changing_archive():
    state, plan = fixture()
    original = deepcopy(plan.payload())
    result = ContextEngine.read_evidence(plan, 'step-1', offset=3, limit=2)
    assert result['products'] == original['steps'][0]['result']['products'][3:5]
    assert result['observation']['timestamp'] == 'original-time'
    assert result['evidence_page'] == dict(source_step='step-1', offset=3, next_offset=5, total=10, refreshed=False)
    result['products'][0]['specs']['降噪'] = '篡改'
    assert plan.payload() == original


def test_preference_injection_is_included_in_context_size():
    state, plan = fixture()
    engine = ContextEngine(state['request'], preferences=[{'key': 'features', 'value': {'value': ['完整偏好' * 200]}}])
    messages, _ = engine.decision_messages('system', state, plan, [], max_chars=3000)
    assert sum(_text_length(m) for m in engine.model_messages(messages)) <= 3000


def test_required_context_that_cannot_fit_fails_without_truncating():
    state, plan = fixture()
    state['request'] *= 1000
    original = deepcopy(state)
    with pytest.raises(ContextBudgetError):
        ContextEngine(state['request']).decision_messages('system', state, plan, [], max_chars=1500)
    assert state == original


@pytest.mark.parametrize('reference,offset,limit', [('foreign:step-1', 0, 1), ('step-1', -1, 1),
                                                ('step-1', 11, 1), ('step-1', 0, 4)])
def test_invalid_page_or_foreign_reference_cannot_read(reference, offset, limit):
    _, plan = fixture()
    with pytest.raises(AgentError):
        ContextEngine.read_evidence(plan, reference, offset=offset, limit=limit)


def test_long_history_is_evicted_before_failing_required_context():
    from langchain_core.messages import HumanMessage, AIMessage
    state, plan = fixture()
    messages, manifest = ContextEngine(state['request']).decision_messages('system', state, plan,
        [HumanMessage(content='很久以前的对话' * 5000), AIMessage(content='旧回复' * 5000),
         HumanMessage(content='最近的短消息')], max_chars=1800)
    payload = json.loads(messages[-1].content)
    assert payload['request'] == state['request'] and payload['goal'] == state['goal']
    assert payload['history'][-1]['content'] == '最近的短消息'
    assert manifest['history_total'] == 3
    assert manifest['omitted_history_indices'] == [0, 1]


def test_newly_requested_evidence_page_takes_priority_over_old_history():
    from langchain_core.messages import HumanMessage
    state, plan = fixture()
    page = ContextEngine.read_evidence(plan, 'step-1', offset=3, limit=1)
    plan.steps.append(TaskStep('step-2', 'shopping', 'read_evidence', {'reference': 'step-1'},
                               status='SUCCEEDED', result=page))
    messages, manifest = ContextEngine(state['request']).decision_messages('system', state, plan,
        [HumanMessage(content='旧历史' * 700), HumanMessage(content='更近历史' * 500)], max_chars=3500)
    payload = json.loads(messages[-1].content)
    assert payload['observations'][-1]['result']['products'] == page['products']
    assert manifest['omitted_history_indices']
    assert state['evidence_access']['step-2'] == [page['products'][0]['product_id']]
    assert sum(_text_length(m) for m in messages) <= 3500


def test_all_verified_outcome_evidence_precedes_old_history():
    from langchain_core.messages import HumanMessage
    state, plan = fixture()
    for i in (2, 3):
        plan.steps.append(TaskStep(f'step-{i}', 'shopping', 'read_evidence', {'reference': 'step-1'},
            status='SUCCEEDED', result=ContextEngine.read_evidence(plan, 'step-1', offset=i, limit=1)))
    plan.steps.append(TaskStep('step-4', 'order', 'get_order_status', {'order_id': 1001},
                              status='SUCCEEDED', result={'order_id': 1001, 'status': 'delivered'}))
    state['objective_ledger'] = [dict(objective_id=f'outcome-{i}', status='VERIFIED', evidence_refs=[f'step-{i}'])
                                 for i in (2, 3, 4)]
    messages, _ = ContextEngine(state['request']).decision_messages('system', state, plan,
        [HumanMessage(content='旧历史' * 600), HumanMessage(content='更近历史' * 450)], max_chars=6000)
    observations = json.loads(messages[-1].content)['observations']
    assert all('products' in observations[i - 1]['result'] for i in (2, 3))
    assert observations[3]['result']['status'] == 'delivered'
    assert sum(_text_length(m) for m in messages) <= 6000
