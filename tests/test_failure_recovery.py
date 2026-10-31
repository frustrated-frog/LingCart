from copy import deepcopy

import pytest

from src.failure_recovery import recovery_basis, register_failure, signature
from src.runtime import RunRuntime
from src.runtime_state import new_runtime_snapshot, seal_runtime_snapshot
from src.task_plan import TaskPlan, TaskStep


@pytest.mark.parametrize('stage,code', [
    ('decision', 'MODEL_FORMAT_ERROR'), ('decision', 'INVALID_TOOL_ARGUMENTS'),
    ('decision', 'REQUEST_AUDIT_REJECTED'), ('recommendation', 'RECOMMENDATION_VERIFICATION_FAILED'),
    ('audit_protocol', 'AUDIT_FORMAT_ERROR'), ('dialogue_protocol', 'MODEL_FORMAT_ERROR'),
])
def test_shared_failure_identity_is_persistent_idempotent_and_bounded(stage, code):
    runtime = RunRuntime('failure-test', snapshot=new_runtime_snapshot('failure-test', conversation={}, history=[]))
    failure = signature(stage, code, {'target': 'same'}, {'revision': 1})
    first = register_failure(runtime, failure, attempt_id='one')
    assert first['occurrence'] == 1 and not first['blocked']
    restored = RunRuntime('failure-test', snapshot=seal_runtime_snapshot(deepcopy(runtime.snapshot), 'failure-test'))
    assert register_failure(restored, failure, attempt_id='one')['occurrence'] == 1
    assert not restored.events
    assert register_failure(restored, failure, attempt_id='two')['occurrence'] == 2
    final = register_failure(restored, failure, attempt_id='three')
    assert final['blocked'] and final['recovery_action'] == 'STOP'
    assert register_failure(restored, failure, attempt_id='four')['occurrence'] == 3
    assert len(restored.events) == 2
    changed = signature(stage, code, {'target': 'different'}, {'revision': 1})
    assert register_failure(restored, changed, attempt_id='five')['occurrence'] == 1


def test_identical_catalog_reads_do_not_reset_recovery_but_changed_facts_do():
    state = dict(request='推荐耳机', goal={'objectives': []}, goal_revision=0)
    product = {'product_id': 'P1001', 'stock': 0, 'price': 199}
    plan = TaskPlan([], [TaskStep('step-1', 'shopping', 'search_products', {},
                                 status='SUCCEEDED', result={'products': [product], 'observation': {'id': 'old'}})])
    first = recovery_basis(state, plan, product_id='P1001')
    duplicate = deepcopy(plan.steps[0])
    duplicate.step_id = 'step-2'
    duplicate.result['observation'] = {'id': 'new', 'timestamp': 'later'}
    plan.steps.append(duplicate)
    assert recovery_basis(state, plan, product_id='P1001') == first
    duplicate.result['products'][0]['stock'] = 2
    assert recovery_basis(state, plan, product_id='P1001') != first
    revised = {**state, 'goal_revision': 1}
    assert recovery_basis(revised, plan) != recovery_basis(state, plan)
