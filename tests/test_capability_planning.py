from copy import deepcopy

import pytest

from src.capability_planner import PlanningAction, search_plan
from src.capability_planning import CapabilityPlanningMechanism, built_in_contracts, compile_problem
from src.direct_executor import _READ_ONLY_TOOLS
from src.errors import AgentError
from src.harness.registry import get_registry
from src.runtime import BudgetLimits, RunRuntime
from src.task_plan import TaskPlan, TaskStep
from tests.test_adaptive import database, goal, model, recommendation, search  # noqa: F401


def action(name, requires, adds, deletes=()):
    return PlanningAction(action_id=name, tool=name, scope='test', requires=tuple(requires),
                          adds=tuple(adds), deletes=tuple(deletes))


def test_search_derives_shared_prerequisites_and_respects_delete_effects():
    operators = [action('lookup', [], ['identity']), action('read', ['identity'], ['details']),
                 action('check', ['identity'], ['eligible']),
                 action('invalidate', ['details'], ['result'], ['eligible']),
                 action('finish', ['eligible', 'result'], ['done'])]
    solved = search_plan(operators, [], ['done', 'details'])
    assert solved.status == 'SOLVED'
    facts = set()
    for item in solved.actions:
        assert set(item.requires) <= facts
        facts = (facts - set(item.deletes)) | set(item.adds)
    assert {'done', 'details'} <= facts
    assert len(solved.actions) == 5


def test_search_distinguishes_missing_capabilities_and_budget_exhaustion():
    operators = [action('one', [], ['a']), action('two', ['a'], ['b'])]
    assert search_plan(operators, [], ['missing']).status == 'UNREACHABLE'
    assert search_plan(operators, [], ['b'], max_steps=1).status == 'SEARCH_LIMIT'
    assert search_plan(operators, [], ['b'], max_nodes=1).status == 'SEARCH_LIMIT'
    assert search_plan(operators, ['b'], ['b'], max_steps=0).actions == ()


def state_for(*objectives):
    return dict(goal={'objectives': list(objectives)}, goal_revision=0, capability_step_floor=0)


def order_objective(kind='check_refund_eligibility', order_id=1001, identity='refund'):
    return dict(id=identity, kind=kind, order_id=order_id)


def order_step(order_id=1001, tool='get_order_status', **updates):
    return TaskStep('step-1', 'order', tool, {'order_id': order_id}, status='SUCCEEDED',
                    result=updates or {'order_id': order_id, 'status': 'delivered'})


def problem(state, steps=()):
    return compile_problem(state, TaskPlan([], list(steps)), built_in_contracts(), _READ_ONLY_TOOLS)


def test_verified_observation_is_scoped_to_one_order():
    state = state_for(order_objective(), order_objective(order_id=1002, identity='other'))
    actions, facts, targets, _, _ = problem(state, [order_step()])
    assert 'order:1001/order.status' in facts
    assert 'order:1002/order.status' not in facts
    plan = search_plan(actions, facts, targets)
    second = next(a for a in plan.actions if a.tool == 'check_refund_eligibility' and a.args['order_id'] == 1002)
    assert 'order:1002/order.status' in second.requires
    assert len(plan.actions) == 3


@pytest.mark.parametrize('step', [
    TaskStep('step-1', 'order', 'get_order_status', {'order_id': 1001}, status='SUCCEEDED',
             result={'order_id': 1002, 'status': 'delivered'}),
    order_step(order_id=1001, error='NOT_FOUND'),
    TaskStep('step-1', 'shopping', 'search_products', {}, status='SUCCEEDED', result={'products': [], 'total': 0}),
    TaskStep('step-1', 'order', 'get_order_status', {'order_id': 1001}, status='FAILED',
             result={'order_id': 1001, 'status': 'delivered'}),
])
def test_missing_failed_and_empty_outputs_cannot_satisfy_predicted_effects(step):
    assert problem(state_for(order_objective()), [step])[1] == {}


def test_valid_negative_business_answer_is_still_evidence():
    step = order_step(tool='check_refund_eligibility', order_id=1001,
                      eligible=False, reason='超过期限')
    step.result['order_id'] = 1001
    assert 'order:1001/order.refund_check' in problem(state_for(order_objective()), [step])[1]


def test_failed_producer_is_replaced_and_completed_steps_are_unchanged():
    state = state_for(order_objective())
    step = TaskStep('step-1', 'order', 'get_order_status', {'order_id': 1001}, status='FAILED',
                    error={'code': 'TOOL_TIMEOUT'})
    original = deepcopy(step)
    actions, facts, targets, _, _ = problem(state, [step])
    solved = search_plan(actions, facts, targets)
    assert solved.status == 'SOLVED'
    assert [a.tool for a in solved.actions][-1] == 'check_refund_eligibility'
    assert all(a.tool != 'get_order_status' for a in solved.actions)
    assert step == original


def test_goal_revision_does_not_reuse_old_observations():
    state = state_for(order_objective())
    state.update(goal_revision=1, capability_step_floor=1)
    assert problem(state, [order_step()])[1] == {}


def test_precommit_check_blocks_missing_dependency_and_links_actual_evidence():
    mechanism = CapabilityPlanningMechanism()
    state = state_for(order_objective())
    candidate = TaskStep('step-2', 'refund', 'check_refund_eligibility', {'order_id': 1001})
    with pytest.raises(AgentError, match='前置观察'):
        mechanism.bind_step(state, TaskPlan([], []), candidate)
    mechanism.bind_step(state, TaskPlan([], [order_step()]), candidate)
    assert candidate.depends_on == ['step-1']
    assert candidate.argument_provenance['order_id']['objective_ids'] == ['refund']
    with pytest.raises(AgentError) as error:
        mechanism.bind_step(state, TaskPlan([], []), TaskStep('x', 'order', 'get_order_details', {'order_id': 1002}))
    assert error.value.code == 'CAPABILITY_INPUT_UNGROUNDED'


def test_replanning_is_idempotent_and_preserves_unknown_objectives():
    mechanism = CapabilityPlanningMechanism()
    state = state_for(order_objective(), {'id': 'unknown', 'kind': 'unresolved'})
    context = RunRuntime('planning', BudgetLimits.adaptive())
    plan = TaskPlan([], [])
    first = mechanism.refresh(context, state, plan)
    assert mechanism.refresh(context, state, plan) == first
    assert len(context.events) == 1
    assert first['delegated_objectives'] == ['unknown']
    assert not first['completion_verified']
    plan.steps.append(order_step())
    changed = mechanism.refresh(context, state, plan)
    assert changed['previous_plan_id'] == first['plan_id']
    assert changed['nodes'][0]['tool'] == 'check_refund_eligibility'
    assert changed['nodes'][0]['evidence_refs'] == ['step-1']


def test_cgap_runs_real_tools_and_completes_through_existing_audit(monkeypatch):
    request = '推荐预算500的耳机，同时查询订单1001的详情和退款资格'
    current = goal()
    current['objectives'] += [
        dict(id='details', kind='get_order_details', order_id=1001, source_quote='订单1001的详情', description='查询订单详情'),
        dict(id='refund', kind='check_refund_eligibility', order_id=1001, source_quote='退款资格', description='查询退款资格')]

    def choose(payload):
        observed = payload['observations']
        if not any(s['tool'] == 'search_products' for s in observed):
            return search()
        if not any(s['tool'] == 'select_product' for s in observed):
            ref = next(s['step_id'] for s in observed if s['tool'] == 'search_products')
            return recommendation(payload, ref=ref)
        return dict(action='complete', reason='逐项目标已有观察', claims=[
            dict(objective_id=oid, evidence_refs=[next(s['step_id'] for s in reversed(observed) if s['tool'] == tool)])
            for oid, tool in [('shopping', 'select_product'), ('details', 'get_order_details'), ('refund', 'check_refund_eligibility')]])

    script = model(monkeypatch, dict(goal=current, reason='理解复合目标'), *([choose] * 10))
    result = get_registry().resolve('adaptive-cgap-v1')(request)
    assert result['status'] == 'SUCCEEDED', result
    steps = result['plan']['steps']
    details = next(s for s in steps if s['tool'] == 'get_order_details')
    refund = next(s for s in steps if s['tool'] == 'check_refund_eligibility')
    assert refund['depends_on'] == [details['step_id']]
    assert not any(s['tool'] == 'get_order_status' for s in steps)
    assert any(e['type'] == 'adaptive_decision' and e['source'] == 'capability_planner' for e in result['events'])
    assert result['plan']['verification']['all_goals_satisfied']
    assert any('capability_plan' in p for p in script.inputs)


def test_plan_projection_reports_empty_partial_and_stale_without_inventing_state():
    from src.capability_planning import project_capability_planning
    assert project_capability_planning([], {})['status'] == 'EMPTY'
    state = state_for(order_objective())
    context = RunRuntime('projection', BudgetLimits.adaptive())
    view = CapabilityPlanningMechanism().refresh(context, state, TaskPlan([], []))
    result = project_capability_planning([*context.events, {'type': 'capability_plan_updated'}],
        {'goal_revision': 1, 'adaptive': {'capability_plan': view}})
    assert result['status'] == 'PARTIAL' and result['stale']
    assert result['invalid_records'] == 1
    assert len(result['history']) == 1


def test_durable_resume_reuses_committed_reads_without_replanning_pending_action(monkeypatch):
    from src import tools
    from src.durable import run_conversation, resume_run
    from src.persistence import RunStore
    from tests.test_adaptive import Crash

    class CrashAfterOrder(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            if event and event.get('type') == 'tool_result' and event.get('tool') == 'get_order_details':
                raise Crash()
            return super().save_checkpoint(run_id, token, snapshot, event)

    current = goal()
    current['objectives'].append(dict(id='details', kind='get_order_details', order_id=1001,
        source_quote='订单1001详情', description='查询详情'))
    model(monkeypatch, dict(goal=current), *([dict(action='finish', message='保留真实部分结果')] * 4))
    runner = get_registry().resolve('adaptive-cgap-v1')
    with pytest.raises(Crash):
        run_conversation('推荐预算500的耳机，查询订单1001详情', 'cgap-resume', runner=runner, store=CrashAfterOrder())
    store = RunStore()
    active = store.conversation('cgap-resume')['active_run_id']
    before = store.run(active)['checkpoint']['adaptive']['pending']
    assert before['decision']['tool'] == 'get_order_details'
    result = resume_run('cgap-resume', active, runner=runner)
    details = [s for s in result['plan']['steps'] if s['tool'] == 'get_order_details']
    assert len(details) == 1 and details[0]['status'] == 'REUSED'
    assert store.run(active)['checkpoint']['adaptive']['capability_plan']['facts']


def test_payment_and_refund_writes_are_never_planning_actions():
    from src.runtime import ToolSpec
    def write(order_id: int):
        raise AssertionError('规划不应执行写工具')
    registry = dict(_READ_ONLY_TOOLS, get_order_details=ToolSpec(write, side_effect=True))
    actions, _, _, _, _ = compile_problem(state_for(order_objective()), TaskPlan([], []), built_in_contracts(), registry)
    assert all(a.tool not in {'get_order_details', 'process_payment', 'initiate_refund'} for a in actions)


def test_profile_and_experiment_freeze_one_planning_variable_and_reject_product_use():
    from pathlib import Path
    from eval.experiment_contract import ExperimentSpec
    from eval.run_live_eval import load_dataset
    from src.harness.registry import get_product_runner
    root = Path(__file__).resolve().parents[1]
    spec = ExperimentSpec.from_path(root / 'research/experiments/capability-grounded-planning-v1/experiment-spec.json')
    assert [(c.slot, c.implementation_id) for c in spec.changes] == [('planning', 'capability-grounded-planning')]
    baseline = get_registry().resolve(spec.baseline_profile).profile
    candidate = get_registry().resolve(spec.candidate_profile).profile
    assert [m for m in baseline.mechanisms if m.slot != 'planning'] == [m for m in candidate.mechanisms if m.slot != 'planning']
    assert load_dataset(root / spec.dataset)['split_counts'] == {'dev': 3, 'test': 5}
    with pytest.raises(AgentError) as error:
        get_product_runner(spec.candidate_profile)
    assert error.value.code == 'HARNESS_PROFILE_NOT_APPROVED'
