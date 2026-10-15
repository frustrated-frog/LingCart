"""S4: real catalog evidence, constrained patches, budgets and durable recovery."""
from copy import deepcopy
import json

from langchain_core.messages import AIMessage
import pytest
from sqlalchemy import select

from src import tools
from src.adaptive import PlanPatch, StepDraft, apply_plan_patch, run_adaptive
from src.backend.db import Product, ToolLedger, get_engine, get_session_factory
from src.backend.seed import seed
from src.durable import resume_run, run_conversation
from src.errors import AgentError
from src.persistence import RunStore
from src.runtime import BudgetLimits, RunRuntime, ToolSpec, runtime_scope, tool_spec
from src.task_plan import TaskPlan, TaskStep


@pytest.fixture(autouse=True)
def database(tmp_path):
    path = tmp_path / 'adaptive.db'
    engine = get_engine(f'sqlite:///{path}')
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
    yield path
    engine.dispose()


def selected(result):
    return [s['result']['selected_product'] for s in result['plan']['steps']
            if s['tool'] == 'select_product' and s['status'] == 'SUCCEEDED'][-1]


def install(monkeypatch, name, function):
    import src.direct_executor as direct
    original = direct._READ_ONLY_TOOLS[name]
    monkeypatch.setitem(direct._READ_ONLY_TOOLS, name, ToolSpec(function, tool_spec(name, original).schema))


def test_search_compare_select_are_grounded_and_bounded():
    result = run_adaptive('推荐500元以内的主动降噪头戴式耳机')
    assert result['status'] == 'SUCCEEDED'
    assert result['plan']['verification']['all_goals_satisfied']
    assert result['execution_mode'] == 'adaptive'
    search, comparison = result['trace']
    assert search['tool'] == 'search_products' and comparison['tool'] == 'compare_products'
    observed = {p['product_id'] for p in search['result']['products']}
    assert 2 <= len(comparison['args']['product_ids']) <= 3
    assert set(comparison['args']['product_ids']) <= observed
    product = selected(result)
    assert product['price'] <= 500 and product['stock'] > 0
    assert product['specs']['降噪'] == '主动降噪'
    assert product['specs']['佩戴方式'] == '头戴式'
    assert result['budget']['consumed']['model_calls'] == 0
    assert result['plan']['version'] == 1


def test_single_candidate_uses_search_observation_without_invalid_compare():
    result = run_adaptive('预算1000推荐续航至少60小时的耳机')
    assert result['status'] == 'SUCCEEDED'
    assert selected(result)['product_id'] == 'P1108'
    assert [t['tool'] for t in result['trace']] == ['search_products']
    assert next(s for s in result['plan']['steps'] if s['tool'] == 'compare_products')['status'] == 'NOT_NEEDED'


def test_zero_matches_preserves_budget_and_waits_for_user():
    result = run_adaptive('推荐预算50的降噪耳机')
    assert result['status'] == 'WAITING_CLARIFICATION'
    assert result['terminal_reason'] == 'NO_MATCHING_PRODUCTS'
    assert result['plan']['goal']['max_price'] == 50
    assert not result['plan']['verification']['all_goals_satisfied']
    assert result['plan']['replan_count'] == 0
    assert len(result['trace']) == 1
    assert '预算和条件已保留' in result['reply']


@pytest.mark.parametrize('message', ['推荐预算-1的耳机', '推荐500元以内的头戴式开放式耳机'])
def test_constraint_conflict_stops_before_tools(message):
    result = run_adaptive(message)
    assert result['status'] == 'WAITING_CLARIFICATION'
    assert result['terminal_reason'] == 'CONSTRAINT_CONFLICT'
    assert not result['trace']


def test_feature_keys_do_not_prove_the_feature_and_negative_constraints_hold():
    result = run_adaptive('推荐500元以内的降噪耳机')
    assert selected(result)['specs']['降噪'] != '无'
    excluded = run_adaptive('推荐500元以内不要降噪的耳机')
    assert selected(excluded)['specs']['降噪'] == '无'
    unavailable = run_adaptive('推荐500元以内防水的耳机')
    assert unavailable['status'] == 'WAITING_CLARIFICATION'
    assert unavailable['terminal_reason'] == 'NO_MATCHING_PRODUCTS'


@pytest.mark.parametrize('field,value', [('stock', 0), ('price', 9999)])
def test_observed_stock_or_price_change_replans_only_unfinished_steps(monkeypatch, field, value):
    original = tools.compare_products
    calls = []
    def changing(product_ids: list[str]):
        calls.append(product_ids)
        if len(calls) == 1:
            with tools._session() as session:
                for identity in product_ids:
                    setattr(session.get(Product, identity), field, value)
                session.commit()
        return original(product_ids)
    install(monkeypatch, 'compare_products', changing)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'SUCCEEDED'
    assert result['plan']['version'] == 2 and result['plan']['replan_count'] == 1
    assert result['plan']['journal'][0]['reason'] == 'PRODUCT_CONSTRAINT_CHANGED'
    steps = result['plan']['steps']
    assert steps[0]['tool'] == 'search_products' and steps[0]['status'] == 'SUCCEEDED'
    assert steps[1]['tool'] == 'compare_products' and steps[1]['status'] == 'SUCCEEDED'
    assert len(steps) == 5
    assert len([t for t in result['trace'] if t['tool'] == 'search_products']) == 2
    assert selected(result)['product_id'] not in calls[0]
    assert result['plan']['goal']['max_price'] == 500
    assert any(e['type'] == 'verification_failed' for e in result['events'])
    assert any(e['type'] == 'plan_replanned' for e in result['events'])


def test_replanning_stops_after_two_changes(monkeypatch):
    original = tools.compare_products
    calls = []
    def drifting(product_ids: list[str]):
        calls.append(product_ids)
        result = original(product_ids)
        for p in result['products']:
            p['stock'] = 0
        return result
    install(monkeypatch, 'compare_products', drifting)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'PARTIAL'
    assert result['terminal_reason'] == 'REPLAN_BUDGET_EXHAUSTED'
    assert result['plan']['replan_count'] == 2
    assert len(result['plan']['journal']) == 2
    assert len(calls) == 3
    assert result['budget']['consumed']['tool_calls'] == 6
    assert not result['plan']['verification']['all_goals_satisfied']


def test_tool_retry_is_bounded_by_shared_runtime(monkeypatch):
    calls = []
    def timeout(query='', category=None, max_price=None, available_only=True, limit=12):
        calls.append(query)
        raise TimeoutError('catalog timeout')
    install(monkeypatch, 'search_products', timeout)
    result = run_adaptive('推荐预算500的耳机')
    assert result['terminal_reason'] == 'TOOL_TIMEOUT'
    assert len(calls) == 2
    assert result['budget']['consumed']['tool_calls'] == 2
    assert result['plan']['replan_count'] == 0


@pytest.mark.parametrize('limits,reason', [
    (BudgetLimits(max_steps=2), 'PLAN_STEP_BUDGET_EXHAUSTED'),
    (BudgetLimits(max_depth=2), 'PLAN_DEPTH_BUDGET_EXHAUSTED'),
    (BudgetLimits(max_tool_calls=0), 'TOOL_BUDGET_EXHAUSTED'),
])
def test_shared_budgets_cannot_be_bypassed(limits, reason):
    result = run_adaptive('推荐预算500的耳机', limits=limits)
    assert result['terminal_reason'] == reason
    assert not result['trace']
    assert result['budget']['consumed']['tool_calls'] == 0


def test_replanning_retains_the_total_step_limit(monkeypatch):
    original = tools.compare_products
    def drift(product_ids: list[str]):
        observation = original(product_ids)
        for product in observation['products']:
            product['stock'] = 0
        return observation
    install(monkeypatch, 'compare_products', drift)
    result = run_adaptive('推荐预算500的耳机', limits=BudgetLimits(max_steps=4))
    assert result['terminal_reason'] == 'PLAN_STEP_BUDGET_EXHAUSTED'
    assert result['plan']['version'] == 1
    assert result['plan']['replan_count'] == 0
    assert len(result['plan']['steps']) == 3


def test_untrusted_comparison_id_cannot_complete_selection(monkeypatch):
    original = tools.compare_products
    def invented(product_ids: list[str]):
        observation = original(product_ids)
        observation['products'][0]['product_id'] = 'P9999'
        return observation
    install(monkeypatch, 'compare_products', invented)
    result = run_adaptive('推荐预算500的耳机')
    assert result['terminal_reason'] == 'INVALID_TOOL_RESULT'
    assert not result['plan']['verification']['all_goals_satisfied']
    assert not any(s['tool'] == 'select_product' and s['status'] == 'SUCCEEDED' for s in result['plan']['steps'])


def test_plan_patch_cannot_change_completed_work_or_add_unguarded_write():
    done = TaskStep('step-1', 'payment', 'process_payment', {'order_id': 1003, 'method': 'card'}, status='SUCCEEDED', result={'success': True})
    pending = TaskStep('step-2', 'shopping', 'search_products', {})
    plan = TaskPlan(['PAYMENT_PROCESS', 'PRODUCT_SEARCH'], [done, pending])
    original = deepcopy(plan.payload())
    for patch in [
        PlanPatch(reason='change', replace=[StepDraft(step_id='step-1', tool='search_products')]),
        PlanPatch(reason='cancel', cancel=['step-1']),
        PlanPatch(reason='write', append=[StepDraft(step_id='step-3', tool='process_payment', args={'order_id': 1003, 'method': 'card'})]),
        PlanPatch(reason='untrusted id', append=[StepDraft(step_id='step-3', tool='compare_products', args={'product_ids': ['P1101', 'P9999']})]),
    ]:
        with pytest.raises(AgentError):
            apply_plan_patch(plan, patch, BudgetLimits())
        assert plan.payload() == original
    valid = apply_plan_patch(plan, PlanPatch(reason='cancel unfinished', cancel=['step-2']), BudgetLimits())
    assert valid.steps[0] == done and valid.steps[1].status == 'CANCELLED'


def test_payment_and_pending_approval_use_existing_safe_path():
    result = run_adaptive('帮我支付订单1003')
    assert result['status'] == 'WAITING_APPROVAL'
    assert tools.get_payment_status(1003)['status'] == 'pending'
    assert not any(t['tool'] == 'process_payment' for t in result['trace'])
    unrelated = run_adaptive('推荐500元以内的耳机', pending_action=result['pending_action'])
    assert unrelated['pending_action'] == result['pending_action']
    assert tools.get_payment_status(1003)['status'] == 'pending'


class Crash(BaseException):
    pass


class CrashAfterSearch(RunStore):
    def save_checkpoint(self, run_id, token, snapshot, event=None):
        if event and event.get('type') == 'tool_result' and event.get('tool') == 'search_products':
            raise Crash()
        return super().save_checkpoint(run_id, token, snapshot, event)


def test_resume_reuses_original_plan_and_tool_ledger(database, monkeypatch):
    with pytest.raises(Crash):
        run_conversation('推荐预算500的耳机', 'resume', runner=run_adaptive, store=CrashAfterSearch())
    store = RunStore()
    active = store.conversation('resume')['active_run_id']
    original = deepcopy(store.run(active)['checkpoint']['adaptive'])
    with tools._session() as session:
        # Recovery must use the committed observation even if the database changes.
        for product in session.scalars(select(Product)):
            product.price = 9999
        session.commit()
    tools.configure_db(get_engine(f'sqlite:///{database}'))
    result = resume_run('resume', active, runner=run_adaptive)
    search = next(t for t in result['trace'] if t['tool'] == 'search_products')
    assert search['status'] == 'REUSED'
    assert result['plan']['steps'][0]['args'] == original['plan']['steps'][0]['args']
    assert result['plan']['steps'][0]['result'] == search['result']
    assert result['plan']['replan_count'] == 1
    assert result['status'] == 'WAITING_CLARIFICATION'
    assert result['terminal_reason'] == 'NO_MATCHING_PRODUCTS'
    with tools._session() as session:
        ledgers = session.scalars(select(ToolLedger).where(ToolLedger.tool_name == 'search_products')).all()
        assert len(ledgers) == 2  # Original evidence plus a deliberate fresh replan scope.


def test_steering_changes_only_unfinished_steps_and_keeps_prior_observations(monkeypatch):
    from src.session_control import ControlStore
    original = tools.search_products
    calls = []
    def search(query='', category=None, max_price=None, available_only=True, limit=12):
        result = original(query, category, max_price, available_only, limit)
        calls.append(max_price)
        if len(calls) == 1:
            # A tool runs on the bounded worker, so use the durable active run ID.
            active = RunStore().conversation('steer')['active_run_id']
            ControlStore().enqueue('steer', active, 'STEER', message='预算改为200')
        return result
    install(monkeypatch, 'search_products', search)
    result = run_conversation('推荐预算500的耳机', 'steer', runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED'
    assert selected(result)['price'] <= 200
    assert result['plan']['goal_revision'] == 1 and result['plan']['version'] == 2
    assert result['plan']['goal']['max_price'] == 200
    assert calls == [500, 200]
    original_step = result['plan']['steps'][0]
    assert original_step['status'] == 'SUCCEEDED' and original_step['args']['max_price'] == 500
    assert any(p['price'] > 200 for p in original_step['result']['products'])
    assert result['plan']['journal'][0]['reason'] == 'USER_CONSTRAINT_CHANGED'


def test_natural_goal_uses_budgeted_model_only_for_intake(monkeypatch):
    import src.planner as planner
    class Model:
        def invoke(self, messages):
            return AIMessage(content=json.dumps({'goals': [{'intent': 'PRODUCT_SEARCH', 'category': '耳机', 'max_price': 500.0, 'query': '主动降噪'}]}))
    monkeypatch.setattr(planner, 'get_llm', lambda **kwargs: Model())
    result = run_adaptive('最近想换一副主动降噪耳机，花费控制在500左右')
    assert result['status'] == 'SUCCEEDED'
    assert result['budget']['consumed']['model_calls'] == 1
    assert selected(result)['specs']['降噪'] == '主动降噪'
    assert selected(result)['price'] <= 500


def test_model_intake_cannot_invent_constraints_or_bypass_model_budget(monkeypatch):
    import src.planner as planner
    class Model:
        def invoke(self, messages):
            return AIMessage(content=json.dumps({'goals': [{'intent': 'PRODUCT_SEARCH', 'category': '耳机', 'query': '防水'}]}))
    monkeypatch.setattr(planner, 'get_llm', lambda **kwargs: Model())
    result = run_adaptive('通勤时很吵想换一副耳机')
    assert result['terminal_reason'] == 'UNGROUNDED_PLAN_CONSTRAINT'
    assert not result['trace']
    exhausted = run_adaptive('通勤时很吵想换一副耳机', limits=BudgetLimits(max_model_calls=0))
    assert exhausted['terminal_reason'] == 'MODEL_ROUND_BUDGET_EXHAUSTED'
    assert exhausted['budget']['consumed']['model_calls'] == 0


@pytest.mark.parametrize('message', ['推荐价格不超过100元的耳机', '推荐不高于100元的耳机', '推荐最多100元的耳机'])
def test_front_position_budget_is_a_hard_limit(message):
    result = run_adaptive(message)
    assert result['plan']['goal']['max_price'] == 100
    assert result['status'] == 'WAITING_CLARIFICATION'
    assert result['terminal_reason'] == 'NO_MATCHING_PRODUCTS'


def test_model_plan_retains_explicit_comparison_and_unique_step_ids(monkeypatch):
    import src.planner as planner
    class Model:
        def invoke(self, messages):
            return AIMessage(content=json.dumps({'goals': [
                {'intent': 'PRODUCT_SEARCH', 'category': '耳机', 'max_price': 500.0},
                {'intent': 'PRODUCT_COMPARE', 'product_ids': ['P1102', 'P1103']},
                {'intent': 'ORDER_STATUS', 'order_id': 1001}]}))
    monkeypatch.setattr(planner, 'get_llm', lambda **kwargs: Model())
    result = run_adaptive('我在考虑一副耳机，花费控制在500左右，P1102与P1103这两款如何，另外订单1001现在怎样了')
    assert result['terminal_reason'] != 'INVALID_TASK_PLAN'
    assert set(result['plan']['goal']['product_ids']) == {'P1102', 'P1103'}
    ids = [s['step_id'] for s in result['plan']['steps']]
    assert len(ids) == len(set(ids))
    # 旧版商品编号在当前目录无库存证据，不能改选无关商品声称成功。
    if result['status'] == 'SUCCEEDED':
        assert selected(result)['product_id'] in {'P1102', 'P1103'}
    else:
        assert result['terminal_reason'] == 'NO_MATCHING_PRODUCTS'


def test_explicit_unlimited_budget_clears_initial_limit():
    result = run_adaptive('推荐耳机，预算500元，这次不限制预算')
    assert result['plan']['goal']['max_price'] is None
