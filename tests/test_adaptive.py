"""Scripted model protocol tests with real tools; these do not measure LLM quality."""
from copy import deepcopy
import json
from pathlib import Path

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
from src.runtime import BudgetLimits, ToolSpec, tool_spec
from src.task_plan import TaskPlan, TaskStep


@pytest.fixture(autouse=True)
def database(tmp_path, monkeypatch):
    engine = get_engine(f'sqlite:///{tmp_path / "adaptive.db"}')
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
    def unavailable(**kwargs):
        raise AgentError('MODEL_UNAVAILABLE', '未配置模型', retryable=False)
    monkeypatch.setattr('src.adaptive_decision.get_llm', unavailable)
    monkeypatch.setattr('src.request_audit.get_llm', unavailable)
    yield
    engine.dispose()


def goal(**updates):
    return dict(dict(summary='选购耳机', objectives=[dict(id='shopping', description='推荐耳机',
        source_quote='耳机', kind='recommend')], hard_constraints=[
        dict(description='预算上限500', source_quote='预算500', field='price', op='lte', value=500)],
        soft_preferences=[], uncertainties=[]), **updates)


def search(*, first=False, query='', **kwargs):
    return dict(action='tool', reason='获取候选证据', tool='search_products',
                args=dict(query=query, category='耳机', limit=2, **kwargs), **({'goal': goal()} if first else {}))


def recommendation(payload, index=0, ref=None):
    observation = next(s for s in reversed(payload['observations']) if s['step_id'] == ref) if ref else payload['observations'][-1]
    product = observation['result']['products'][index]
    return dict(action='recommend', reason='根据用户偏好和已观察规格选择', product_id=product['product_id'], objective_id='shopping',
                evidence_refs=[observation['step_id']])


def clarification(payload=None):
    return dict(action='clarify', reason='现有证据不足，需要用户决定', message='是否调整预算或选择条件？')


class Script:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.inputs = []
    def invoke(self, messages):
        payload = json.loads(messages[-1].content)
        self.inputs.append(payload)
        decision = next(self.decisions)
        content = decision(payload) if callable(decision) else decision
        return AIMessage(content=json.dumps(content, ensure_ascii=False),
                         usage_metadata={'input_tokens': 100, 'output_tokens': 80, 'total_tokens': 180})


class DialogueScript:
    """Structured dialogue model used before the adaptive shopping loop."""

    def __init__(self, payload):
        self.payload = payload

    def json_mode(self):
        return self

    def invoke(self, _prompt):
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False),
                         usage_metadata={'input_tokens': 100, 'output_tokens': 80, 'total_tokens': 180})


def model(monkeypatch, *decisions):
    script = Script(decisions)
    monkeypatch.setattr('src.adaptive_decision.get_llm', lambda **kwargs: script)
    class AuditModel:
        def invoke(self, messages):
            payload = json.loads(messages[-1].content)
            current = payload['goal']
            requirements = [dict(source_quote=o['source_quote'], interpretation=o['description'],
                objective_ids=[o['id']], constraint_refs=[], status='COVERED') for o in current['objectives']]
            for i, c in enumerate(current['hard_constraints']):
                requirements.append(dict(source_quote=c['source_quote'], interpretation=c['description'],
                    objective_ids=[o['id'] for o in current['objectives']], constraint_refs=[f'global:{i}'], status='COVERED'))
            for o in current['objectives']:
                for i, c in enumerate(o['hard_constraints']):
                    requirements.append(dict(source_quote=c['source_quote'], interpretation=c['description'],
                        objective_ids=[o['id']], constraint_refs=[f"objective:{o['id']}:{i}"], status='COVERED'))
            return AIMessage(content=json.dumps(dict(verdict='APPROVED', requirements=requirements, issues=[])),
                usage_metadata={'input_tokens': 100, 'output_tokens': 80, 'total_tokens': 180})
    monkeypatch.setattr('src.request_audit.get_llm', lambda **kwargs: AuditModel())
    return script


def selected(result):
    return [s['result']['selected_product'] for s in result['plan']['steps']
            if s['tool'] == 'select_product' and s['status'] == 'SUCCEEDED'][-1]


def install(monkeypatch, name, function):
    import src.direct_executor as direct
    original = direct._READ_ONLY_TOOLS[name]
    monkeypatch.setitem(direct._READ_ONLY_TOOLS, name, ToolSpec(function, tool_spec(name, original).schema))


def change(identity, **values):
    with tools._session() as session:
        product = session.get(Product, identity)
        for key, value in values.items():
            setattr(product, key, value)
        session.commit()


def test_no_model_does_not_run_an_autonomous_workflow():
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'FAILED'
    assert result['terminal_reason'] == 'MODEL_UNAVAILABLE'
    assert '自主模式未执行' in result['reply']
    assert not result['trace'] and not result['autonomous_executed']
    assert not result['plan']['verification']['all_goals_satisfied']


def test_uncertain_initial_intent_can_be_clarified_without_fabricating_a_goal(monkeypatch):
    dialogue = DialogueScript(dict(
        reason='用户未说明要处理的消费事项',
        commands=[dict(kind='CLARIFY', source_quote='帮我处理一下', message='你希望处理哪件消费事项？')],
        objectives=[],
    ))
    monkeypatch.setattr('src.llm.get_llm', lambda **kwargs: dialogue)
    result = run_adaptive('帮我处理一下')
    assert result['status'] == 'WAITING_CLARIFICATION'
    assert result['plan']['goal'] is None
    assert not result['trace']
    assert not result['plan']['verification']['all_goals_satisfied']


def test_intent_declaration_is_a_separate_persisted_model_phase(monkeypatch):
    script = model(monkeypatch, {'reason': '先完整理解用户目标，不预设工具动作', 'goal': goal()}, search(), recommendation)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'SUCCEEDED', result
    decisions = [e['decision'] for e in result['events'] if e['type'] == 'adaptive_decision']
    assert decisions[0]['action'] == 'set_goal' and decisions[0]['tool'] is None
    assert decisions[1]['action'] == 'tool'
    assert script.inputs[1]['goal']['objectives'][0]['id'] == 'shopping'


def test_missing_explanatory_reason_does_not_block_a_valid_read(monkeypatch):
    first = search(first=True)
    first.pop('reason')
    model(monkeypatch, first, recommendation)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'SUCCEEDED', result


def test_adaptive_faq_question_is_bound_to_its_grounded_source_quote(monkeypatch):
    from src.adaptive_decision import ShoppingGoal, validate_goal

    request = '最后说明标准配送时效和 30 天退货限制'
    declared = ShoppingGoal.model_validate({
        'summary': '回答配送与退货政策',
        'objectives': [{
            'id': 'policy',
            'description': '回答配送与退货政策',
            'source_quote': request,
            'kind': 'answer_faq',
            'question': '标准配送通常多久，退货有哪些条件？',
        }],
    })

    normalized = validate_goal(declared, request, None)

    assert declared.objectives[0].question == request
    assert normalized == [{
        'objective_id': 'policy',
        'field': 'question',
        'raw_value': '标准配送通常多久，退货有哪些条件？',
        'normalized_value': request,
        'resolver': 'exact_source_quote_binding',
        'resolver_version': '1',
    }]


def test_adaptive_search_normalizes_grounded_catalog_alias_with_provenance(monkeypatch):
    keyboard_goal = goal(
        summary='选购键盘',
        objectives=[dict(id='shopping', description='推荐键盘', source_quote='键盘', kind='recommend')],
        hard_constraints=[
            dict(description='键盘品类', source_quote='键盘', field='category', op='eq', value='keyboard'),
            dict(description='预算上限400', source_quote='预算400', field='price', op='lte', value=400),
        ],
    )
    first = dict(action='tool', reason='查询真实目录', goal=keyboard_goal, tool='search_products',
                 args=dict(query='键盘', category='keyboard', max_price=400, limit=2))
    model(monkeypatch, first, recommendation)

    result = run_adaptive('推荐预算400的键盘')

    assert result['status'] == 'SUCCEEDED', result
    search_step = next(step for step in result['plan']['steps'] if step['tool'] == 'search_products')
    assert search_step['args']['category'] == '键盘'
    assert search_step['argument_provenance']['category']['raw_text'] == 'keyboard'
    assert search_step['argument_provenance']['category']['normalized_value'] == '键盘'
    normalized = next(event for event in result['events'] if event['type'] == 'tool_arguments_normalized')
    assert normalized['raw_args']['category'] == 'keyboard'
    assert normalized['normalized_args']['category'] == '键盘'


def test_pg_contract_blocks_exact_empty_result_replay_before_second_tool_call(monkeypatch):
    from src.context_engine import ContextEngine
    from src.pg_context import PGContextEngine
    from src.procedural_graph import ProceduralGraph
    from src.runtime import RunRuntime, runtime_scope

    calls = []
    def empty_search(**kwargs):
        calls.append(kwargs)
        return {'products': [], 'total': 0, 'filters': kwargs, 'data_source': 'TEST_CATALOG'}
    install(monkeypatch, 'search_products', empty_search)
    first = search(first=True)
    repeated = search()
    model(monkeypatch, first, repeated, clarification)
    monkeypatch.setattr('src.pg_context.get_llm',
                        lambda **_: pytest.fail('确定性指导契约不得调用指导模型'))
    context = RunRuntime('pg-contract-integration', BudgetLimits(max_model_calls=8))
    context.context_engine = PGContextEngine(ContextEngine('推荐预算500的耳机'),
        ProceduralGraph.from_path(Path('research/graphs/pg-seed-v1.json')),
        mechanism_version='3', retrieval_strategy='trajectory-state-v2',
        guidance_strategy='deterministic-contract-v1')

    with runtime_scope(context):
        result = run_adaptive('推荐预算500的耳机')

    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert len(calls) == 1
    reviews = [event for event in result['events'] if event['type'] == 'pg_decision_reviewed']
    assert any(event['status'] == 'BLOCKED' and event['reason'] == 'EXACT_NO_PROGRESS_REPEAT'
               for event in reviews)
    assert any(event['type'] == 'decision_rejected'
               and event['code'] == 'PG_GUIDANCE_CONTRACT_REJECTED' for event in result['events'])
    assert not any(event['type'] == 'model_call' and event.get('purpose') == 'pg_guidance'
                   for event in result['events'])


def test_pg_v4_blocks_catalog_alias_replay_by_validated_execution_identity(monkeypatch):
    from src.context_engine import ContextEngine
    from src.pg_context import PGContextEngine
    from src.procedural_graph import ProceduralGraph
    from src.runtime import RunRuntime, runtime_scope

    calls = []
    def empty_search(**kwargs):
        calls.append(kwargs)
        return {'products': [], 'total': 0, 'filters': kwargs, 'data_source': 'TEST_CATALOG'}
    install(monkeypatch, 'search_products', empty_search)
    keyboard_goal = goal(
        summary='选购键盘',
        objectives=[dict(id='shopping', description='推荐键盘', source_quote='键盘', kind='recommend')],
        hard_constraints=[
            dict(description='键盘品类', source_quote='键盘', field='category', op='eq', value='keyboard'),
        ],
    )
    first = dict(action='tool', reason='查询真实目录', goal=keyboard_goal, tool='search_products',
                 args=dict(query='键盘', category='keyboard', limit=2))
    alias_replay = dict(action='tool', reason='换用中文品类别名重试', tool='search_products',
                        args=dict(query='键盘', category='键盘', limit=2))
    model(monkeypatch, first, alias_replay, clarification)
    context = RunRuntime('pg-v4-alias-integration', BudgetLimits(max_model_calls=8))
    context.context_engine = PGContextEngine(ContextEngine('推荐键盘'),
        ProceduralGraph.from_path(Path('research/graphs/pg-seed-v2.json')),
        mechanism_version='4', retrieval_strategy='trajectory-state-v2',
        guidance_strategy='deterministic-contract-v1', applicability_strategy='typed-evidence-v1',
        path_ranking_strategy='goal-progress-cost-v1')

    with runtime_scope(context):
        result = run_adaptive('推荐键盘')

    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert calls == [{'query': '键盘', 'category': '键盘', 'limit': 2}]
    blocked = next(event for event in result['events']
                   if event['type'] == 'pg_decision_reviewed' and event['status'] == 'BLOCKED')
    assert blocked['decision_identity_basis'] == 'validated_normalized_tool_args'
    assert blocked['blocked_by']['raw_decision_fingerprint'] != blocked['actual_decision_fingerprint']
    outcomes = [event for event in result['events'] if event['type'] == 'pg_action_outcome']
    assert outcomes and outcomes[0]['review_id']
    assert outcomes[0]['progress_status'] == 'PROGRESSED'
    assert 'GOAL_DEFINED' in outcomes[0]['signals']


def test_repeated_refreshes_do_not_create_artificial_dependency_depth():
    from src.adaptive import _selection_step
    from src.adaptive_decision import Decision
    state = {'goal_revision': 0}
    plan = TaskPlan(['shopping'], [TaskStep('step-1', 'shopping', 'search_products', {})])
    for i in range(2, 8):
        decision = Decision(action='recommend', reason='继续核验同一选择', objective_id='shopping',
                            product_id='P1101', evidence_refs=[f'step-{i - 1}'])
        step = _selection_step(state, plan, decision)
        plan.steps.append(step)
        plan.validate(10, 2)
        assert step.depends_on == ['step-1']
        assert decision.evidence_refs == [f'step-{i - 1}']


def test_pg_v4_repeated_real_product_reads_record_execution_without_information_gain(monkeypatch):
    from src.context_engine import ContextEngine
    from src.pg_context import PGContextEngine
    from src.pg_retrieval import build_trajectory_state
    from src.procedural_graph import ProceduralGraph
    from src.runtime import RunRuntime, runtime_scope

    model(monkeypatch, search(first=True), search(), clarification)
    context = RunRuntime('pg-repeat-products', BudgetLimits(max_model_calls=8))
    context.context_engine = PGContextEngine(ContextEngine('推荐预算500的耳机'),
        ProceduralGraph.from_path(Path('research/graphs/pg-seed-v2.json')),
        mechanism_version='4', retrieval_strategy='trajectory-state-v2',
        guidance_strategy='deterministic-contract-v1', applicability_strategy='typed-evidence-v1',
        path_ranking_strategy='goal-progress-cost-v1')
    with runtime_scope(context):
        result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'WAITING_CLARIFICATION'
    outcomes = [event for event in result['events'] if event['type'] == 'pg_action_outcome'
                and event['action'] == 'search_products']
    assert len(outcomes) == 2
    assert outcomes[0]['information_gain']['status'] == 'NEW_INFORMATION'
    assert outcomes[1]['progress_status'] == 'NO_PROGRESS'
    assert outcomes[1]['execution_progress'] is True
    assert outcomes[1]['learning_eligible'] is False
    # 持久事件恢复后仍然不能将第二次读取奖励给程序边。
    restored = json.loads(json.dumps(result['events']))
    feedback = build_trajectory_state({'goal_revision': 0}, TaskPlan([], []), restored)['edge_feedback']
    assert sum(row['progressed'] for row in feedback.values()) <= len(outcomes[0]['matched_edge_ids'])


def test_final_audit_format_error_is_repairable_and_reaches_auditor(monkeypatch):
    script = model(monkeypatch, search(first=True), recommendation, complete(('shopping', ['step-2'])))
    seen = audit_sequence(monkeypatch, None, {})
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'SUCCEEDED', result
    assert len(script.inputs) == 2  # 审查协议自己修复，不要求规划模型重做业务决策。
    assert sum(e['type'] == 'audit_protocol_retry' for e in result['events']) == 1
    assert 'audit_format_error' in seen[-1]['feedback']
    assert 'delivery_draft' in seen[-1]['feedback']


def test_binding_auditor_sees_field_facts_without_prior_failed_completion(monkeypatch):
    first = search(first=True)
    first['goal']['uncertainties'] = ['目录中的品类字段待查证']
    first['goal']['hard_constraints'].append(dict(description='商品是耳机', source_quote='耳机',
        field=None, op='contains', value='耳机'))
    model(monkeypatch, first, recommendation, refine(field='category', op='eq', value='耳机'), recommendation)
    seen = audit_sequence(monkeypatch)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'SUCCEEDED', result
    mapping = next(p for p in seen if p['feedback']['audit_phase'] == 'bindings')
    assert not mapping['observations'] and not mapping['objective_ledger']
    assert mapping['feedback']['catalog_evidence'][0]['observations'][0]['value'] == '耳机'
    assert mapping['feedback']['intent_constraints']['global:1']['field'] is None
    assert mapping['goal']['hard_constraints'][1]['field'] == 'category'
    assert mapping['goal']['uncertainties'] == []
    assert result['plan']['goal']['uncertainties'] == ['目录中的品类字段待查证']
    assert result['plan']['goal']['hard_constraints'][1]['field'] is None
    assert 'delivery_draft' not in mapping['feedback']


def test_model_receives_action_validation_reason_and_repairs_protocol(monkeypatch):
    malformed = {**search(first=True), 'objective_id': 'shopping'}
    def corrected(payload):
        feedback = payload['feedback']
        assert feedback['code'] == 'MODEL_FORMAT_ERROR'
        assert 'Only a recommendation can select a product' in feedback['message']
        return search(first=True)
    model(monkeypatch, malformed, corrected, recommendation)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'SUCCEEDED'
    assert sum(e['type'] == 'decision_rejected' for e in result['events']) == 1


def refine(ref='global:1', field='specs.降噪', op='not_contains', value='主动降噪', evidence='step-1'):
    return dict(action='refine_constraints', reason='按已读目录字段修正验证映射，保留用户原意',
        bindings=[dict(constraint_ref=ref, field=field, op=op, value=value,
                       evidence_refs=[evidence], reason='真实目录字段与此前的猜测不同')])


def negated_search(*, unknown=False):
    first = search(first=True)
    first['goal']['hard_constraints'].append(dict(description='排除主动降噪耳机', source_quote='不要主动降噪',
        field=None if unknown else 'specs.noise_cancellation', op='not_contains', value='主动降噪'))
    return first


@pytest.mark.parametrize('unknown', [False, True])
def test_agent_refines_observed_catalog_mapping_without_mutating_user_goal(monkeypatch, unknown):
    first = negated_search(unknown=unknown)
    original = deepcopy(first['goal'])
    script = model(monkeypatch, first, refine(), recommendation)
    result = run_adaptive('推荐预算500的耳机，不要主动降噪')
    assert result['status'] == 'SUCCEEDED', result
    assert result['plan']['goal']['hard_constraints'] == original['hard_constraints']
    assert result['plan']['binding_revision'] == 1
    assert result['plan']['constraint_bindings']['global:1']['field'] == 'specs.降噪'
    assert selected(result)['specs']['降噪'] != '主动降噪'
    assert result['binding_audit']['verdict'] == 'APPROVED'
    assert result['completion_audit']['binding_revision'] == 1
    assert script.inputs[-1]['constraint_bindings']['global:1']['evidence_refs'] == ['step-1']
    assert script.inputs[-1]['feedback']['code'] == 'BINDINGS_REFINED'


def test_registry_binding_reserves_terminal_audit_and_completes_with_six_model_calls(monkeypatch):
    from dataclasses import replace
    from src.catalog_concepts import CATALOG_CONCEPT_SCHEMA_REVISION

    request = '推荐一款350元以内的键盘，要三模连接，而且必须是静音轴'
    semantic_goal = dict(summary='选购三模静音键盘', objectives=[dict(id='shopping',
        description='推荐键盘', source_quote='键盘', kind='recommend')], hard_constraints=[
            dict(description='预算不超过350元', source_quote='350元以内', field='price', op='lte', value=350),
            dict(description='使用三模连接', source_quote='三模连接', field=None, op='eq', value=None,
                 catalog_concept=dict(schema_revision=CATALOG_CONCEPT_SCHEMA_REVISION,
                    concept_id='keyboard.connection_mode', operator='eq', canonical_value='tri_mode')),
            dict(description='使用静音轴', source_quote='静音轴', field=None, op='eq', value=None,
                 catalog_concept=dict(schema_revision=CATALOG_CONCEPT_SCHEMA_REVISION,
                    concept_id='keyboard.switch_type', operator='eq', canonical_value='silent')),
        ], soft_preferences=[], uncertainties=[])

    def choose(payload):
        assert payload['feedback']['code'] == 'BINDINGS_REFINED'
        assert payload['goal']['hard_constraints'][1]['field'] == 'specs.连接方式'
        product = next(product for product in payload['observations'][-1]['result']['products']
                       if product['specs']['连接方式'] == '三模'
                       and product['specs']['轴体'] == '静音轴')
        return dict(action='recommend', reason='候选同时满足预算、三模连接和静音轴',
                    product_id=product['product_id'], objective_id='shopping', evidence_refs=['step-1'])

    script = model(monkeypatch,
        dict(reason='冻结用户语义后再读取目录', goal=semantic_goal),
        dict(action='tool', reason='读取键盘目录', tool='search_products',
             args=dict(query='', category='键盘', max_price=350, available_only=True, limit=12)),
        choose)
    limits = replace(BudgetLimits.adaptive(), max_model_calls=6)
    result = run_adaptive(request, limits=limits)

    assert result['status'] == 'SUCCEEDED', result
    assert result['budget']['consumed']['model_calls'] == 6
    assert len(script.inputs) == 3
    assert result['plan']['binding_revision'] == 1
    assert result['plan']['binding_transaction']['source'] == 'catalog_registry'
    assert result['plan']['binding_transaction']['status'] == 'APPLIED'
    assert not any(event['type'] == 'adaptive_decision'
                   and event['decision']['action'] == 'refine_constraints' for event in result['events'])
    surfaces = [event for event in result['events'] if event['type'] == 'decision_action_surface']
    assert surfaces[-1]['mode'] == 'recommendation_commit'
    assert surfaces[-1]['allowed_actions'] == ['recommend', 'clarify', 'finish']
    assert [event['phase'] for event in result['events']
            if event['type'] == 'request_audited'] == ['goal', 'bindings', 'completion']
    assert selected(result)['specs']['连接方式'] == '三模'
    assert selected(result)['specs']['轴体'] == '静音轴'


def test_registry_binding_transaction_resumes_without_repeating_planner_or_revision(monkeypatch):
    from dataclasses import replace
    from src.catalog_concepts import CATALOG_CONCEPT_SCHEMA_REVISION

    class CrashAfterProposal(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            if event and event.get('type') == 'constraint_binding_proposed':
                raise Crash()

    request = '推荐一款350元以内的三模键盘'
    semantic_goal = dict(summary='选购三模键盘', objectives=[dict(id='shopping',
        description='推荐键盘', source_quote='键盘', kind='recommend')], hard_constraints=[
            dict(description='预算不超过350元', source_quote='350元以内', field='price', op='lte', value=350),
            dict(description='使用三模连接', source_quote='三模', field=None, op='eq', value=None,
                 catalog_concept=dict(schema_revision=CATALOG_CONCEPT_SCHEMA_REVISION,
                    concept_id='keyboard.connection_mode', operator='eq', canonical_value='tri_mode')),
        ], soft_preferences=[], uncertainties=[])

    def choose(payload):
        product = next(product for product in payload['observations'][-1]['result']['products']
                       if product['specs']['连接方式'] == '三模')
        return dict(action='recommend', reason='满足预算与三模连接', product_id=product['product_id'],
                    objective_id='shopping', evidence_refs=['step-1'])

    script = model(monkeypatch, dict(reason='冻结语义', goal=semantic_goal),
        dict(action='tool', tool='search_products', reason='读取目录',
             args=dict(query='', category='键盘', max_price=350, available_only=True, limit=12)), choose)
    limits = replace(BudgetLimits.adaptive(), max_model_calls=6)
    with pytest.raises(Crash):
        run_conversation(request, 'registry-binding-resume', runner=run_adaptive,
                         limits=limits, store=CrashAfterProposal())

    store = RunStore()
    active = store.conversation('registry-binding-resume')['active_run_id']
    saved = store.run(active)['checkpoint']['adaptive']
    assert saved['binding_tx']['status'] == 'PROPOSED'
    result = resume_run('registry-binding-resume', active, runner=run_adaptive)

    assert result['status'] == 'SUCCEEDED', result
    assert len(script.inputs) == 3
    assert result['plan']['binding_revision'] == 1
    assert sum(event['type'] == 'constraint_binding_proposed' for event in result['events']) == 1
    assert sum(event['type'] == 'constraint_bindings_refined' for event in result['events']) == 1


def test_verification_feedback_distinguishes_duplicate_constraint_scopes(monkeypatch):
    first = negated_search(unknown=True)
    first['goal']['objectives'][0]['hard_constraints'] = [deepcopy(first['goal']['hard_constraints'][1])]
    # 只绑定目标级条件，全局同义条件仍未绑定，不能被描述相同掩盖。
    script = model(monkeypatch, first, refine('objective:shopping:0'), recommendation, clarification)
    result = run_adaptive('推荐预算500的耳机，不要主动降噪')
    assert result['status'] == 'WAITING_CLARIFICATION'
    assert script.inputs[-1]['feedback']['constraint_failures'][0]['constraint_ref'] == 'global:1'
    assert 'global:1' in script.inputs[-1]['constraint_refs']
    assert 'objective:shopping:0' in script.inputs[-1]['constraint_refs']


@pytest.mark.parametrize('patch,code', [
    (refine(ref='global:99'), 'INVALID_CONSTRAINT_BINDING'),
    (refine(field='specs.invented'), 'UNOBSERVED_BINDING_FIELD'),
    (refine(evidence='step-999'), 'UNREAD_BINDING_EVIDENCE'),
])
def test_invalid_mapping_is_repairable_but_never_accepted(monkeypatch, patch, code):
    script = model(monkeypatch, negated_search(), patch, clarification)
    result = run_adaptive('推荐预算500的耳机，不要主动降噪')
    assert result['status'] == 'WAITING_CLARIFICATION'
    assert result['plan']['binding_revision'] == 0
    assert not result['plan']['constraint_bindings']
    assert script.inputs[-1]['feedback']['code'] == code
    assert not result['plan']['verification']['all_goals_satisfied']


def test_mapping_review_cannot_silently_weaken_budget(monkeypatch):
    script = model(monkeypatch, search(first=True), refine('global:0', 'price', 'lte', 9999), clarification)
    audit_sequence(monkeypatch, None, dict(verdict='REVISE', issues=['将预算500放宽到9999，改变了用户要求'], requirements=[
        dict(source_quote='预算500', interpretation='价格上限仍是500', objective_ids=['shopping'],
             constraint_refs=['global:0'], status='MISINTERPRETED')]))
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'WAITING_CLARIFICATION'
    assert script.inputs[-1]['feedback']['code'] == 'BINDING_AUDIT_REJECTED'
    assert result['plan']['goal']['hard_constraints'][0]['value'] == 500
    assert not result['plan']['constraint_bindings']


def test_mapping_revision_invalidates_old_recommendation_claims(monkeypatch):
    first = compound_search()
    first['goal']['hard_constraints'].append(dict(description='排除主动降噪', source_quote='不要主动降噪',
        field='specs.降噪', op='not_contains', value='主动降噪'))
    script = model(monkeypatch, first, recommendation, refine(op='ne'),
                   order_query(), complete(('shopping', ['step-2']), ('order', ['step-3'])), clarification)
    result = run_adaptive(COMPOUND_REQUEST + '，不要主动降噪')
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert script.inputs[-1]['feedback']['failures'] == [dict(objective_id='shopping', code='RECOMMENDATION_BINDING_STALE')]
    assert result['plan']['objective_ledger'][1]['status'] == 'VERIFIED'


def test_repeated_current_binding_does_not_consume_audit_or_change_revision(monkeypatch):
    script = model(monkeypatch, search(first=True), refine('global:0', 'price', 'lte', 500), recommendation)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'SUCCEEDED', result
    assert script.inputs[-1]['feedback']['code'] == 'BINDING_ALREADY_CURRENT'
    assert result['plan']['binding_revision'] == 0
    assert not any(e['type'] == 'request_audited' and e['phase'] == 'bindings' for e in result['events'])


def test_mapping_and_independent_audit_survive_resume(monkeypatch):
    class CrashAfterBinding(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            if event and event.get('type') == 'adaptive_decision' and event['decision']['action'] == 'refine_constraints':
                raise Crash()
    script = model(monkeypatch, negated_search(), refine(), recommendation)
    with pytest.raises(Crash):
        run_conversation('推荐预算500的耳机，不要主动降噪', 'binding-resume', runner=run_adaptive, store=CrashAfterBinding())
    store = RunStore()
    active = store.conversation('binding-resume')['active_run_id']
    result = resume_run('binding-resume', active, runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert len(script.inputs) == 3
    assert sum(e['type'] == 'request_audited' and e['phase'] == 'bindings' for e in result['events']) == 1
    assert result['plan']['binding_revision'] == 1


def test_steering_during_binding_audit_discards_old_mapping(monkeypatch):
    from src.session_control import ControlStore
    revised = negated_search()
    revised['goal']['hard_constraints'][0].update(source_quote='预算改为200', description='预算200', value=200)
    script = model(monkeypatch, negated_search(), refine(), revised, refine(evidence='step-2'), recommendation)
    import src.request_audit as auditing
    from langchain_core.messages import HumanMessage
    fallback = auditing.get_llm()
    def interrupting_audit(payload):
        active = RunStore().conversation('binding-steer')['active_run_id']
        ControlStore().enqueue('binding-steer', active, 'STEER', message='预算改为200',
                               command_id='binding-steer-change', expected_revision=0)
        return json.loads(fallback.invoke([HumanMessage(content=json.dumps(payload, ensure_ascii=False))]).content)
    audit_sequence(monkeypatch, None, interrupting_audit)
    result = run_conversation('推荐预算500的耳机，不要主动降噪', 'binding-steer', runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert result['plan']['goal_revision'] == 1
    assert selected(result)['price'] <= 200
    assert script.inputs[2]['goal'] is None and 'constraint_bindings' not in script.inputs[2]
    assert sum(e['type'] == 'constraint_bindings_refined' for e in result['events']) == 1


def test_model_decides_after_each_observation_and_can_choose_more_expensive(monkeypatch):
    def compare(payload):
        return dict(action='tool', reason='比较候选的规格差异', tool='compare_products',
                    args={'product_ids': [p['product_id'] for p in payload['observations'][-1]['result']['products']]})
    script = model(monkeypatch, search(first=True), compare, lambda p: recommendation(p, 1))
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'SUCCEEDED', result
    assert result['budget']['consumed']['model_calls'] == len(script.inputs) + sum(e['type'] == 'request_audited' for e in result['events'])
    assert [t['tool'] for t in result['trace']] == ['search_products', 'compare_products', 'search_products']
    assert selected(result)['price'] > script.inputs[1]['observations'][0]['result']['products'][0]['price']
    assert script.inputs[2]['observations'][-1]['tool'] == 'compare_products'
    assert result['plan']['verification']['all_goals_satisfied']


def test_model_can_change_query_after_empty_search_without_forced_compare(monkeypatch):
    script = model(monkeypatch, search(first=True, query='不存在的型号'),
        dict(action='revise_plan', reason='精确型号没有结果，改查品类', next_steps=['按品类查找']),
        search(), recommendation)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'SUCCEEDED', result
    assert script.inputs[1]['observations'][0]['result']['products'] == []
    assert [t['tool'] for t in result['trace']] == ['search_products'] * 3
    assert result['budget']['consumed']['model_calls'] == 6


def test_liveplan_profile_detects_and_blocks_an_exact_no_progress_repeat(monkeypatch):
    from src.harness import get_runner

    script = model(monkeypatch,
        search(first=True), search(), search(), search(query='佩戴舒适'), clarification)

    class Advisor:
        def json_mode(self):
            return self

        def invoke(self, _messages):
            return AIMessage(content=json.dumps({
                'recommended_action': 'CHANGE_APPROACH',
                'target_objective_ids': ['shopping'],
                'reason_code': 'ACTION_OSCILLATION',
                'guidance': '更换检索策略，不要重复相同查询。',
                'confidence': 0.9,
            }), usage_metadata={'input_tokens': 20, 'output_tokens': 10, 'total_tokens': 30})

    monkeypatch.setattr('src.llm.get_llm', lambda **_kwargs: Advisor())
    result = get_runner('adaptive-liveplan-v1')('推荐预算500的耳机')

    assert result['status'] == 'WAITING_CLARIFICATION'
    assert [entry['args']['query'] for entry in result['trace']] == ['', '', '佩戴舒适']
    assert any(event['type'] == 'trajectory_drift_detected' for event in result['events'])
    assert any(event['type'] == 'mechanism_applied' and event.get('slot') == 'steering'
               for event in result['events'])
    assert any(event['type'] == 'intervention_applied' and event['outcome'] == 'ACTION_REJECTED'
               for event in result['events'])
    assert result['liveplan']['advisor_calls'] == 1
    assert result['budget']['consumed']['model_calls'] == len(script.inputs) + 2


def test_negative_scope_is_model_supplied_without_substring_expansion(monkeypatch):
    understood = goal()
    understood['hard_constraints'].append(dict(description='排除主动降噪', source_quote='不要主动降噪',
        field='specs.降噪', op='not_contains', value='主动降噪'))
    first = search(first=True, query='P1102')
    first['goal'] = understood
    with tools._session() as session:
        product = session.get(Product, 'P1102')
        product.specs = {**product.specs, '降噪': '无'}
        product.stock = 8
        product.price = 100
        session.commit()
    model(monkeypatch, first, recommendation)
    result = run_adaptive('推荐预算500的耳机，不要主动降噪')
    assert result['status'] == 'SUCCEEDED', result
    assert selected(result)['specs']['降噪'] == '无'
    assert result['plan']['goal']['hard_constraints'] == understood['hard_constraints']


def test_model_clarifies_uncertainty_before_tools(monkeypatch):
    first = clarification()
    first['goal'] = goal(uncertainties=['不清楚用户是否要隔绝环境声'])
    model(monkeypatch, first)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'WAITING_CLARIFICATION'
    assert not result['trace'] and result['budget']['consumed']['model_calls'] == 1


@pytest.mark.parametrize('values', [{'price': 9999}, {'stock': 0}, {'price': 9999, 'stock': 0},
                                  {'specs': {'降噪': '主动降噪'}}])
def test_changes_after_comparison_are_freshly_verified_and_returned_to_model(monkeypatch, values):
    def compare(p):
        return dict(action='tool', reason='查证候选', tool='compare_products',
                    args={'product_ids': [x['product_id'] for x in p['observations'][-1]['result']['products']]})
    def stale(p):
        decision = recommendation(p)
        change(decision['product_id'], **values)
        return decision
    script = model(monkeypatch, search(first=True), compare, stale, clarification)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert not result['plan']['verification']['all_goals_satisfied']
    assert script.inputs[-1]['feedback']['code'] == 'RECOMMENDATION_VERIFICATION_FAILED'
    fresh = script.inputs[-1]['observations'][-1]['result']['products'][0]
    for key, value in values.items():
        assert fresh[key] == value
    assert not any(s['tool'] == 'select_product' and s['status'] == 'SUCCEEDED' for s in result['plan']['steps'])


def test_model_can_accept_changed_price_using_new_evidence(monkeypatch):
    def stale(p):
        decision = recommendation(p)
        change(decision['product_id'], price=199)
        return decision
    script = model(monkeypatch, search(first=True), stale, recommendation)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'SUCCEEDED', result
    assert selected(result)['price'] == 199
    assert '¥199.00' in result['reply']
    assert script.inputs[-1]['observations'][-1]['status'] == 'REJECTED'
    assert result['budget']['consumed']['tool_calls'] == 3


def test_model_can_choose_another_candidate_after_stock_failure(monkeypatch):
    def stale(p):
        decision = recommendation(p)
        change(decision['product_id'], stock=0)
        return decision
    model(monkeypatch, search(first=True), stale, lambda p: recommendation(p, 1, ref='step-1'))
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'SUCCEEDED', result
    assert selected(result)['stock'] > 0


def test_repeated_failed_selection_is_stopped_before_model_budget_exhaustion(monkeypatch):
    first = search(first=True)
    first['goal']['hard_constraints'][0].update(
        description='价格不超过1元', field='price', op='lte', value=1,
    )
    script = model(monkeypatch, first, recommendation, recommendation, recommendation)

    result = run_adaptive('推荐预算500的耳机')

    assert result['status'] == 'PARTIAL'
    assert result['terminal_reason'] == 'NO_PROGRESS_RECOMMENDATION_LOOP'
    assert len([step for step in result['plan']['steps'] if step['tool'] == 'select_product']) == 2
    assert any(event['type'] == 'loop_blocked'
               and event['code'] == 'NO_PROGRESS_RECOMMENDATION_LOOP' for event in result['events'])
    assert result['budget']['consumed']['model_calls'] < result['budget']['limits']['max_model_calls']
    assert script.inputs[-1]['feedback']['repeat_count'] == 2
    failures = [event for event in result['events'] if event['type'] == 'failure_registered']
    assert [event['occurrence'] for event in failures] == [1, 2]
    assert len({event['signature_sha256'] for event in failures}) == 1


def test_argument_repair_count_survives_crash_without_double_counting_cached_response(monkeypatch):
    class CrashAfterSecondFailure(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            if event and event['type'] == 'failure_registered' and event['occurrence'] == 2:
                raise Crash()
    invalid = dict(action='tool', reason='修正搜索', tool='search_products', args={'unexpected': 'invalid'})
    script = model(monkeypatch, search(first=True), invalid, invalid, invalid)
    with pytest.raises(Crash):
        run_conversation('推荐预算500的耳机', 'failure-resume', runner=run_adaptive, store=CrashAfterSecondFailure())
    store = RunStore()
    run_id = store.conversation('failure-resume')['active_run_id']
    ledger = store.run(run_id)['checkpoint']['failure_ledger']
    assert len(ledger) == 1 and next(iter(ledger.values()))['occurrence'] == 2
    result = resume_run('failure-resume', run_id, runner=run_adaptive)
    assert result['terminal_reason'] == 'DECISION_REPAIR_BUDGET_EXHAUSTED'
    failures = [event for event in result['events'] if event['type'] == 'failure_registered']
    assert [event['occurrence'] for event in failures] == [1, 2, 3]
    assert failures[-1]['blocked']
    assert len(script.inputs) == 4
    assert len(result['trace']) == 1


@pytest.mark.parametrize('predicate', [dict(field='specs.未知特征', op='not_contains', value='主动降噪'),
                                     dict(field=None, op='eq', value=None),
                                     dict(field='price', op='lte', value=1)])
def test_unverifiable_or_failed_hard_constraints_cannot_complete(monkeypatch, predicate):
    first = search(first=True)
    first['goal']['hard_constraints'].append(dict(description='必须核验', source_quote='预算500', **predicate))
    model(monkeypatch, first, recommendation, clarification)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'WAITING_CLARIFICATION'
    assert not result['plan']['verification']['all_goals_satisfied']


def test_model_cannot_silently_relax_goal(monkeypatch):
    second = search()
    second['goal'] = goal(summary='忽略预算', hard_constraints=[])
    model(monkeypatch, search(first=True), second)
    result = run_adaptive('推荐预算500的耳机')
    assert result['terminal_reason'] == 'GOAL_CHANGE_REQUIRES_USER'
    assert len(result['trace']) == 1


@pytest.mark.parametrize('decision,code', [
    (dict(action='tool', reason='写入', tool='process_payment', args={'order_id': 1003, 'method': 'card'}), 'UNKNOWN_TOOL'),
    (dict(action='tool', reason='对比', tool='compare_products', args={'product_ids': ['P9998', 'P9999']}), 'UNGROUNDED_PLAN_ENTITY'),
    (dict(action='recommend', reason='伪造', product_id='P9999', objective_id='shopping', evidence_refs=['step-1']), 'UNGROUNDED_RECOMMENDATION'),
])
def test_unsafe_actions_are_rejected(monkeypatch, decision, code):
    script = model(monkeypatch, search(first=True), decision, *([clarification] if code == 'UNKNOWN_TOOL' else []))
    result = run_adaptive('推荐预算500的耳机')
    if code == 'UNKNOWN_TOOL':
        assert result['status'] == 'WAITING_CLARIFICATION'
        assert script.inputs[-1]['feedback']['code'] == code
        assert all(t['tool'] != 'process_payment' for t in result['trace'])
    else:
        assert result['terminal_reason'] == code
    assert tools.get_payment_status(1003)['status'] == 'pending'


@pytest.mark.parametrize('limits,code', [
    (BudgetLimits(max_model_calls=0), 'MODEL_ROUND_BUDGET_EXHAUSTED'),
    (BudgetLimits(max_tool_calls=0), 'TOOL_BUDGET_EXHAUSTED'),
    (BudgetLimits(max_steps=0), 'PLAN_STEP_BUDGET_EXHAUSTED'),
    (BudgetLimits(max_prompt_tokens=1), 'PROMPT_TOKEN_BUDGET_EXHAUSTED'),
    (BudgetLimits(max_context_chars=1), 'CONTEXT_BUDGET_EXHAUSTED'),
])
def test_shared_budgets_cannot_be_bypassed(monkeypatch, limits, code):
    model(monkeypatch, search(first=True))
    result = run_adaptive('推荐预算500的耳机', limits=limits)
    assert result['terminal_reason'] == code, result
    assert not result['trace']


def test_final_refresh_consumes_tool_budget_and_never_uses_cached_search(monkeypatch):
    model(monkeypatch, search(first=True), recommendation)
    result = run_adaptive('推荐预算500的耳机', limits=BudgetLimits(max_tool_calls=1))
    assert result['terminal_reason'] == 'TOOL_BUDGET_EXHAUSTED'
    assert result['status'] != 'SUCCEEDED'


def test_tool_failure_is_an_observation_for_next_model_decision(monkeypatch):
    calls = []
    def timeout(query='', category=None, max_price=None, available_only=True, limit=12):
        calls.append(query)
        raise TimeoutError('catalog timeout')
    install(monkeypatch, 'search_products', timeout)
    script = model(monkeypatch, search(first=True), clarification)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'WAITING_CLARIFICATION'
    assert len(calls) == 2
    assert script.inputs[-1]['feedback']['code'] == 'TOOL_TIMEOUT'


class Crash(BaseException):
    pass


class CrashAfterSearch(RunStore):
    def save_checkpoint(self, run_id, token, snapshot, event=None):
        if event and event.get('type') == 'tool_result' and event.get('tool') == 'search_products':
            raise Crash()
        return super().save_checkpoint(run_id, token, snapshot, event)


def test_resume_replays_committed_observation_but_refreshes_recommendation(monkeypatch):
    script = model(monkeypatch, search(first=True), recommendation, clarification)
    with pytest.raises(Crash):
        run_conversation('推荐预算500的耳机', 'resume', runner=run_adaptive, store=CrashAfterSearch())
    store = RunStore()
    active = store.conversation('resume')['active_run_id']
    with tools._session() as session:
        for product in session.scalars(select(Product)):
            product.price = 9999
        session.commit()
    result = resume_run('resume', active, runner=run_adaptive)
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert result['trace'][0]['status'] == 'REUSED'
    assert script.inputs[-1]['observations'][-1]['result']['products'][0]['price'] == 9999
    assert len(script.inputs) == 3  # Initial decision wasn't rerun.
    assert result['budget']['consumed']['model_calls'] == len(script.inputs) + sum(e['type'] == 'request_audited' for e in result['events'])


def test_version_four_resume_preserves_goal_and_initializes_empty_bindings(monkeypatch):
    from src.backend.db import Checkpoint
    from src.runtime_state import seal_runtime_snapshot
    script = model(monkeypatch, search(first=True), recommendation)
    with pytest.raises(Crash):
        run_conversation('推荐预算500的耳机', 'v4-resume', runner=run_adaptive, store=CrashAfterSearch())
    store = RunStore()
    active = store.conversation('v4-resume')['active_run_id']
    with tools._session() as session:
        checkpoint = session.get(Checkpoint, active)
        payload = deepcopy(checkpoint.payload)
        original_goal = deepcopy(payload['adaptive']['goal'])
        payload['adaptive']['schema_version'] = 4
        for field in ('constraint_bindings', 'binding_revision', 'binding_audit'):
            payload['adaptive'].pop(field)
        checkpoint.payload = seal_runtime_snapshot(payload, active)
        session.commit()
    result = resume_run('v4-resume', active, runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert result['plan']['goal'] == original_goal
    assert result['plan']['constraint_bindings'] == {} and result['plan']['binding_revision'] == 0
    assert store.run(active)['checkpoint']['adaptive']['schema_version'] == 7
    assert len(script.inputs) == 2


def test_steering_reinterprets_goal_in_model_and_keeps_previous_evidence(monkeypatch):
    from src.session_control import ControlStore
    def steer(p):
        active = RunStore().conversation('steer')['active_run_id']
        ControlStore().enqueue('steer', active, 'STEER', message='预算改为200', command_id='steer-1', expected_revision=0)
        return recommendation(p)  # This now-stale decision must never execute.
    revised = search()
    revised['goal'] = goal(summary='新预算', hard_constraints=[dict(description='预算200',
        source_quote='预算改为200', field='price', op='lte', value=200)], soft_preferences=[], uncertainties=[])
    first = search(first=True)
    first['working_memory'] = dict(summary='按原预算500考虑候选', evidence_refs=[], open_questions=[])
    script = model(monkeypatch, first, steer, revised, recommendation)
    result = run_conversation('推荐预算500的耳机', 'steer', runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert result['plan']['goal_revision'] == 1
    assert selected(result)['price'] <= 200
    assert script.inputs[2]['goal'] is None
    assert script.inputs[1]['working_memory']['summary'] == '按原预算500考虑候选'
    assert script.inputs[2]['working_memory'] is None
    assert script.inputs[2]['objective_ledger'] == []
    assert len(script.inputs[2]['observations']) == 1
    assert result['plan']['journal'][0]['reason'] == 'USER_CONSTRAINT_CHANGED'


def test_payment_uses_existing_approval_path():
    result = run_adaptive('帮我支付订单1003')
    assert result['status'] == 'WAITING_APPROVAL'
    assert result['execution_mode'] == 'auto' and not result['autonomous_executed']
    assert tools.get_payment_status(1003)['status'] == 'pending'
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




def test_resume_after_final_verification_refreshes_again_and_preserves_history(monkeypatch):
    class CrashBeforeCommit(RunStore):
        def finish(self, *args, **kwargs):
            raise Crash()
    model(monkeypatch, search(first=True), recommendation, clarification)
    with pytest.raises(Crash):
        run_conversation('推荐预算500的耳机', 'final-crash', runner=run_adaptive, store=CrashBeforeCommit())
    store = RunStore()
    active = store.conversation('final-crash')['active_run_id']
    old = deepcopy(store.run(active)['checkpoint']['adaptive']['plan']['steps'][-1])
    change(old['result']['selected_product']['product_id'], price=9999, stock=0)
    result = resume_run('final-crash', active, runner=run_adaptive)
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert not result['plan']['verification']['all_goals_satisfied']
    assert result['plan']['steps'][1] == old
    assert result['plan']['steps'][-1]['result']['products'][0]['price'] == 9999


def test_resume_reuses_model_response_saved_before_decision_checkpoint(monkeypatch):
    class CrashAfterModel(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            if event and event.get('type') == 'model_call':
                raise Crash()
    script = model(monkeypatch, search(first=True), recommendation)
    with pytest.raises(Crash):
        run_conversation('推荐预算500的耳机', 'model-crash', runner=run_adaptive, store=CrashAfterModel())
    active = RunStore().conversation('model-crash')['active_run_id']
    result = resume_run('model-crash', active, runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert len(script.inputs) == 2
    assert any(e['type'] == 'model_reused' for e in result['events'])


def test_model_cancellation_during_inference_prevents_action(monkeypatch):
    from src.session_control import ControlStore
    def cancel(p):
        active = RunStore().conversation('cancel-model')['active_run_id']
        ControlStore().enqueue('cancel-model', active, 'CANCEL', command_id='stop', expected_revision=0)
        return search(first=True)
    model(monkeypatch, cancel)
    result = run_conversation('推荐预算500的耳机', 'cancel-model', runner=run_adaptive)
    assert result['status'] == 'CANCELLED'
    assert not result['trace']


def test_autonomous_goal_is_not_rewritten_by_preference_rules(monkeypatch):
    from src.context_engine import ContextEngine
    monkeypatch.setattr(ContextEngine, 'effective_request', lambda self: (_ for _ in ()).throw(AssertionError('不应改写目标')))
    first = clarification()
    first['goal'] = goal()
    script = model(monkeypatch, first)
    result = run_conversation('推荐预算500的耳机', 'original-goal', runner=run_adaptive)
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert script.inputs[0]['request'] == '推荐预算500的耳机'


@pytest.mark.parametrize('plans', [[['策略0'], ['策略1'], ['策略2']], [[], ['策略1'], []]])
def test_plan_revisions_are_model_authored_and_bounded(monkeypatch, plans):
    first = search(first=True)
    first['next_steps'] = ['搜型号']
    model(monkeypatch, first, *[dict(action='revise_plan', reason=f'调整策略{i}', next_steps=steps) for i, steps in enumerate(plans)])
    result = run_adaptive('推荐预算500的耳机')
    assert result['terminal_reason'] == 'REPLAN_BUDGET_EXHAUSTED', result
    assert len(result['plan']['journal']) == 2
    assert len(result['trace']) == 1


def test_invalid_tool_evidence_cannot_be_cited(monkeypatch):
    original = tools.search_products
    def corrupt(query='', category=None, max_price=None, available_only=True, limit=12):
        result = original(query, category, max_price, available_only, limit)
        result['products'][0]['stock'] = -1
        return result
    install(monkeypatch, 'search_products', corrupt)
    model(monkeypatch, search(first=True), recommendation)
    result = run_adaptive('推荐预算500的耳机')
    assert result['terminal_reason'] == 'INVALID_TOOL_RESULT'
    assert not result['plan']['verification']['all_goals_satisfied']


def test_finish_without_recommendation_cannot_claim_success(monkeypatch):
    model(monkeypatch, search(first=True), dict(action='finish', reason='证据不足', message='无法完成推荐。'))
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'PARTIAL'
    assert not result['plan']['verification']['all_goals_satisfied']


@pytest.mark.parametrize('unknown', [None, {}, [], True, 0])
def test_unknown_or_wrong_type_spec_cannot_prove_a_negative(monkeypatch, unknown):
    with tools._session() as session:
        product = session.get(Product, 'P1102')
        product.specs = {**product.specs, '降噪': unknown}
        product.stock, product.price = 8, 100
        session.commit()
    first = search(first=True, query='P1102')
    first['goal']['hard_constraints'].append(dict(description='排除主动降噪', source_quote='不要主动降噪',
        field='specs.降噪', op='ne', value='主动降噪'))
    model(monkeypatch, first, recommendation, clarification)
    result = run_adaptive('推荐预算500的耳机，不要主动降噪')
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert not result['plan']['verification']['all_goals_satisfied']


def test_final_verification_timeout_returns_to_model_without_recommending(monkeypatch):
    original = tools.search_products
    def timeout_on_verify(query='', category=None, max_price=None, available_only=True, limit=12):
        if not available_only:
            raise TimeoutError('final refresh failed')
        return original(query, category, max_price, available_only, limit)
    install(monkeypatch, 'search_products', timeout_on_verify)
    script = model(monkeypatch, search(first=True), recommendation, clarification)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'WAITING_CLARIFICATION'
    assert script.inputs[-1]['feedback']['failures'] == ['TOOL_TIMEOUT']
    assert result['budget']['consumed']['tool_calls'] == 3


def large_search():
    first = search(first=True)
    first['args'].update(limit=60)
    first['goal']['hard_constraints'].append(dict(description='耳机品类', source_quote='耳机', field='category', op='eq', value='耳机'))
    return first


def inspect_page(offset=0, limit=1):
    return dict(action='inspect', reason='按需读取候选完整规格', evidence_ref='step-1', offset=offset, limit=limit)


def test_large_catalog_is_archived_and_model_can_inspect_arbitrary_page(monkeypatch):
    def inspect(payload):
        archived = payload['observations'][0]['result']
        assert archived['archived'] is True and 'products' not in archived
        assert archived['product_count'] > 10
        return inspect_page(offset=4)
    script = model(monkeypatch, large_search(), inspect, recommendation)
    result = run_adaptive('推荐预算500的耳机', limits=BudgetLimits(max_context_chars=9000))
    assert result['status'] == 'SUCCEEDED', result
    all_products = result['plan']['steps'][0]['result']['products']
    assert len(all_products) > 10
    assert selected(result) == all_products[4]
    loaded = script.inputs[-1]['observations'][-1]['result']
    assert loaded['products'] == [all_products[4]]
    assert loaded['evidence_page']['next_offset'] == 5
    assert loaded['evidence_page']['refreshed'] is False
    assert result['budget']['consumed']['tool_calls'] == 2  # Search + final refresh, not archive reads.
    assert any(e['type'] == 'evidence_read' for e in result['events'])
    assert result['plan']['goal']['hard_constraints'][0]['value'] == 500


def test_index_only_recommendation_is_rejected_then_model_reads_evidence(monkeypatch):
    def premature(p):
        return dict(action='recommend', reason='只看到了编号', objective_id='shopping',
                    product_id=p['observations'][0]['result']['product_ids'][0], evidence_refs=['step-1'])
    def repair(p):
        assert p['feedback']['code'] == 'UNREAD_RECOMMENDATION_EVIDENCE'
        return inspect_page()
    model(monkeypatch, large_search(), premature, repair, recommendation)
    result = run_adaptive('推荐预算500的耳机', limits=BudgetLimits(max_context_chars=9000))
    assert result['status'] == 'SUCCEEDED', result
    assert result['budget']['consumed']['model_calls'] == 6
    assert len([s for s in result['plan']['steps'] if s['tool'] == 'select_product']) == 1


def test_archived_reads_do_not_refresh_prices_or_bypass_final_verifier(monkeypatch):
    def drift(p):
        identity = p['observations'][0]['result']['product_ids'][0]
        change(identity, price=9999, stock=0)
        return inspect_page()
    script = model(monkeypatch, large_search(), drift, recommendation, clarification)
    result = run_adaptive('推荐预算500的耳机', limits=BudgetLimits(max_context_chars=9000))
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert script.inputs[2]['observations'][-1]['result']['products'][0]['price'] < 500
    assert script.inputs[3]['feedback']['code'] == 'RECOMMENDATION_VERIFICATION_FAILED'
    assert script.inputs[3]['observations'][-1]['result']['products'][0]['price'] == 9999


def test_archive_reference_is_run_local_and_unknown_reference_returns_to_model(monkeypatch):
    script = model(monkeypatch, large_search(), dict(action='inspect', reason='读取错误编号',
        evidence_ref='other-run:step-1'), clarification)
    result = run_adaptive('推荐预算500的耳机', limits=BudgetLimits(max_context_chars=9000))
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert script.inputs[-1]['feedback']['code'] == 'EVIDENCE_NOT_FOUND'
    assert result['budget']['consumed']['tool_calls'] == 1


def test_archive_and_pending_inspection_survive_restart(monkeypatch):
    class CrashBeforeRead(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            if event and event.get('type') == 'adaptive_decision' and event['decision']['action'] == 'inspect':
                raise Crash()
    script = model(monkeypatch, large_search(), inspect_page(offset=2), recommendation)
    with pytest.raises(Crash):
        run_conversation('推荐预算500的耳机', 'archive-resume', runner=run_adaptive,
                         limits=BudgetLimits(max_context_chars=9000), store=CrashBeforeRead())
    store = RunStore()
    active = store.conversation('archive-resume')['active_run_id']
    source = deepcopy(store.run(active)['checkpoint']['adaptive']['plan']['steps'][0])
    result = resume_run('archive-resume', active, runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert result['plan']['steps'][0] == source
    assert selected(result) == source['result']['products'][2]
    assert len(script.inputs) == 3


def test_evidence_inspection_is_bounded_by_step_budget(monkeypatch):
    model(monkeypatch, large_search(), inspect_page())
    result = run_adaptive('推荐预算500的耳机', limits=BudgetLimits(max_steps=1, max_context_chars=9000))
    assert result['terminal_reason'] == 'PLAN_STEP_BUDGET_EXHAUSTED'
    assert not any(e['type'] == 'evidence_read' for e in result['events'])


def test_model_working_memory_survives_archiving_and_is_not_a_hard_constraint(monkeypatch):
    first = large_search()
    first['working_memory'] = dict(summary='接下来查看不同候选的参数', evidence_refs=[], open_questions=['佩戴方式是否合适？'])
    def inspect(p):
        assert p['working_memory']['open_questions'] == ['佩戴方式是否合适？']
        return dict(inspect_page(offset=1), working_memory=dict(summary='先查看第二个候选，不因价格排序直接决定',
            evidence_refs=['step-1'], open_questions=[]))
    def choose(p):
        assert p['working_memory']['evidence_refs'] == ['step-1']
        assert p['goal']['hard_constraints'][0]['value'] == 500
        return recommendation(p)
    model(monkeypatch, first, inspect, choose)
    result = run_conversation('推荐预算500的耳机', 'working-memory', runner=run_adaptive,
                              limits=BudgetLimits(max_context_chars=9500))
    assert result['status'] == 'SUCCEEDED', result
    saved = RunStore().run(result['run_id'])['checkpoint']['adaptive']
    assert saved['working_memory']['summary'] == '先查看第二个候选，不因价格排序直接决定'


def test_working_memory_cannot_fabricate_evidence(monkeypatch):
    first = search(first=True)
    first['working_memory'] = dict(summary='虚构证据', evidence_refs=['foreign:step-1'], open_questions=[])
    model(monkeypatch, first)
    result = run_adaptive('推荐预算500的耳机')
    assert result['terminal_reason'] == 'UNGROUNDED_WORKING_MEMORY'
    assert not result['trace']


COMPOUND_REQUEST = '推荐预算500的耳机，并查询订单1001状态'


def compound_search(order_id=1001):
    first = search(first=True)
    first['goal']['objectives'].append(dict(id='order', kind='get_order_status',
        description=f'查询订单{order_id}状态', source_quote=f'订单{order_id}', order_id=order_id))
    return first


def order_query(order_id=1001, tool='get_order_status'):
    return dict(action='tool', reason='完成订单查询目标', tool=tool, args=dict(order_id=order_id))


def complete(*claims):
    return dict(action='complete', reason='逐项提交已完成目标的真实证据',
                claims=[dict(objective_id=identity, evidence_refs=refs) for identity, refs in claims])


COMPOUND_COMPLETE = complete(('shopping', ['step-2']), ('order', ['step-3']))


def test_recommendation_does_not_skip_other_objectives(monkeypatch):
    script = model(monkeypatch, compound_search(), recommendation, order_query(), COMPOUND_COMPLETE)
    result = run_adaptive(COMPOUND_REQUEST)
    assert result['status'] == 'SUCCEEDED', result
    assert len(script.inputs) == 4
    assert script.inputs[2]['objective_ledger'][0]['status'] == 'VERIFIED'
    assert script.inputs[2]['objective_ledger'][1]['status'] == 'PENDING'
    assert script.inputs[2]['feedback']['code'] == 'OBJECTIVES_REQUIRE_COMPLETION'
    assert result['plan']['verification']['all_goals_satisfied']
    assert result['plan']['verification']['incomplete_objectives'] == []
    assert [t['tool'] for t in result['trace']] == ['search_products', 'search_products', 'get_order_status', 'search_products']
    assert '订单 1001' in result['reply'] and selected(result)['product_id'] in result['reply']


def test_model_can_do_order_before_shopping_without_fixed_order(monkeypatch):
    first = dict(order_query(), goal=compound_search()['goal'])
    model(monkeypatch, first, search(), recommendation,
          complete(('shopping', ['step-3']), ('order', ['step-1'])))
    result = run_adaptive(COMPOUND_REQUEST)
    assert result['status'] == 'SUCCEEDED', result
    assert result['trace'][0]['tool'] == 'get_order_status'


def test_missing_goal_claim_is_returned_to_model_for_repair(monkeypatch):
    script = model(monkeypatch, compound_search(), recommendation,
        complete(('shopping', ['step-2'])), order_query(), COMPOUND_COMPLETE)
    result = run_adaptive(COMPOUND_REQUEST)
    assert result['status'] == 'SUCCEEDED', result
    assert {key: value for key, value in script.inputs[3]['feedback'].items() if key != 'recovery'} == dict(code='COMPLETION_VERIFICATION_FAILED',
        failures=[dict(objective_id='order', code='MISSING_COMPLETION_CLAIM')])


@pytest.mark.parametrize('query,code', [
    (order_query(1002), 'EVIDENCE_TARGET_MISMATCH'),
    (order_query(tool='get_payment_status'), 'EVIDENCE_KIND_MISMATCH'),
])
def test_wrong_order_or_wrong_query_cannot_satisfy_objective(monkeypatch, query, code):
    script = model(monkeypatch, compound_search(), recommendation, query, COMPOUND_COMPLETE, clarification)
    result = run_adaptive(COMPOUND_REQUEST + '，不用查询订单1002')
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert script.inputs[-1]['feedback']['failures'] == [dict(objective_id='order', code=code)]
    assert not result['plan']['verification']['all_goals_satisfied']
    assert 'order' in result['plan']['verification']['incomplete_objectives']
    assert '尚未完成：查询订单1001状态' in result['reply']


def test_failed_business_lookup_is_not_completed_even_if_model_claims_it(monkeypatch):
    script = model(monkeypatch, compound_search(9999), recommendation, order_query(9999), COMPOUND_COMPLETE, clarification)
    result = run_adaptive(COMPOUND_REQUEST.replace('1001', '9999'))
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert script.inputs[-1]['feedback']['failures'] == [dict(objective_id='order', code='EVIDENCE_NOT_SUCCESSFUL')]
    assert result['plan']['steps'][2]['status'] == 'FAILED'


def test_search_evidence_cannot_replace_verified_recommendation(monkeypatch):
    script = model(monkeypatch, compound_search(), order_query(),
        complete(('shopping', ['step-1']), ('order', ['step-2'])), clarification)
    result = run_adaptive(COMPOUND_REQUEST)
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert script.inputs[-1]['feedback']['failures'] == [dict(objective_id='shopping', code='RECOMMENDATION_NOT_VERIFIED')]
    assert '订单 1001' in result['reply']


def test_supporting_search_citation_is_allowed_with_verified_selection(monkeypatch):
    model(monkeypatch, compound_search(), recommendation, order_query(),
          complete(('shopping', ['step-1', 'step-2']), ('order', ['step-3'])))
    result = run_adaptive(COMPOUND_REQUEST)
    assert result['status'] == 'SUCCEEDED', result
    assert result['plan']['objective_ledger'][0]['evidence_refs'] == ['step-4']
    assert result['plan']['steps'][3]['depends_on'] == ['step-1']


def test_recommendation_cannot_exceed_declared_quantity(monkeypatch):
    script = model(monkeypatch, compound_search(), recommendation,
        lambda p: recommendation(p, 1, 'step-1'), clarification)
    result = run_adaptive(COMPOUND_REQUEST)
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert script.inputs[-1]['feedback']['code'] == 'RECOMMENDATION_QUANTITY_SATISFIED'
    assert len([step for step in result['plan']['steps'] if step['tool'] == 'select_product']) == 1


def test_model_cannot_drop_an_unfinished_objective(monkeypatch):
    changed = dict(search(), goal=goal())
    model(monkeypatch, compound_search(), changed)
    result = run_adaptive(COMPOUND_REQUEST)
    assert result['terminal_reason'] == 'GOAL_CHANGE_REQUIRES_USER'
    assert len(result['plan']['goal']['objectives']) == 2
    assert not result['plan']['verification']['all_goals_satisfied']


def test_product_changes_while_querying_another_goal_are_reverified(monkeypatch):
    def query_after_price_change(payload):
        change(payload['observations'][-1]['result']['selected_product']['product_id'], price=9999, stock=0)
        return order_query()
    script = model(monkeypatch, compound_search(), recommendation, query_after_price_change, COMPOUND_COMPLETE, clarification)
    result = run_adaptive(COMPOUND_REQUEST)
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert script.inputs[-1]['feedback']['code'] == 'RECOMMENDATION_VERIFICATION_FAILED'
    assert script.inputs[-1]['objective_ledger'][0]['status'] == 'INVALIDATED'
    assert script.inputs[-1]['objective_ledger'][1]['status'] == 'VERIFIED'
    assert not result['plan']['verification']['all_goals_satisfied']
    assert '订单 1001' in result['reply'] and '¥109.65' not in result['reply']


def test_final_compound_refresh_cannot_bypass_shared_budget(monkeypatch):
    model(monkeypatch, compound_search(), recommendation, order_query(), COMPOUND_COMPLETE)
    result = run_adaptive(COMPOUND_REQUEST, limits=BudgetLimits(max_tool_calls=3))
    assert result['terminal_reason'] == 'TOOL_BUDGET_EXHAUSTED'
    assert result['status'] == 'PARTIAL'
    assert not result['plan']['verification']['all_goals_satisfied']
    assert result['plan']['objective_ledger'][0]['status'] == 'PENDING'
    assert '订单 1001' in result['reply']


def test_compound_completion_crash_requires_another_fresh_read(monkeypatch):
    class CrashBeforeCommit(RunStore):
        def finish(self, *args, **kwargs):
            raise Crash()
    script = model(monkeypatch, compound_search(), recommendation, order_query(), COMPOUND_COMPLETE, clarification)
    with pytest.raises(Crash):
        run_conversation(COMPOUND_REQUEST, 'compound-resume', runner=run_adaptive, store=CrashBeforeCommit())
    active = RunStore().conversation('compound-resume')['active_run_id']
    saved = RunStore().run(active)['checkpoint']['adaptive']
    original_steps = deepcopy(saved['plan']['steps'])
    change(original_steps[-1]['result']['selected_product']['product_id'], stock=0)
    result = resume_run('compound-resume', active, runner=run_adaptive)
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert result['plan']['steps'][:len(original_steps)] == original_steps
    assert len(script.inputs) == 5  # The accepted completion decision was resumed, not inferred twice.
    assert len([t for t in result['trace'] if t['tool'] == 'get_order_status']) == 1
    assert result['plan']['objective_ledger'][0]['status'] == 'INVALIDATED'


@pytest.mark.parametrize('mutation', ['missing', 'empty', 'duplicate', 'invented_order'])
def test_objective_contract_rejects_missing_or_ungrounded_goals(monkeypatch, mutation):
    first = compound_search()
    if mutation == 'missing':
        del first['goal']['objectives']
    elif mutation == 'empty':
        first['goal']['objectives'] = []
    elif mutation == 'duplicate':
        first['goal']['objectives'].append(first['goal']['objectives'][0])
    else:
        first['goal']['objectives'][1]['order_id'] = 7777
    model(monkeypatch, first, first, first)
    result = run_adaptive(COMPOUND_REQUEST)
    assert result['status'] == 'FAILED', result
    assert result['terminal_reason'] == ('UNGROUNDED_GOAL' if mutation == 'invented_order' else 'DECISION_REPAIR_BUDGET_EXHAUSTED')
    assert not result['trace']


def test_unresolved_objective_cannot_be_silently_treated_as_done(monkeypatch):
    first = search(first=True)
    first['goal']['objectives'].append(dict(id='unknown', kind='unresolved',
        description='确认长期佩戴舒适度', source_quote='佩戴舒适'))
    script = model(monkeypatch, first, recommendation,
        complete(('shopping', ['step-2']), ('unknown', ['step-1'])), clarification)
    result = run_adaptive('推荐预算500的耳机，要求佩戴舒适')
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert script.inputs[-1]['feedback']['failures'][0]['code'] == 'EVIDENCE_KIND_MISMATCH'
    assert 'unknown' in result['plan']['verification']['incomplete_objectives']


def test_goal_specific_constraints_are_checked_for_selected_objective(monkeypatch):
    first = search(first=True)
    first['goal']['objectives'][0]['hard_constraints'] = [dict(description='不能超过1元',
        source_quote='1元以内', field='price', op='lte', value=1)]
    model(monkeypatch, first, recommendation, clarification)
    result = run_adaptive('推荐预算500的耳机，实际上只能买1元以内')
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert result['plan']['objective_ledger'][0]['status'] == 'INVALIDATED'


def test_multiple_recommendations_need_separate_receipts_and_final_refresh(monkeypatch):
    first = search(first=True)
    first['goal']['objectives'].append(dict(id='backup', kind='recommend',
        description='推荐一款备用耳机', source_quote='备用耳机'))
    def backup(p):
        return dict(recommendation(p, 1, 'step-1'), objective_id='backup')
    script = model(monkeypatch, first, recommendation, backup,
        complete(('shopping', ['step-2']), ('backup', ['step-3'])))
    result = run_adaptive('推荐预算500的耳机，再推荐一款备用耳机')
    assert result['status'] == 'SUCCEEDED', result
    assert len(script.inputs) == 4
    assert len([t for t in result['trace'] if t['tool'] == 'search_products']) == 5
    assert len(result['plan']['verification']['objectives']) == 2
    assert all(row['status'] == 'VERIFIED' for row in result['plan']['objective_ledger'])
    assert {row['evidence_refs'][0] for row in result['plan']['objective_ledger']} == {'step-4', 'step-5'}


def test_single_objective_quantity_refresh_rejects_changed_earlier_selection(monkeypatch):
    first = search(first=True)
    first['goal']['objectives'][0]['quantity'] = 2
    def second(payload):
        first_id = payload['objective_ledger'][0]['selected_product_ids'][0]
        change(first_id, stock=0)
        return recommendation(payload, 1, 'step-1')
    script = model(monkeypatch, first, recommendation, second, clarification)
    result = run_adaptive('推荐两款预算500的耳机')
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert not result['plan']['verification']['all_goals_satisfied']
    assert 'OUT_OF_STOCK' in script.inputs[-1]['feedback']['failures']
    assert any(event['type'] == 'completion_requested' and event['source'] == 'runtime'
               for event in result['events'])


def test_one_recommendation_receipt_cannot_complete_two_objectives(monkeypatch):
    first = search(first=True)
    first['goal']['objectives'].append(dict(id='backup', kind='recommend',
        description='推荐备用耳机', source_quote='备用耳机'))
    script = model(monkeypatch, first, recommendation,
        complete(('shopping', ['step-2']), ('backup', ['step-2'])), clarification)
    result = run_adaptive('推荐预算500的耳机，再推荐备用耳机')
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert script.inputs[-1]['feedback']['failures'] == [dict(objective_id='backup', code='RECOMMENDATION_NOT_VERIFIED')]


@pytest.mark.parametrize('question,expected', [('退货政策是什么？', 'SUCCEEDED'), ('如何付款？', 'WAITING_CLARIFICATION')])
def test_faq_completion_matches_the_requested_question(monkeypatch, question, expected):
    first = search(first=True)
    first['goal']['objectives'].append(dict(id='faq', kind='answer_faq',
        description='回答退货政策', source_quote='退货政策是什么？', question='退货政策是什么？'))
    script = model(monkeypatch, first, recommendation,
        dict(action='tool', reason='查询政策', tool='answer_faq', args=dict(question=question)),
        complete(('shopping', ['step-2']), ('faq', ['step-3'])), clarification)
    result = run_adaptive('推荐预算500的耳机，退货政策是什么？')
    assert result['status'] == expected, result
    if expected != 'SUCCEEDED':
        assert script.inputs[-1]['feedback']['failures'] == [dict(objective_id='faq', code='EVIDENCE_TARGET_MISMATCH')]


def test_steering_after_verified_selection_invalidates_its_receipt(monkeypatch):
    from src.session_control import ControlStore
    def steer(p):
        assert p['objective_ledger'][0]['status'] == 'VERIFIED'
        active = RunStore().conversation('compound-steer')['active_run_id']
        ControlStore().enqueue('compound-steer', active, 'STEER', message='只要耳机，预算改为200',
            command_id='goal-steer', expected_revision=0)
        return order_query()
    revised = dict(complete(('shopping', ['step-2'])), goal=goal(hard_constraints=[
        dict(description='预算200', source_quote='预算改为200', field='price', op='lte', value=200)]))
    script = model(monkeypatch, compound_search(), recommendation, steer, revised, clarification)
    result = run_conversation(COMPOUND_REQUEST, 'compound-steer', runner=run_adaptive)
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert script.inputs[3]['objective_ledger'] == []
    assert script.inputs[-1]['feedback']['failures'] == [dict(objective_id='shopping', code='RECOMMENDATION_NOT_VERIFIED')]
    assert result['plan']['goal_revision'] == 1
    assert not any(t['tool'] == 'get_order_status' for t in result['trace'])


def test_crash_during_compound_refresh_retires_pending_attempt(monkeypatch):
    class CrashDuringRefresh(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            pending = snapshot.get('adaptive', {}).get('pending') or {}
            if (event and event.get('type') == 'tool_result' and event.get('tool') == 'search_products'
                    and pending.get('decision', {}).get('action') == 'complete'):
                raise Crash()
    script = model(monkeypatch, compound_search(), recommendation, order_query(), COMPOUND_COMPLETE)
    with pytest.raises(Crash):
        run_conversation(COMPOUND_REQUEST, 'refresh-crash', runner=run_adaptive, store=CrashDuringRefresh())
    active = RunStore().conversation('refresh-crash')['active_run_id']
    result = resume_run('refresh-crash', active, runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert len(script.inputs) == 4
    assert result['plan']['steps'][3]['status'] == 'CANCELLED'
    assert result['plan']['steps'][4]['status'] == 'SUCCEEDED'
    assert not any(s['status'] == 'PENDING' for s in result['plan']['steps'])


def test_old_goal_checkpoint_cannot_skip_new_objective_contract():
    from src.runtime import RunRuntime, runtime_scope
    context = RunRuntime('legacy-adaptive')
    context.snapshot['adaptive'] = dict(schema_version=2)
    with runtime_scope(context):
        result = run_adaptive('推荐预算500的耳机')
    assert result['terminal_reason'] == 'ADAPTIVE_CHECKPOINT_INCOMPATIBLE'
    assert not result['autonomous_executed'] and not result['trace']


def audit_sequence(monkeypatch, *responses):
    import src.request_audit as audit
    fallback = audit.get_llm()
    pending = iter(responses)
    seen = []
    class Auditor:
        def invoke(self, messages):
            payload = json.loads(messages[-1].content)
            seen.append(payload)
            reply = next(pending, None)
            if reply is None:
                return fallback.invoke(messages)
            reply = reply(payload) if callable(reply) else reply
            return AIMessage(content=json.dumps(reply, ensure_ascii=False),
                usage_metadata={'input_tokens': 100, 'output_tokens': 80, 'total_tokens': 180})
    monkeypatch.setattr(audit, 'get_llm', lambda **kwargs: Auditor())
    return seen


def test_independent_audit_catches_initially_omitted_order_before_tools(monkeypatch):
    corrected = compound_search()
    script = model(monkeypatch, search(first=True), corrected, order_query(),
        lambda p: recommendation(p, ref='step-1'), complete(('shopping', ['step-3']), ('order', ['step-2'])))
    seen = audit_sequence(monkeypatch, dict(verdict='REVISE', issues=['漏掉用户要求的订单1001状态查询'], requirements=[
        dict(source_quote='订单1001', interpretation='需要另一个订单状态目标', objective_ids=[], constraint_refs=[], status='MISSING')]))
    result = run_adaptive(COMPOUND_REQUEST)
    assert result['status'] == 'SUCCEEDED', result
    assert script.inputs[1]['feedback']['code'] == 'REQUEST_AUDIT_REJECTED'
    assert len([e for e in result['events'] if e['type'] == 'tool_call' and e['tool'] == 'search_products']) == 3
    assert result['request_audit']['verdict'] == 'APPROVED'
    assert result['completion_audit']['verdict'] == 'APPROVED'
    assert seen[0]['request'] == COMPOUND_REQUEST
    assert any(t['tool'] == 'get_order_status' for t in result['trace'])


def test_independent_audit_rejects_reversed_negation_then_model_corrects(monkeypatch):
    wrong = search(first=True)
    wrong['goal']['hard_constraints'].append(dict(description='必须主动降噪', source_quote='不要主动降噪',
        field='specs.降噪', op='contains', value='主动降噪'))
    corrected = deepcopy(wrong)
    corrected['goal']['hard_constraints'][-1].update(description='排除主动降噪', op='not_contains')
    model(monkeypatch, wrong, corrected, recommendation)
    audit_sequence(monkeypatch, dict(verdict='REVISE', issues=['否定极性理解反了，应排除主动降噪'], requirements=[
        dict(source_quote='不要主动降噪', interpretation='排除主动降噪', objective_ids=['shopping'],
             constraint_refs=['global:1'], status='MISINTERPRETED')]))
    result = run_adaptive('推荐预算500的耳机，不要主动降噪')
    assert result['status'] == 'SUCCEEDED', result
    assert selected(result)['specs']['降噪'] != '主动降噪'
    assert result['plan']['goal']['hard_constraints'][-1]['op'] == 'not_contains'
    assert len([t for t in result['trace'] if t['tool'] == 'search_products']) == 2


def test_model_can_repair_protocol_error_without_losing_budget(monkeypatch):
    def corrected(payload):
        assert payload['feedback']['code'] == 'MODEL_FORMAT_ERROR'
        return search(first=True)
    model(monkeypatch, {'action': 'invented-action'}, corrected, recommendation)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'SUCCEEDED', result
    assert result['budget']['consumed']['model_calls'] == 5
    assert sum(e['type'] == 'decision_rejected' for e in result['events']) == 1


def test_malformed_auditor_response_never_falls_back_to_approval(monkeypatch):
    model(monkeypatch, search(first=True), search(first=True), search(first=True))
    audit_sequence(monkeypatch, *[{} for _ in range(6)])
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'FAILED' and not result['trace']
    assert result['terminal_reason'] == 'NO_PROGRESS_AUDIT_LOOP'
    assert sum(e['type'] == 'failure_registered' and e['signature']['stage'] == 'audit_protocol'
               for e in result['events']) == 3
    assert result.get('request_audit') is None


def test_approved_audit_distinguishes_goal_requirements_from_execution_boundaries():
    from pydantic import ValidationError
    from src.request_audit import RequestAudit
    base = dict(source_quote='耳机', interpretation='推荐耳机', status='COVERED')
    audit = RequestAudit.model_validate(dict(verdict='APPROVED', issues=[], requirements=[
        {**base, 'scope': 'GOAL_REQUIREMENT', 'objective_ids': ['shopping'], 'constraint_refs': []},
        {**base, 'source_quote': '不要执行支付', 'interpretation': '禁止支付',
         'scope': 'EXECUTION_BOUNDARY', 'objective_ids': [], 'constraint_refs': []},
    ]))
    assert audit.verdict == 'APPROVED'

    with pytest.raises(ValidationError):
        RequestAudit.model_validate(dict(verdict='APPROVED', issues=[], requirements=[
            {**base, 'scope': 'GOAL_REQUIREMENT', 'objective_ids': [], 'constraint_refs': []},
        ]))
    with pytest.raises(ValidationError):
        RequestAudit.model_validate(dict(verdict='APPROVED', issues=[], requirements=[
            {**base, 'scope': 'EXECUTION_BOUNDARY', 'objective_ids': ['shopping'], 'constraint_refs': []},
        ]))


def test_final_semantic_rejection_returns_feedback_instead_of_success(monkeypatch):
    def ask(payload):
        assert payload['feedback']['code'] == 'COMPLETION_SEMANTIC_REJECTED'
        return clarification()
    model(monkeypatch, search(first=True), recommendation, ask)
    # 初审通过，最终审查发现证据不足；不能让工具的成功代替用户需求完成。
    audit_sequence(monkeypatch, None, dict(verdict='CLARIFY', issues=['现有证据未说明用户关心的佩戴取舍'], requirements=[
        dict(source_quote='耳机', interpretation='需要说明选购取舍', objective_ids=['shopping'], constraint_refs=[], status='UNCERTAIN')]))
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert not result['plan']['verification']['all_goals_satisfied']
    assert result['completion_audit']['verdict'] == 'CLARIFY'


def test_unknown_invoice_policy_cannot_complete_an_objective(monkeypatch):
    first = search(first=True)
    question = '能否开具增值税专用发票'
    first['goal']['objectives'].append(dict(id='invoice', description=question, source_quote=question,
                                          kind='answer_faq', question=question))
    script = model(monkeypatch, first,
        dict(action='tool', reason='检索发票政策', tool='answer_faq', args={'question': question}),
        lambda p: recommendation(p, ref='step-1'),
        complete(('shopping', ['step-3']), ('invoice', ['step-2'])), clarification)
    result = run_adaptive('推荐预算500的耳机，' + question)
    assert result['status'] == 'WAITING_CLARIFICATION'
    assert not result['plan']['verification']['all_goals_satisfied']
    assert script.inputs[-1]['feedback']['failures'] == [dict(objective_id='invoice', code='EVIDENCE_NOT_SUCCESSFUL')]


def test_tool_parameter_repair_does_not_execute_the_invalid_call(monkeypatch):
    first = search(first=True)
    first['args']['unexpected'] = '不得传入工具'
    def correction(payload):
        assert payload['feedback']['code'] == 'INVALID_TOOL_ARGUMENTS'
        return search()
    model(monkeypatch, first, correction, recommendation)
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'SUCCEEDED'
    assert len(result['trace']) == 2
    assert all('unexpected' not in t['args'] for t in result['trace'])


def test_audit_must_account_for_every_hard_predicate(monkeypatch):
    model(monkeypatch, search(first=True), search(first=True), search(first=True))
    incomplete = dict(verdict='APPROVED', issues=[], requirements=[dict(source_quote='耳机', interpretation='推荐耳机',
                        objective_ids=['shopping'], constraint_refs=[], status='COVERED')])
    audit_sequence(monkeypatch, *[incomplete for _ in range(6)])
    result = run_adaptive('推荐预算500的耳机')
    assert result['status'] == 'FAILED' and not result['trace']
    assert result['terminal_reason'] == 'NO_PROGRESS_AUDIT_LOOP'


def test_steering_during_request_audit_prevents_old_action(monkeypatch):
    from src.session_control import ControlStore
    revised = search(max_price=200)
    revised['goal'] = goal(hard_constraints=[dict(description='新预算200', source_quote='预算改为200', field='price',op='lte',value=200)])
    model(monkeypatch, search(first=True, max_price=500), revised, recommendation)
    def steer_in_audit(payload):
        store=ControlStore()
        run=store.conversation('audit-steer')['active_run_id']
        store.enqueue('audit-steer',run,'STEER',message='预算改为200',command_id='audit-steer-command',expected_revision=0)
        return dict(verdict='APPROVED', issues=[], requirements=[dict(source_quote='预算500',interpretation='预算500元推荐耳机',
            objective_ids=['shopping'],constraint_refs=['global:0'],status='COVERED')])
    audit_sequence(monkeypatch, steer_in_audit)
    result=run_conversation('推荐预算500的耳机','audit-steer',runner=run_adaptive)
    assert result['status']=='SUCCEEDED', result
    assert result['plan']['goal_revision']==1
    assert len(result['trace'])==2
    assert not any(t['args'].get('max_price')==500 for t in result['trace'])


def test_unavailable_final_audit_cannot_turn_a_recommendation_into_success(monkeypatch):
    model(monkeypatch, search(first=True), recommendation)
    import src.request_audit as auditing
    fallback = auditing.get_llm()
    count = []
    def getter(**kwargs):
        count.append(1)
        if len(count)>1:
            raise AgentError('MODEL_UNAVAILABLE','审查模型不可用',retryable=False)
        return fallback
    monkeypatch.setattr(auditing,'get_llm',getter)
    result=run_adaptive('推荐预算500的耳机')
    assert result['status']=='PARTIAL'
    assert result['terminal_reason']=='MODEL_UNAVAILABLE'
    assert not result['plan']['verification']['all_goals_satisfied']
