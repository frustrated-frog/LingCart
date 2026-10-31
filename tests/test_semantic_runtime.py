"""独立响应驱动真实执行链；覆盖原文隔离、修订、恢复和副作用拦截。"""
from copy import deepcopy
import json

from langchain_core.messages import AIMessage, HumanMessage
import pytest

from src import tools
from src.adaptive import run_adaptive
from src.backend.db import get_engine, get_session_factory
from src.backend.seed import seed
from src.durable import run_conversation, resume_run
from src.errors import AgentError
from src.persistence import RunStore
from src.request_semantics_runtime import extract_contract
from src.runtime import BudgetLimits, RunRuntime, ToolExecutor, ToolSpec, TraceRecorder, runtime_scope
from tests.test_adaptive import model, search, recommendation, clarification, goal, CrashAfterSearch, Crash


@pytest.fixture(autouse=True)
def database(tmp_path):
    original = tools._engine, tools._SessionLocal
    engine = get_engine(f'sqlite:///{tmp_path / "semantics.db"}')
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
    yield
    tools._engine, tools._SessionLocal = original
    engine.dispose()


def extracted(*, quantity=1, price=500, quote='预算500', restriction=False):
    rows = [dict(requirement_id='ear', kind='OUTCOME', description='耳机推荐', source_quote='耳机',
                 outcome=dict(kind='recommend', quantity=quantity)),
            dict(requirement_id='budget', kind='CONDITION', description='预算', source_quote=quote, targets=['ear'],
                 predicate=dict(description='预算', source_quote=quote, field='price', op='lte', value=price))]
    if restriction:
        rows.append(dict(requirement_id='no_pay', kind='PROHIBITION', description='不支付', source_quote='不要支付', restriction='NO_PAYMENT'))
    return dict(changes=[dict(operation='ADD', requirement=row) for row in rows])


class SemanticModel:
    def __init__(self, *responses, native=False):
        self.responses = iter(responses)
        self.inputs = []
        self.native_protocol = native
    def native_tools(self, declarations):
        self.declarations = declarations
        return self
    def invoke(self, messages):
        payload = json.loads(messages[-1].content)
        self.inputs.append(payload)
        response = next(self.responses)
        response = response(payload) if callable(response) else response
        if self.native_protocol:
            return AIMessage(content='', tool_calls=[dict(name='extract_requirements', args=response, id='semantics', type='tool_call')])
        return AIMessage(content=json.dumps(response, ensure_ascii=False))


def semantic_model(monkeypatch, *responses, native=False):
    semantic = SemanticModel(*responses, native=native)
    monkeypatch.setattr('src.request_semantics_runtime.get_llm', lambda **_: semantic)
    return semantic


@pytest.mark.parametrize('native', [False, True])
def test_independent_extraction_is_goal_free_cached_and_budgeted(monkeypatch, native):
    actor = model(monkeypatch, search(first=True), recommendation)
    semantic = semantic_model(monkeypatch, extracted(), native=native)
    result = run_conversation('推荐预算500的耳机', 'semantic', runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert len(semantic.inputs) == 1
    assert not {'goal', 'observations', 'plan', 'objective_ledger'} & semantic.inputs[0].keys()
    assert result['budget']['consumed']['model_calls'] == len(actor.inputs) + 3  # 提取、目标审查、完成审查
    assert result['request_semantics']['sha256'] == RunStore().conversation('semantic')['state']['request_semantics']['sha256']
    assert RunStore().run(result['run_id'])['checkpoint']['adaptive']['schema_version'] == 8


def test_quantity_disagreement_blocks_tools_and_repair_keeps_independent_contract(monkeypatch):
    first = search(first=True)
    corrected = deepcopy(first)
    corrected['goal']['objectives'][0]['quantity'] = 2
    actor = model(monkeypatch, first, corrected, recommendation, lambda p: recommendation(p, 1, 'step-1'))
    semantic = semantic_model(monkeypatch, extracted(quantity=2))
    result = run_adaptive('推荐两款预算500的耳机')
    assert result['status'] == 'SUCCEEDED', result
    assert len(semantic.inputs) == 1
    assert actor.inputs[1]['feedback']['code'] == 'REQUEST_SEMANTICS_MISMATCH'
    assert not actor.inputs[1]['observations']
    assert result['plan']['objective_ledger'][0]['target_count'] == 2


def test_unknown_requires_clarification_before_tools(monkeypatch):
    model(monkeypatch, search(first=True))
    response = extracted()
    response['changes'].append(dict(operation='ADD', requirement=dict(requirement_id='unknown', kind='UNKNOWN',
        description='确认预算是总额还是单件', source_quote='合计还是单件没想好')))
    semantic_model(monkeypatch, response)
    result = run_adaptive('推荐预算500的耳机，合计还是单件没想好')
    assert result['status'] == 'WAITING_CLARIFICATION', result
    assert result['terminal_reason'] == 'REQUEST_SEMANTICS_UNCERTAIN'
    assert not result['trace']


def test_invalid_semantic_output_is_bounded_and_never_executes(monkeypatch):
    model(monkeypatch, search(first=True))
    semantic = semantic_model(monkeypatch, {'changes': []}, {'changes': []})
    result = run_adaptive('推荐预算500的耳机')
    assert len(semantic.inputs) == 2
    assert result['status'] == 'WAITING_CLARIFICATION'
    assert result['terminal_reason'] == 'SEMANTIC_EXTRACTION_INVALID'
    assert not result['trace']


def test_interrupted_run_reuses_extraction_and_preserves_v8(monkeypatch):
    model(monkeypatch, search(first=True), recommendation)
    semantic = semantic_model(monkeypatch, extracted())
    with pytest.raises(Crash):
        run_conversation('推荐预算500的耳机', 'semantic-resume', runner=run_adaptive, store=CrashAfterSearch())
    active = RunStore().conversation('semantic-resume')['active_run_id']
    result = resume_run('semantic-resume', active, runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert len(semantic.inputs) == 1
    assert RunStore().run(active)['checkpoint']['adaptive']['schema_version'] == 8


def test_crash_after_extraction_commit_replays_without_a_second_model_call(monkeypatch):
    class CrashAfterExtraction(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            if event and event['type'] == 'request_semantics_extracted':
                raise Crash()
    actor = model(monkeypatch, search(first=True), recommendation)
    semantic = semantic_model(monkeypatch, extracted())
    with pytest.raises(Crash):
        run_conversation('推荐预算500的耳机', 'extraction-crash', runner=run_adaptive, store=CrashAfterExtraction())
    active = RunStore().conversation('extraction-crash')['active_run_id']
    result = resume_run('extraction-crash', active, runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert len(actor.inputs) == 2 and len(semantic.inputs) == 1


def test_compound_steering_preserves_keyboard_candidates_and_reverifies_final_delivery(monkeypatch):
    from tests.test_request_semantics import REQUEST, goal as compound_goal, initial, revision
    from src.session_control import ControlStore
    initial_goal = compound_goal()
    latest = '耳机改一款，不超过400元，键盘不变，先别购买'
    revised_goal = deepcopy(initial_goal)
    revised_goal['objectives'][0].update(quantity=1, source_quote='耳机改一款', hard_constraints=[
        dict(description='预算', source_quote='不超过400元', field='price', op='lte', value=400)])
    def choose(objective, ref):
        def choose_from(payload):
            observation = next(s for s in payload['observations'] if s['step_id'] == ref)
            return dict(action='recommend', objective_id=objective,
                product_id=observation['result']['products'][0]['product_id'], evidence_refs=[ref], reason='已读候选满足预算')
        return choose_from
    def steer(payload):
        active = RunStore().conversation('compound-semantics')['active_run_id']
        ControlStore().enqueue('compound-semantics', active, 'STEER', message=latest,
                               command_id='compound-semantics-steer', expected_revision=0)
        return clarification()
    def select_preserved(payload):
        rows = {row['objective_id']: row for row in payload['objective_ledger']}
        assert rows['key']['candidate_product_ids']
        assert rows['key']['candidate_evidence_refs'] == ['step-2']
        assert rows['key']['status'] == 'PENDING'
        assert rows['ear']['status'] == 'INVALIDATED'
        return choose('key', 'step-1')(payload)
    actor = model(monkeypatch,
        dict(action='tool', goal=initial_goal, tool='search_products', args=dict(category='键盘', max_price=300, limit=2)),
        choose('key', 'step-1'), steer, dict(action='set_goal', goal=revised_goal), select_preserved,
        dict(action='tool', tool='search_products', args=dict(category='耳机', max_price=400, limit=2)),
        choose('ear', 'step-4'),
        dict(action='complete', claims=[dict(objective_id='key', evidence_refs=['step-3']), dict(objective_id='ear', evidence_refs=['step-5'])]))
    initial_extraction = dict(changes=[dict(operation='ADD', requirement=row['requirement']) for row in initial()['requirements']])
    semantic = semantic_model(monkeypatch, initial_extraction, revision())
    result = run_conversation(REQUEST, 'compound-semantics', runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert result['semantic_revision']['preserved'] == ['keyboard']
    assert result['semantic_revision']['changed'] == ['headphones']
    assert semantic.inputs[1]['change_text'] == latest
    assert len(actor.inputs) == 8
    assert result['plan']['verification']['all_goals_satisfied']
    assert all(row['evidence_refs'][0] not in {'step-2', 'step-3', 'step-5'} for row in result['plan']['objective_ledger'])
    assert result['execution_boundaries'][0]['restriction'] == 'READ_ONLY'


def test_completed_conversation_inherits_and_explicitly_replaces_budget(monkeypatch):
    revised = search(first=True)
    revised['goal'] = goal(hard_constraints=[dict(description='预算', source_quote='改成200元以内', field='price', op='lte', value=200)])
    model(monkeypatch, search(first=True), recommendation, revised, recommendation)
    def revise(payload):
        assert payload['prior_contract']['requirements'][0]['requirement']['requirement_id'] == 'ear'
        row = extracted(price=200, quote='改成200元以内')['changes'][1]['requirement']
        return dict(changes=[dict(operation='PRESERVE', previous_id='ear'),
                             dict(operation='REPLACE', previous_id='budget', source_quote='改成200元以内', requirement=row)])
    semantic = semantic_model(monkeypatch, extracted(), revise)
    first = run_conversation('推荐预算500的耳机', 'semantic-followup', runner=run_adaptive)
    second = run_conversation('改成200元以内，其他要求保留', 'semantic-followup', runner=run_adaptive)
    assert first['status'] == second['status'] == 'SUCCEEDED', second
    assert second['semantic_revision']['changed'] == ['ear']
    assert second['request_semantics']['previous_sha256'] == first['request_semantics']['sha256']
    assert len(semantic.inputs) == 2


def test_semantic_extraction_refuses_to_truncate_human_history(monkeypatch):
    semantic = semantic_model(monkeypatch, extracted())
    context = RunRuntime('small-context', BudgetLimits(max_context_chars=100))
    with pytest.raises(AgentError, match='上下文预算'):
        extract_contract(context, '推荐预算500的耳机', [HumanMessage(content='原始条件' * 1000)])
    assert not semantic.inputs


@pytest.mark.parametrize('restriction', ['READ_ONLY', 'NO_PAYMENT', 'NO_SUBMISSION'])
def test_boundary_blocks_even_an_approved_write_before_preflight_and_replay(restriction):
    from src.execution_boundaries import apply_dialogue_boundaries
    calls = []
    def pay(order_id: int):
        calls.append(order_id)
        return {'success': True}
    context = RunRuntime('boundary', BudgetLimits())
    context.approval = {'tool': 'pay', 'args': {'order_id': 1003}}
    with runtime_scope(context):
        recorder = TraceRecorder(context.run_id)
        executor = ToolExecutor({'pay': ToolSpec(pay, side_effect=True, approval_action='PROCESS_PAYMENT',
            preflight=lambda args, ctx: calls.append('preflight'))}, recorder)
        executor.execute('pay', {'order_id': 1003})
        apply_dialogue_boundaries(context, [dict(restriction=restriction, source_quote='先别支付')], '先别支付')
        with pytest.raises(AgentError) as error:
            executor.execute('pay', {'order_id': 1003})
    assert error.value.code == 'EXECUTION_BOUNDARY_VIOLATION'
    assert calls == ['preflight', 1003]
    assert context.events[-1]['type'] == 'execution_boundary_blocked'


def test_boundary_release_is_grounded_and_does_not_authorize_payment():
    from src.execution_boundaries import apply_dialogue_boundaries
    from src.runtime import ApprovalRequired
    context = RunRuntime('release', BudgetLimits())
    with runtime_scope(context):
        apply_dialogue_boundaries(context, [dict(restriction='NO_PAYMENT', source_quote='不要支付')], '不要支付')
        with pytest.raises(AgentError):
            apply_dialogue_boundaries(context, [dict(restriction='NO_PAYMENT', operation='RETRACT', source_quote='允许支付')], '不要支付')
        apply_dialogue_boundaries(context, [dict(restriction='NO_PAYMENT', operation='RETRACT', source_quote='解除支付限制')], '解除支付限制')
        with pytest.raises(ApprovalRequired):
            ToolExecutor({'process_payment': tools.process_payment}, TraceRecorder(context.run_id)).execute('process_payment', {'order_id': 1003, 'method': 'alipay'})
    assert tools.get_payment_status(1003)['status'] == 'pending'


def test_prohibition_persists_into_subsequent_fast_path(monkeypatch):
    model(monkeypatch, search(first=True), recommendation)
    semantic_model(monkeypatch, extracted(restriction=True))
    first = run_conversation('推荐预算500的耳机，不要支付', 'no-pay', runner=run_adaptive)
    assert first['status'] == 'SUCCEEDED', first
    second = run_conversation('帮我支付订单1003', 'no-pay', runner=run_adaptive)
    assert second['terminal_reason'] == 'EXECUTION_BOUNDARY_VIOLATION', second
    assert not second.get('pending_action')
    assert tools.get_payment_status(1003)['status'] == 'pending'


def test_all_outcomes_can_be_retracted_without_fabricating_a_replacement_goal(monkeypatch):
    model(monkeypatch, search(first=True), recommendation,
          dict(action='respond', message='已撤销推荐任务。', commands=[dict(kind='ACKNOWLEDGEMENT', source_quote='耳机推荐全部撤销')]))
    semantic_model(monkeypatch, extracted(), dict(changes=[
        dict(operation='RETRACT', previous_id=identity, source_quote='耳机推荐全部撤销') for identity in ('ear', 'budget')]))
    run_conversation('推荐预算500的耳机', 'retract-all', runner=run_adaptive)
    result = run_conversation('耳机推荐全部撤销', 'retract-all', runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert not result['trace']
    assert result['request_semantics']['requirements'] == []
    assert result['semantic_revision']['retracted'] == ['ear']
    assert RunStore().conversation('retract-all')['state']['request_semantics']['requirements'] == []


def test_dialogue_releases_adaptive_restriction_without_reviving_it_next_turn(monkeypatch):
    from src.execution_boundaries import apply_dialogue_boundaries, apply_semantic_boundaries, current_boundaries
    from src.request_semantics import apply_extraction
    request = '推荐预算500的耳机，不要支付'
    contract = apply_extraction(extracted(restriction=True), request)
    context = RunRuntime('release-contract', BudgetLimits())
    context.snapshot['conversation'] = {'request_semantics': contract}
    with runtime_scope(context):
        apply_semantic_boundaries(context, contract)
        apply_dialogue_boundaries(context, [dict(restriction='NO_PAYMENT', operation='RETRACT', source_quote='解除支付限制')],
                                  '解除支付限制', [HumanMessage(content=request)])
        revised = context.snapshot['request_semantics']
        assert not any(row['requirement']['restriction'] for row in revised['requirements'])
        apply_semantic_boundaries(context, revised)
        assert not current_boundaries(context)
        assert context.approval is None


def test_dialogue_query_only_is_persisted_and_cannot_create_payment_approval(monkeypatch):
    from src import orchestrator
    from tests.test_dialogue_understanding import ScriptedModel
    monkeypatch.setattr(orchestrator, 'get_llm', lambda **_: ScriptedModel(dict(
        reason='只查询，不执行支付', commands=[], execution_boundaries=[dict(restriction='READ_ONLY', operation='DENY', source_quote='只查询')],
        objectives=[dict(id='payment', kind='PAYMENT_PROCESS', source_quote='订单1003支付', order_id=1003)])))
    result = run_conversation('帮我看看订单1003支付，先只查询', 'dialogue-readonly')
    assert result['terminal_reason'] == 'EXECUTION_BOUNDARY_VIOLATION', result
    assert result['pending_action'] is None
    assert result['execution_boundaries'][0]['restriction'] == 'READ_ONLY'


def test_boundary_release_replay_preserves_contract_identity():
    from src.execution_boundaries import apply_dialogue_boundaries
    from src.request_semantics import apply_extraction
    request = '推荐预算500的耳机，不要支付'
    context = RunRuntime('boundary-replay', BudgetLimits())
    context.snapshot['conversation'] = {'request_semantics': apply_extraction(extracted(restriction=True), request)}
    changes = [dict(restriction='NO_PAYMENT', operation='RETRACT', source_quote='解除支付限制')]
    with runtime_scope(context):
        apply_dialogue_boundaries(context, changes, '解除支付限制', [HumanMessage(content=request)])
        identity = context.snapshot['request_semantics']['sha256']
        apply_dialogue_boundaries(context, changes, '解除支付限制', [HumanMessage(content=request)])
    assert context.snapshot['request_semantics']['sha256'] == identity


def test_invalid_persisted_boundary_cannot_fail_open():
    from src.execution_boundaries import require_allowed
    context = RunRuntime('invalid-boundary', BudgetLimits())
    context.snapshot['execution_boundaries'] = [{'restriction': 'UNKNOWN'}]
    with pytest.raises(AgentError) as error:
        require_allowed(context, 'process_payment', ToolSpec(tools.process_payment, side_effect=True))
    assert error.value.code == 'EXECUTION_BOUNDARY_INVALID'


def test_response_cannot_hide_unknown_requirement_revision(monkeypatch):
    from src.request_semantics import apply_extraction
    from src.request_semantics_runtime import revise_without_execution
    prior = '推荐预算500的耳机'
    context = RunRuntime('unknown-revision', BudgetLimits())
    context.snapshot['conversation'] = {'request_semantics': apply_extraction(extracted(), prior)}
    semantic_model(monkeypatch, dict(changes=[
        dict(operation='PRESERVE', previous_id=identity) for identity in ('ear', 'budget')
    ] + [dict(operation='ADD', requirement=dict(requirement_id='unknown', kind='UNKNOWN',
        description='预算范围不明确', source_quote='总预算还是单件没想好'))]))
    with runtime_scope(context), pytest.raises(AgentError) as error:
        revise_without_execution(context, {'schema_version': 8, 'request': '总预算还是单件没想好'}, [HumanMessage(content=prior)])
    assert error.value.code == 'REQUEST_SEMANTICS_UNCERTAIN'
    assert 'request_semantics' not in context.snapshot


def test_native_schema_preserves_required_property_names(monkeypatch):
    from src.request_semantics import IntentExtraction
    semantic = semantic_model(monkeypatch, extracted(), native=True)
    context = RunRuntime('native-schema', BudgetLimits.adaptive())
    with runtime_scope(context):
        extract_contract(context, '推荐预算500的耳机', [])
    parameters = semantic.declarations[0]['function']['parameters']
    change = parameters['properties']['changes']['items']
    requirement = next(part for part in change['properties']['requirement']['anyOf'] if part.get('type') == 'object')
    assert 'description' in requirement['properties']
    assert set(requirement['required']) == set(requirement['properties'])
    predicate = next(part for part in requirement['properties']['predicate']['anyOf'] if part.get('type') == 'object')
    assert 'description' in predicate['properties']
    assert set(IntentExtraction.model_json_schema()['$defs']['IntentRequirement']['required']) <= set(requirement['properties'])


def test_first_response_cannot_silently_drop_requested_business_outcomes(monkeypatch):
    from src.request_semantics_runtime import revise_without_execution
    semantic_model(monkeypatch, extracted())
    context = RunRuntime('missing-first-goal', BudgetLimits.adaptive())
    with runtime_scope(context), pytest.raises(AgentError) as error:
        revise_without_execution(context, {'schema_version': 8, 'request': '推荐预算500的耳机'}, [])
    assert error.value.code == 'REQUEST_SEMANTICS_MISMATCH'


def test_semantic_extractor_ignores_planner_context_injection(monkeypatch):
    class ForeignContext:
        def model_messages(self, messages):
            raise AssertionError('Independent extractor must not use planner context')
    context = RunRuntime('isolated-semantics', BudgetLimits.adaptive())
    context.context_engine = ForeignContext()
    semantic_model(monkeypatch, extracted())
    with runtime_scope(context):
        contract = extract_contract(context, '推荐预算500的耳机', [])
    assert contract['requirements']


def test_invalid_schema_feedback_keeps_paths_without_echoing_model_input(monkeypatch):
    invalid = extracted()
    invalid['changes'][0]['requirement']['description'] = ''
    semantic = semantic_model(monkeypatch, invalid, extracted())
    context = RunRuntime('schema-feedback', BudgetLimits.adaptive())
    with runtime_scope(context):
        extract_contract(context, '推荐预算500的耳机', [])
    feedback = json.loads(semantic.inputs[1]['format_error'])
    assert feedback[0]['loc'] == ['changes', 0, 'requirement', 'description']
    assert 'input' not in feedback[0]
    assert any(event['type'] == 'request_semantics_rejected' for event in context.events)
