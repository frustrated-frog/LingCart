"""Native function schemas, real SQLite tools, batching, steering and restart."""
from copy import deepcopy
import json

import jsonschema
from langchain_core.messages import AIMessage
import pytest

from src import tools
from src.adaptive import run_adaptive
from src.adaptive_decision import IntentDeclaration
from src.backend.db import get_engine, get_session_factory
from src.backend.seed import seed
from src.durable import run_conversation, resume_run
from src.model_gateway import ModelCandidate, ProviderConfig, RotatingChatModel
from src.persistence import RunStore
from src.trace_projector import project_trace

REQUEST = '推荐预算500元的耳机，并查询订单1001状态'


@pytest.fixture(autouse=True)
def database(tmp_path):
    original = tools._engine, tools._SessionLocal
    engine = get_engine(f'sqlite:///{tmp_path / "native.db"}')
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
    yield
    tools._engine, tools._SessionLocal = original
    engine.dispose()


def intent(quote=REQUEST, price=500, order=True):
    objectives = [dict(id='shopping', kind='recommend', description='推荐耳机', source_quote=quote)]
    if order:
        objectives.append(dict(id='order', kind='get_order_status', order_id=1001,
                               description='查询订单状态', source_quote=quote))
    return IntentDeclaration.model_validate(dict(goal=dict(summary='消费任务', objectives=objectives,
        hard_constraints=[dict(description='预算', source_quote=quote, field='price', op='lte', value=price)]))).model_dump()


def search(price=500):
    return ('search_products', dict(query='耳机', category='耳机', max_price=price,
                                   available_only=True, limit=2, reason=None))


def recommend(payload, ref='step-1'):
    observation = next(s for s in payload['observations'] if s['step_id'] == ref)
    product = observation['result']['products'][-1]
    return ('recommend', dict(product_id=product['product_id'], objective_id='shopping',
                              evidence_refs=[ref], reason='按已读商品价格、库存和预算选择第二个候选'))


def approve(payload):
    from src.constraint_bindings import constraint_refs
    goal = payload['goal']
    rows = [dict(source_quote=o['source_quote'], interpretation=o['description'], status='COVERED',
                 objective_ids=[o['id']], constraint_refs=[]) for o in goal['objectives']]
    rows[0]['constraint_refs'] = list(constraint_refs(goal))
    return ('submit_audit', dict(verdict='APPROVED', issues=[], requirements=rows))


class Wire:
    def __init__(self, responses, requests=None, declarations=None):
        self.responses = iter(responses) if isinstance(responses, list) else responses
        self.requests = [] if requests is None else requests
        self.declarations = declarations

    def bind(self, **options):
        return self

    def bind_tools(self, declarations, **options):
        assert options['tool_choice'] in {'auto', 'required'}
        return Wire(self.responses, self.requests, declarations)

    def invoke(self, messages):
        payload = json.loads(messages[-1].content)
        self.requests.append(payload)
        response = next(self.responses)
        response = response(payload) if callable(response) else response
        batch = response if isinstance(response, list) else [response]
        functions = {d['function']['name']: d['function'] for d in self.declarations}
        calls = []
        for i, (name, args) in enumerate(batch):
            assert functions[name]['strict'] is True
            jsonschema.validate(args, functions[name]['parameters'])
            calls.append(dict(name=name, args=args, id=f'native_{len(self.requests)}_{i}', type='tool_call'))
        return AIMessage(content='', tool_calls=calls,
                         usage_metadata={'input_tokens': 100, 'output_tokens': 80, 'total_tokens': 180})


def models(monkeypatch, responses, audits=2):
    actor = Wire(responses)
    auditor = Wire([approve] * audits)
    def gateway(wire):
        return RotatingChatModel([ModelCandidate(ProviderConfig('openai_compatible', 'fake-native', 'test-key',
                                    strict_tools=True, thinking_mode='disabled'), wire)], max_attempts=1)
    monkeypatch.setattr('src.adaptive_decision.get_llm', lambda **_: gateway(actor))
    monkeypatch.setattr('src.request_audit.get_llm', lambda **_: gateway(auditor))
    return actor, auditor


def standard_responses():
    return [('declare_intent', intent()), [search(), ('get_order_status', {'order_id': 1001, 'reason': None})],
            recommend, ('complete', {'evidence': {'shopping': ['step-1', 'step-3'], 'order': ['step-2']}})]


def test_native_batch_and_completion_execute_real_tools_without_json_action_wrapper(monkeypatch):
    actor, auditor = models(monkeypatch, standard_responses())
    result = run_conversation(REQUEST, 'native', runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert len(actor.requests) == 4 and len(auditor.requests) == 2
    assert result['budget']['consumed']['model_calls'] == 6
    assert [s['tool'] for s in result['plan']['steps']] == ['search_products', 'get_order_status', 'select_product', 'select_product']
    assert result['plan']['steps'][-1]['depends_on'] == ['step-1']
    assert sum(e['type'] == 'native_submission' for e in result['events']) == 7
    assert project_trace(result['events'])['integrity']['complete']


class Crash(BaseException):
    pass


def test_native_batch_remainder_and_model_response_survive_restart(monkeypatch):
    class CrashAfterRead(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            if event and event.get('type') == 'tool_result' and event.get('tool') == 'search_products':
                raise Crash()
    actor, _ = models(monkeypatch, standard_responses())
    with pytest.raises(Crash):
        run_conversation(REQUEST, 'native-resume', runner=run_adaptive, store=CrashAfterRead())
    active = RunStore().conversation('native-resume')['active_run_id']
    saved = RunStore().run(active)['checkpoint']['adaptive']
    assert saved['native_queue'][0]['decision']['tool'] == 'get_order_status'
    result = resume_run('native-resume', active, runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert len(actor.requests) == 4  # 批次响应没有重新请求模型。
    assert len([s for s in result['plan']['steps'] if s['tool'] == 'search_products']) == 1
    assert len([t for t in result['trace'] if t['tool'] == 'get_order_status']) == 1


def test_steering_cancels_unexecuted_native_batch_entries(monkeypatch):
    from src.direct_executor import _READ_ONLY_TOOLS
    from src.runtime import ToolSpec, tool_spec
    from src.session_control import ControlStore
    original = tool_spec('search_products', _READ_ONLY_TOOLS['search_products'])
    changed = False
    def steer(**args):
        nonlocal changed
        result = original.function(**args)
        if not changed:
            changed = True
            active = RunStore().conversation('native-steer')['active_run_id']
            ControlStore().enqueue('native-steer', active, 'STEER', message='改为只推荐200元以内的耳机，不查订单',
                                   command_id='native-change', expected_revision=0)
        return result
    monkeypatch.setitem(_READ_ONLY_TOOLS, 'search_products', ToolSpec(steer, original.schema))
    responses = standard_responses()[:2] + [('declare_intent', intent('改为只推荐200元以内的耳机', 200, False)),
                 search(200), lambda p: recommend(p, 'step-2')]
    models(monkeypatch, responses, audits=3)
    result = run_conversation(REQUEST, 'native-steer', runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert result['plan']['goal_revision'] == 1
    assert not any(t['tool'] == 'get_order_status' for t in result['trace'])
    assert any(e['type'] == 'native_batch_cancelled' and e['function'] == 'get_order_status' for e in result['events'])


def test_strict_wire_schema_has_no_unsupported_length_keywords():
    from src.native_protocol import strict_schema
    schema = strict_schema(IntentDeclaration.model_json_schema())
    def check(value):
        if isinstance(value, dict):
            assert not set(value) & {'$ref', '$defs', 'minLength', 'maxLength', 'minItems', 'maxItems', 'default'}
            if value.get('type') == 'object':
                assert value['additionalProperties'] is False
                assert set(value['required']) == set(value.get('properties', {}))
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
    check(schema)


def test_multiple_recommendations_in_native_batch_continue_to_completion(monkeypatch):
    request = '推荐两款预算500元以内的耳机'
    goal = intent(quote=request, order=False)
    goal['goal']['objectives'].append({**goal['goal']['objectives'][0], 'id': 'second'})
    def choices(payload):
        name, first = recommend(payload)
        product = payload['observations'][-1]['result']['products'][0]
        return [(name, first), (name, {**first, 'objective_id': 'second', 'product_id': product['product_id']})]
    actor, _ = models(monkeypatch, [('declare_intent', goal), search(), choices,
        ('complete', {'evidence': {'shopping': ['step-2'], 'second': ['step-3']}})])
    result = run_conversation(request, 'native-two', runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert len(actor.requests) == 4
    assert not any(e['type'] == 'native_batch_cancelled' for e in result['events'])


def test_native_batch_rejects_mixed_terminal_and_read_actions_before_execution():
    from src.native_protocol import decode_decisions
    response = AIMessage(content='', tool_calls=[
        dict(name='get_order_status', args={'order_id': 1001}, id='a', type='tool_call'),
        dict(name='finish', args={'message': '结束'}, id='b', type='tool_call')])
    with pytest.raises(ValueError, match='independent'):
        decode_decisions(response, {'goal': intent()['goal']})


def test_native_business_decision_does_not_accept_text_json():
    from src.native_protocol import decode_decisions
    with pytest.raises(ValueError, match='declared function'):
        decode_decisions(AIMessage(content='{"action":"tool","tool":"get_order_status","args":{"order_id":1001}}'),
                         {'goal': intent()['goal']})


def test_discoverable_uncertainty_does_not_force_human_clarification(monkeypatch):
    responses = standard_responses()
    responses[0][1]['goal']['uncertainties'] = ['目录中实际有几款可用候选尚待查询']
    actor, _ = models(monkeypatch, responses)
    result = run_conversation(REQUEST, 'native-discover', runner=run_adaptive)
    assert result['status'] == 'SUCCEEDED', result
    assert len(actor.requests) == 4
    assert result['request_audit']['verdict'] == 'APPROVED'
