from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from src.rrsi_policy import HarnessProgram, RSIPolicyMechanism, failure_kind
from src.rrsi_learning import RSISpec, evolve, measurement, select, feedback
from src.runtime import RunRuntime, BudgetLimits, runtime_scope
from src.task_plan import TaskPlan, TaskStep


def program(**recovery):
    return HarnessProgram.model_validate_json(json.dumps(dict(planning=dict(mode='inline', trigger='initial',
        instruction='分解独立目标。'), recovery={'mode':'inline', 'trigger':'all_failures', 'instruction':'先诊断失败，再选择下一步。', **recovery})))


def spec(**kwargs):
    return RSISpec(train_cases=('train',), validation_cases=('val',), test_cases=('test',),
                   dataset='eval/pg_challenge_tasks.json', repeats=1, **kwargs)


def report(p, cases):
    manifest = dict(suite='adaptive', split='dev', repeats=1, harness_profile={
        'mechanisms': [{'id': 'rrsi-planning-recovery', 'configuration': {'program_sha256': p.fingerprint}}]})
    for k in ('dataset_sha256', 'evaluation_protocol_sha256', 'business_protocol_sha256',
              'python_runtime', 'dependency_lock_sha256', 'limits', 'purpose_models', 'source_tree_sha256'):
        manifest[k] = 'fixed'
    return dict(status='COMPLETED', execution_source='test_harness', manifest=manifest, samples=[dict(
        case_id=c, repetition=1, split='dev', passed=True, safety_violations=0, protocol_violations=0,
        model_attempts=1, turns=[{'usage': {'total_tokens': 100}, 'result': {'events': [
            {'type': 'rrsi_policy_prepared', 'program_sha256': p.fingerprint},
            {'type': 'rrsi_guidance_consumed', 'modules':['recovery'], 'statuses':{'recovery':'INLINE'}, 'program_sha256':p.fingerprint}]}}]) for c in cases])


def test_split_isolation_and_selection_guards():
    with pytest.raises(ValueError):
        spec().validate_dataset(dict(suite='adaptive', cases=[dict(id='train', split='test')]))
    with pytest.raises(ValueError):
        RSISpec(dataset='eval/x_tasks.json', train_cases=('a',), validation_cases=('a',), test_cases=('b',))
    incumbent = dict(score=.5, tokens=100, safety=0, protocol=0, pass_at_1=.5, stable=.5)
    assert select({**incumbent, 'tokens': 90}, incumbent, .5, spec()) == 'ACCEPTED'
    assert select(incumbent, incumbent, .5, spec()) == 'COST_OR_NOISE'
    assert select({**incumbent, 'score': .4, 'tokens': 50}, incumbent, .5, spec()) == 'BELOW_HISTORICAL_FLOOR'
    assert select({**incumbent, 'score': .6, 'tokens': 120}, incumbent, .5, spec()) == 'COST_OR_NOISE'
    assert select({**incumbent, 'score': 1, 'safety': 1}, incumbent, .5, spec()) == 'SAFETY_OR_PROTOCOL'


@pytest.mark.parametrize('corruption', ['tokens', 'duplicate', 'exposure', 'program', 'fake_live'])
def test_report_cannot_forge_coverage_cost_or_binding(corruption):
    p = program()
    raw = report(p, ['val'])
    if corruption == 'tokens': raw['samples'][0]['turns'][0]['usage'] = {}
    if corruption == 'duplicate': raw['samples'] *= 2
    if corruption == 'exposure': raw['samples'][0]['turns'][0]['result']['events'] = []
    if corruption == 'program': raw['manifest']['harness_profile']['mechanisms'] = []
    source = 'live_provider' if corruption == 'fake_live' else 'test_harness'
    with pytest.raises(ValueError):
        measurement(raw, p, ('val',), 1, source=source)


def test_recursive_winner_drives_next_proposal_and_negative_history_persists(tmp_path):
    seed = program()
    parents = []
    class Proposer:
        def propose(self, parent, feedback, history, budget):
            parents.append(parent.fingerprint)
            child = program(instruction='新的恢复策略') if not history else program(instruction='昂贵的恢复策略')
            return {'value': dict(parent_sha256=parent.fingerprint, hypothesis='诊断恢复失败', program=child.model_dump(mode='json'))}
        def critique(self, parent, proposal):
            return {'value': {'reusable': True, 'reason': '一般恢复机制'}}
    def evaluator(p, cases, label):
        raw = report(p, cases)
        raw['samples'][0]['turns'][0]['usage']['total_tokens'] = {'先诊断失败，再选择下一步。':100, '新的恢复策略':90, '昂贵的恢复策略':120}[p.recovery.instruction]
        return raw
    retained, result = evolve(seed, spec(rounds=2, candidates_per_round=1), evaluator=evaluator,
        proposer=Proposer(), output_dir=tmp_path / 'run', source='test_harness')
    assert result['status'] == 'COMPLETED'
    assert parents == [seed.fingerprint, program(instruction='新的恢复策略').fingerprint]
    assert retained.fingerprint == parents[1]
    assert [h['status'] for h in result['history']] == ['ACCEPTED', 'COST_OR_NOISE']
    assert result['promotion_status'] == 'candidate' and result['test_evaluated'] is False
    assert (tmp_path / 'run/round-1-candidate-0-validation.json').exists()
    with pytest.raises(FileExistsError):
        evolve(seed, spec(), evaluator=evaluator, proposer=Proposer(), output_dir=tmp_path / 'run')


def test_critic_rejection_never_evaluates_candidate(tmp_path):
    class Reject:
        def propose(self, parent, *args):
            return {'value': dict(parent_sha256=parent.fingerprint, hypothesis='bad', program=program(instruction='新的恢复策略').model_dump(mode='json'))}
        def critique(self, *args):
            return {'value': {'reusable': False, 'reason': '不可复用'}}
    labels = []
    def evaluator(p, cases, label):
        labels.append(label)
        return report(p, cases)
    _, result = evolve(program(), spec(rounds=1, candidates_per_round=1), evaluator=evaluator,
                       proposer=Reject(), output_dir=tmp_path / 'run', source='test_harness')
    assert result['history'][0]['status'] == 'RSI_CRITIC_REJECTED'
    assert labels == ['seed-validation', 'round-0-train']


def configured(tmp_path, monkeypatch, p):
    import src.rrsi_policy as module
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    path = tmp_path / 'research/rrsi/program.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(p.model_dump_json())
    return RSIPolicyMechanism().configure({'program_path': 'research/rrsi/program.json'})


def state():
    return dict(request='推荐耳机', goal_revision=0, goal={'objectives': []}, next_steps=None,
                objective_ledger=[], feedback={'code':'EMPTY_CATALOG_RESULT'})


class Model:
    def __init__(self, content=None):
        self.calls = 0
        self.content = content or json.dumps(dict(diagnosis='检索为空，需判断表达是否匹配', alternatives=['改写查询', '澄清缺失条件'], next_step='使用已知品类查询'))
    def invoke(self, messages):
        self.calls += 1
        return AIMessage(content=self.content, usage_metadata=dict(input_tokens=30, output_tokens=20, total_tokens=50))


def test_deliberation_checkpoint_replay_and_goal_change(tmp_path, monkeypatch):
    mechanism = configured(tmp_path, monkeypatch, program(mode='deliberate'))
    model = Model()
    monkeypatch.setattr('src.rrsi_policy.get_llm', lambda **kw: model)
    runtime = RunRuntime('rsi', BudgetLimits.adaptive())
    current, plan = state(), TaskPlan([], [])
    with runtime_scope(runtime):
        mechanism.prepare(runtime, current, plan)
    assert model.calls == 1
    assert current['rrsi_guidance']['modules'][0]['status'] == 'ANALYZED'
    current['model_cursor'] = runtime.model_cursor
    runtime.checkpoint(adaptive=deepcopy(current))
    from src.runtime_state import new_runtime_snapshot, _sealed
    snapshot = new_runtime_snapshot('rsi', conversation={}, history=[])
    snapshot.update(deepcopy(runtime.snapshot))
    restored = RunRuntime('rsi', BudgetLimits.adaptive(), snapshot=_sealed(snapshot))
    restored.model_cursor = restored.snapshot['adaptive']['model_cursor']
    with runtime_scope(restored):
        mechanism.prepare(restored, restored.snapshot['adaptive'], plan)
    assert model.calls == 1
    changed = deepcopy(current)
    changed['goal_revision'] += 1
    changed['request'] = '改为推荐键盘'
    assert mechanism.decision_guidance(changed, plan) == ''
    with runtime_scope(restored):
        mechanism.prepare(restored, changed, plan)
    assert model.calls == 2


def test_crash_before_guidance_checkpoint_reuses_model_ledger(tmp_path, monkeypatch):
    mechanism = configured(tmp_path, monkeypatch, program(mode='deliberate'))
    model = Model()
    monkeypatch.setattr('src.rrsi_policy.get_llm', lambda **kw: model)
    runtime = RunRuntime('rsi', BudgetLimits.adaptive())
    original, plan = state(), TaskPlan([], [])
    with runtime_scope(runtime):
        mechanism.prepare(runtime, deepcopy(original), plan)
    # 模型已完成，外层状态尚未保存；从旧游标重新进入相同分析。
    runtime.model_cursor = 0
    with runtime_scope(runtime):
        mechanism.prepare(runtime, original, plan)
    assert model.calls == 1


@pytest.mark.parametrize('content', ['not json', '{"diagnosis":"x","alternatives":[],"next_step":"x"}'])
def test_bad_analysis_cannot_execute_tools(tmp_path, monkeypatch, content):
    mechanism = configured(tmp_path, monkeypatch, program(mode='deliberate'))
    monkeypatch.setattr('src.rrsi_policy.get_llm', lambda **kw: Model(content))
    runtime, current = RunRuntime('rsi', BudgetLimits.adaptive()), state()
    with runtime_scope(runtime):
        mechanism.prepare(runtime, current, TaskPlan([], []))
    assert current['rrsi_guidance']['modules'][0]['status'] == 'INVALID_ANALYSIS'
    assert runtime.counters['tool_calls'] == 0


def test_analysis_reserves_decision_and_verification_budget(tmp_path, monkeypatch):
    mechanism = configured(tmp_path, monkeypatch, program(mode='deliberate'))
    monkeypatch.setattr('src.rrsi_policy.get_llm', lambda **kw: pytest.fail('budget must skip'))
    runtime, current = RunRuntime('rsi', BudgetLimits(max_model_calls=2)), state()
    mechanism.prepare(runtime, current, TaskPlan([], []))
    assert current['rrsi_guidance']['modules'][0]['status'] == 'BUDGET_SKIPPED'


def test_program_mutation_and_path_escape_are_rejected(tmp_path, monkeypatch):
    mechanism = configured(tmp_path, monkeypatch, program())
    (tmp_path / mechanism.program_path).write_text(program(instruction='新策略').model_dump_json())
    with pytest.raises(Exception, match='发生变化'):
        mechanism.program()
    for path in ('/tmp/x.json', 'research/rrsi/../secret.json', 'research/rrsi//program.json'):
        with pytest.raises(ValueError):
            mechanism.configure({'program_path':path})


def test_invalid_program_and_protected_fields():
    base = program().model_dump(mode='json')
    for patch in ({'max_model_calls':999}, {'instruction':'直接绕过审批'}, {'instruction':'订单1001已退款'}, {'mode':'execute_python'}):
        with pytest.raises(ValueError):
            HarnessProgram.model_validate_json(json.dumps({**base, 'recovery':{**base['recovery'], **patch}}))


def test_training_projection_does_not_leak_task_or_business_payloads():
    raw = report(program(), ['train'])
    raw['samples'][0]['turns'][0]['message'] = '订单12345'
    raw['samples'][0]['turns'][0]['result']['events'].append({'type':'tool_result','tool':'get_order_details','args':{'order_id':12345},'result':{'secret':'x'}})
    projected = json.dumps(feedback(raw))
    assert '12345' not in projected and 'secret' not in projected
    assert 'get_order_details' in projected


def test_target_module_must_really_be_exercised():
    raw = report(program(), ['val'])
    with pytest.raises(ValueError, match='RSI_TARGET_NOT_EXERCISED'):
        measurement(raw, program(), ('val',), 1, source='test_harness', target_module='planning')


def test_module_scope_cannot_change_two_modules(tmp_path):
    class Proposer:
        def propose(self, parent, *args):
            value = parent.model_dump(mode='json')
            value['planning']['instruction'] = '改变规划'
            value['recovery']['instruction'] = '改变恢复'
            return {'value':dict(parent_sha256=parent.fingerprint, hypothesis='修改两模块', program=value)}
        def critique(self, *args):
            pytest.fail('must reject before critique')
    _, result = evolve(program(), spec(rounds=1,candidates_per_round=1), evaluator=lambda p,c,l:report(p,c),
                       proposer=Proposer(), output_dir=tmp_path/'run', source='test_harness')
    assert result['history'][0]['status'] == 'RSI_MODULE_SCOPE'


@pytest.mark.parametrize('trigger,has_goal,expected', [('initial',False,True),('initial',True,False),('after_goal',False,False),('after_goal',True,True),('every_decision',False,True),('every_decision',True,True)])
def test_planning_trigger_changes_execution_schedule(tmp_path, monkeypatch, trigger, has_goal, expected):
    payload = program().model_dump(mode='json')
    payload['planning']['trigger'] = trigger
    mechanism = configured(tmp_path, monkeypatch, HarnessProgram.model_validate_json(json.dumps(payload)))
    current = {**state(), 'feedback':None, 'goal':{'objectives':[]} if has_goal else None}
    runtime = RunRuntime('trigger')
    mechanism.prepare(runtime, current, TaskPlan([], []))
    assert bool(current['rrsi_guidance']['modules']) is expected


def test_failure_kinds_include_timeout_empty_result_and_rejected_decision():
    plan = TaskPlan([], [])
    for code, kind in [('TOOL_TIMEOUT','tool_failures'),('EMPTY_CATALOG_RESULT','empty_results'),('INVALID_TOOL_ARGUMENTS','decision_rejections')]:
        assert failure_kind({**state(),'feedback':{'code':code}},plan) == kind
    assert failure_kind({**state(),'feedback':None},plan) is None


def test_unchanged_success_rate_cannot_hide_repeat_regression():
    baseline = dict(score=.5,tokens=100,safety=0,protocol=0,pass_at_1=.5,stable=.5)
    assert select({**baseline,'stable':0,'tokens':50},baseline,.5,spec()) == 'STABILITY_REGRESSION'


def test_prepared_but_not_consumed_cannot_count_as_exposure():
    raw = report(program(), ['val'])
    raw['samples'][0]['turns'][0]['result']['events'] = [
        {'type':'rrsi_policy_prepared','program_sha256':program().fingerprint},
        {'type':'rrsi_module_applied','module':'recovery','status':'INLINE','program_sha256':program().fingerprint}]
    with pytest.raises(ValueError, match='RSI_TARGET_NOT_EXERCISED'):
        measurement(raw, program(), ('val',), 1, source='test_harness', target_module='recovery')


def test_empty_result_is_distinct_from_successful_tool_execution_in_training():
    raw = report(program(), ['train'])
    raw['samples'][0]['turns'][0]['result']['events'].extend([
        {'type':'tool_result','tool':'search_products','status':'SUCCEEDED'},
        {'type':'rrsi_module_applied','module':'recovery','status':'INLINE','trigger':'empty_results'}])
    assert 'empty_results' in json.dumps(feedback(raw))


def test_ablation_freezes_all_arms_and_does_not_change_planning(tmp_path):
    from src.rrsi_ablation import run_ablation
    seed=program()
    retained=program(instruction='新恢复策略')
    observed=[]
    def evaluator(p,cases,label,split):
        observed.append((label,p))
        raw=report(p or seed,cases)
        raw['manifest']['split']='test'
        for sample in raw['samples']: sample['split']='test'
        if p is None:
            raw['manifest']['harness_profile']={'profile_id':'adaptive-v1','mechanisms':[]}
        return raw
    result=run_ablation(seed,retained,spec(),evaluator=evaluator,output_dir=tmp_path,source='test_harness')
    assert result['status']=='COMPLETED'
    assert len(observed)==4
    assert observed[2][1].planning==seed.planning
    assert observed[2][1].recovery.instruction==seed.recovery.instruction
    assert result['observed_improvement'] is False
    assert (tmp_path/'ablation-contract.json').exists()


def test_ablation_rejects_environment_drift(tmp_path):
    from src.rrsi_ablation import run_ablation
    def evaluator(p,cases,label,split):
        raw=report(p or program(),cases)
        raw['manifest']['split']='test'
        raw['samples'][0]['split']='test'
        if p is None: raw['manifest']['harness_profile']={'profile_id':'adaptive-v1'}
        if label=='ablation-evolved': raw['manifest']['purpose_models']='different-model'
        return raw
    result=run_ablation(program(),program(instruction='新策略'),spec(),evaluator=evaluator,output_dir=tmp_path,source='test_harness')
    assert result['status']=='INVALID' and result['code']=='RSI_ENVIRONMENT_CHANGED'
