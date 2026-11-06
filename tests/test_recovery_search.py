from copy import deepcopy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from eval.run_recovery_replay import ReplaySpec, PROFILES, main, summary
from src.recovery_search import ProbeAction, RecoverySearchMechanism, binding
from src.runtime import RunRuntime
from src.task_plan import TaskPlan


def usage_events():
    return [dict(type='model_started'),dict(type='model_call',usage=dict(input_tokens=30,output_tokens=20,total_tokens=50))]*2


def test_search_protocol_and_dimensions_are_closed():
    for configuration in ({'max_depth':3},{'max_searches':True},{'max_branches':4},{'permission':'write'}):
        with pytest.raises(ValueError): RecoverySearchMechanism().configure(configuration)
    with pytest.raises(ValidationError):
        ProbeAction.model_validate({'action':'tool','tool':'search_products','goal':{'summary':'new'}})
    with pytest.raises(ValidationError):
        ProbeAction.model_validate({'action':'recommend'})


def test_search_selection_is_invalidated_by_goal_or_constraint_revision():
    policy=RecoverySearchMechanism(); plan=TaskPlan([],[])
    state=dict(request='原始任务',goal={'summary':'原始任务'},goal_revision=0,binding_revision=0)
    attempt=dict(binding=binding(state,plan),status='SELECTED',search_id='test',
        branches=[dict(branch_id='b0',actions=[dict(action='tool',tool='search_products',args={})])],
        selection=dict(branch_id='b0',reason='读取目录'))
    state['recovery_searches']=[attempt]
    assert policy.choose(RunRuntime('choice'),state,plan).tool=='search_products'
    for field in ('goal_revision','binding_revision'):
        changed=deepcopy(state);changed[field]=1
        assert policy.choose(RunRuntime('choice'),changed,plan) is None


def test_replay_specs_validate_without_model_calls(capsys):
    assert main([])==0
    assert json.loads(capsys.readouterr().out)['max_task_runs']==7
    spec=ReplaySpec.model_validate_json(Path('research/experiments/recovery-search-v1/smoke-spec.json').read_text())
    dataset=json.loads(Path(spec.dataset).read_text())
    assert len(spec.cases(dataset))==1
    wrong=spec.model_copy(update={'split':'test'})
    with pytest.raises(ValueError):wrong.cases(dataset)


def test_replay_summary_rejects_missing_unknown_cost_and_unpaired_samples():
    def row(profile):
        return dict(profile_id=profile,origin_sha256='origin',database_sha256='db',prefix_budget={'model_calls':3},
            repetition=1,passed=True,integrity=True,changed_db_paths=[],suffix_usage={'total_tokens':100},
            suffix_budget={'model_calls':2,'tool_calls':2},suffix_steps=2,result={'events':usage_events()})
    rows=[row(p) for p in PROFILES]
    assert summary(rows,expected_points=1,repeats=1,captures_complete=True)['status']=='COMPLETED'
    assert summary(rows[:-1],expected_points=1,repeats=1,captures_complete=True)['status']=='INVALID'
    rows[-1]['result']['events'][-1]['usage']={}
    assert summary(rows,expected_points=1,repeats=1,captures_complete=True)['status']=='INVALID'
    rows[-1]['result']['events']=usage_events();rows[-1]['database_sha256']='changed'
    assert summary(rows,expected_points=1,repeats=1,captures_complete=True)['status']=='INVALID'


def test_replay_projection_auth_partial_and_path_boundaries(tmp_path,monkeypatch):
    import api
    import src.recovery_search_api as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setenv('LINGCART_ENGINEERING_KEY','replay-test')
    client=TestClient(api.app,headers={'X-Engineering-Key':'replay-test'})
    assert TestClient(api.app).get('/engineering/recovery-replays').status_code==403
    assert client.get('/engineering/recovery-replays').json()=={'runs':[]}
    folder=tmp_path/'partial';folder.mkdir()
    (folder/'contract.json').write_text(json.dumps({'execution_source':'test_harness','manifest':{'source_revision':'abc'}}))
    (folder/'sample-broken.json').write_text('{')
    data=client.get('/engineering/recovery-replays/partial').json()
    assert data['partial'] and data['samples']==[] and data['warnings']
    (tmp_path/'escape').symlink_to(folder,target_is_directory=True)
    assert client.get('/engineering/recovery-replays/escape').status_code==404


def test_replay_projection_excludes_database_model_inputs_and_probe_arguments():
    from src.recovery_search_api import project_sample
    raw=dict(profile_id='search',case_id='R1',repetition=1,origin_sha256='same',database_sha256='same-db',
        passed=False,failures=['not completed'],suffix_steps=1,suffix_budget={'model_calls':2,'tool_calls':3},
        suffix_usage={'total_tokens':100},database={'secret':'do-not-expose'},
        result={'terminal_reason':'MODEL_STOPPED','events':[
            {'type':'recovery_search_probe','search_id':'s1','branch_id':'b0','probe_id':'probe-b0-0',
             'status':'SUCCEEDED','step':{'tool':'search_products','args':{'query':'private-query'},'result':{'products':[{'product_id':'private-product'}]}}},
            {'type':'recovery_search_completed','search_id':'s1','status':'SELECTED','selection':{'branch_id':'b0','reason':'实际返回候选'}},
        ]},recovery={'timeline':[{'step_id':'prefix'},{'step_id':'suffix'}]})
    result=project_sample(raw).model_dump()
    assert result['searches'][0]['chosen_branch']=='b0'
    assert result['searches'][0]['probes'][0]['product_records']==1
    assert result['timeline']==[{'step_id':'suffix'}]
    assert 'private-' not in json.dumps(result) and 'do-not-expose' not in json.dumps(result)


def test_strict_probe_schema_bounds_tool_arguments_before_execution():
    from src.recovery_search_v2 import proposal_schema
    schema = proposal_schema(2, 1)
    valid = dict(branches=[
        dict(expected_observation='检索候选', actions=[dict(action='tool', tool='search_products', args={'query':'耳机'})]),
        dict(expected_observation='读取证据', actions=[dict(action='inspect', evidence_ref='step-1')]),
    ])
    assert schema.model_validate(valid).branches[0].actions[0].args.query == '耳机'
    for change in (
        {'tool':'process_payment', 'args':{'order_id':1003}},
        {'tool':'search_products', 'args':{'query':'耳机','invented':True}},
        {'tool':'search_products', 'args':{'query':'耳机','limit':'2'}},
        {'evidence_ref':'step-1'},
    ):
        bad=deepcopy(valid);bad['branches'][0]['actions'][0].update(change)
        with pytest.raises(ValidationError): schema.model_validate(bad)
    with pytest.raises(ValidationError): schema.model_validate({'branches':valid['branches'][:1]})
    deep=deepcopy(valid);deep['branches'][0]['actions'] *= 2
    with pytest.raises(ValidationError): schema.model_validate(deep)


def test_gate_counts_distinct_failures_and_resets_only_on_real_progress():
    from src.recovery_search_v2 import GatedRecoverySearchMechanism
    from src.task_plan import TaskStep
    policy=GatedRecoverySearchMechanism()
    plan=TaskPlan([],[])
    state=dict(request='原目标',goal={'summary':'原目标'},goal_revision=0,binding_revision=0,
               model_cursor=0,feedback={'code':'TOOL_TIMEOUT'})
    runtime=RunRuntime('gate')
    assert not policy._admit(runtime,state,plan)
    assert not policy._admit(runtime,state,plan)
    assert len(state['recovery_search_gates']) == 1
    # 新一次无效决策可以没有新增 step，也没有增加 repair_count。
    runtime.model_cursor=1;state['model_cursor']=1
    assert policy._admit(runtime,state,plan)
    assert state['recovery_search_gates'][-1]['failure_count']==2
    plan.steps.append(TaskStep('s1','shopping','search_products',{},status='SUCCEEDED',result={'products':[]}))
    assert policy._admit(runtime,state,plan)
    assert state['recovery_search_gates'][-1]['failure_count']==3
    plan.steps.append(TaskStep('s2','shopping','search_products',{},status='SUCCEEDED',result={'products':[{'product_id':'P1001'}]}))
    assert not policy._admit(runtime,state,plan)
    repeated=deepcopy(plan.steps[-1]);repeated.step_id='s3'
    repeated.result['observation']={'timestamp':'later','reference':'new'}
    plan.steps.append(repeated)
    assert policy._admit(runtime,state,plan)
    assert state['recovery_search_gates'][-1]['failure_count']==2
    state['goal_revision']=1
    assert not policy._admit(runtime,state,plan)
    assert len(runtime.events)==6


def test_gate_checkpoint_republishes_without_counting_twice():
    from src.recovery_search_v2 import GatedRecoverySearchMechanism
    from src.runtime_state import new_runtime_snapshot, seal_runtime_snapshot
    from src.errors import AgentError
    policy=GatedRecoverySearchMechanism();plan=TaskPlan([],[])
    state=dict(request='原目标',goal={},feedback={'code':'TOOL_TIMEOUT'},model_cursor=0)
    runtime=RunRuntime('gate-resume',snapshot=new_runtime_snapshot('gate-resume',conversation={},history=[]))
    policy._admit(runtime,state,plan)
    restored=RunRuntime('gate-resume',snapshot=seal_runtime_snapshot(runtime.snapshot,'gate-resume'))
    resumed=deepcopy(runtime.snapshot['adaptive'])
    assert not policy._admit(restored,resumed,plan)
    assert len(resumed['recovery_search_gates'])==1
    with pytest.raises(ValueError): policy.configure({'failure_threshold':True})
    with pytest.raises(ValueError): policy.configure({'failure_threshold':4})
    with pytest.raises(AgentError,match='配置'):
        policy.configure({'failure_threshold':1})._admit(restored,resumed,plan)


def test_v2_replay_spec_freezes_arms_and_summary_rejects_reference_drift(capsys):
    path='research/experiments/recovery-search-v2/smoke-spec.json'
    assert main(['--spec',path])==0
    assert json.loads(capsys.readouterr().out)['max_task_runs']==14
    spec=ReplaySpec.model_validate_json(Path(path).read_text())
    assert spec.profiles==['adaptive-recovery-control-v1','adaptive-recovery-eager-v2','adaptive-recovery-search-v2']
    with pytest.raises(ValidationError): ReplaySpec.model_validate({**spec.model_dump(),'profiles':['adaptive-v1']*3})
    rows=[dict(profile_id=p,origin_sha256='point',database_sha256='db',prefix_budget={'model_calls':3},
        repetition=1,passed=True,integrity=True,changed_db_paths=[],suffix_usage={'total_tokens':100},
        suffix_budget={'model_calls':2,'tool_calls':2},suffix_steps=2,result={'events':usage_events()}) for p in spec.profiles]
    result=summary(rows,expected_points=1,repeats=1,captures_complete=True,profiles=spec.profiles)
    assert result['status']=='COMPLETED' and result['comparison']['control_profile']=='adaptive-recovery-eager-v2'
    rows[0]['prefix_budget']['model_calls']=4
    assert summary(rows,expected_points=1,repeats=1,captures_complete=True,profiles=spec.profiles)['status']=='INVALID'


def test_replay_api_reads_contract_profiles_and_gate_evidence(tmp_path,monkeypatch):
    import api
    import src.recovery_search_api as module
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setenv('LINGCART_ENGINEERING_KEY','replay-test')
    client=TestClient(api.app,headers={'X-Engineering-Key':'replay-test'})
    folder=tmp_path/'v2';folder.mkdir()
    contract={'profiles':[{'profile_id':p} for p in ['adaptive-recovery-control-v1','adaptive-recovery-eager-v2','adaptive-recovery-search-v2']],
              'profile_labels':{'adaptive-recovery-search-v2':'按需恢复搜索'}}
    (folder/'contract.json').write_text(json.dumps(contract))
    gate=dict(type='recovery_search_gate',gate_id='g1',status='DEFERRED',code='ORDINARY_RECOVERY_FIRST',failure_count=1,threshold=2)
    sample=dict(profile_id='adaptive-recovery-search-v2',case_id='R1',repetition=1,origin_sha256='same',database_sha256='same-db',
        passed=True,failures=[],suffix_steps=1,suffix_budget={'model_calls':2,'tool_calls':3},suffix_usage={'total_tokens':100},
        result={'events':[gate]})
    (folder/'sample-one.json').write_text(json.dumps(sample))
    data=client.get('/engineering/recovery-replays/v2').json()
    assert data['partial'] and data['profiles'][2]['display_name']=='按需恢复搜索'
    assert data['profiles'][1]['role']=='control'
    assert data['samples'][0]['gates'][0]['status']=='DEFERRED' and not data['samples'][0]['searches']
    (folder/'contract.json').write_text(json.dumps({'profiles':'broken'}))
    data=client.get('/engineering/recovery-replays/v2').json()
    assert data['partial'] and data['warnings'] and data['profiles'][0]['role']=='unknown'


def test_partial_provider_usage_is_not_a_complete_replay_cost():
    from src.recovery_usage import recovery_usage
    from src.recovery_search_api import project_sample
    events=usage_events()+[dict(type='model_started'),dict(type='error',source='model',code='MODEL_ERROR')]
    raw=dict(profile_id='search',case_id='R1',repetition=1,origin_sha256='origin',database_sha256='db',
        passed=False,failures=['model failed'],suffix_steps=0,integrity=True,changed_db_paths=[],prefix_budget={},
        suffix_budget={'model_calls':3,'tool_calls':0},suffix_usage={'total_tokens':100},
        result={'events':events})
    usage=recovery_usage(raw)
    assert not usage['complete'] and usage['usage'] is None and usage['known_calls']==2
    assert usage['observed_usage']['total_tokens']==100
    projected=project_sample(raw)
    assert projected.tokens is None and projected.known_tokens==100 and not projected.usage_complete
    # 即使旧产物有正数的总量字段，也不能通过完整成本 Gate。
    rows=[dict(raw,profile_id=p) for p in PROFILES]
    assert summary(rows,expected_points=1,repeats=1,captures_complete=True)['status']=='INVALID'
    raw['suffix_budget']['model_calls']=2;raw['result']['events']=usage_events()
    assert recovery_usage(raw)['complete']
    raw['result']['events'][0]['type']='model_reused'
    assert not recovery_usage(raw)['complete']
