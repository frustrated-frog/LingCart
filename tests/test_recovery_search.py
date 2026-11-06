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
            suffix_budget={'model_calls':2,'tool_calls':2},suffix_steps=2)
    rows=[row(p) for p in PROFILES]
    assert summary(rows,expected_points=1,repeats=1,captures_complete=True)['status']=='COMPLETED'
    assert summary(rows[:-1],expected_points=1,repeats=1,captures_complete=True)['status']=='INVALID'
    rows[-1]['suffix_usage']=None
    assert summary(rows,expected_points=1,repeats=1,captures_complete=True)['status']=='INVALID'
    rows[-1]['suffix_usage']={'total_tokens':100};rows[-1]['database_sha256']='changed'
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
