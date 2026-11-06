from copy import deepcopy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from src.recovery_evidence import recovery_trace, training_feedback
from src.rrsi_learning import RSISpec
from src.rrsi_policy import HarnessProgram
from eval.run_live_eval import load_dataset

ROOT = Path(__file__).resolve().parents[1]


def sample(passed=False):
    return dict(case_id='R1', repetition=1, passed=passed, model_attempts=2, turns=[dict(
        usage={'total_tokens':50},result={'plan':{'steps':[
            {'step_id':'s1','tool':'search_products','args':{'query':'private'},'status':'SUCCEEDED','result':{'products':[]}},
            {'step_id':'s2','tool':'search_products','args':{'query':'changed'},'status':'SUCCEEDED','result':{'products':[{'product_id':'private-product'}]}},
        ],'objective_ledger':[]},'events':[{'type':'step_completed','step_id':'s1','sequence':1},{'type':'step_completed','step_id':'s2','sequence':2}]})])


def test_observation_improvement_is_not_task_success_or_private_content():
    row = recovery_trace(sample())
    assert row['episodes'][0]['observation_improved'] is True
    assert row['task_passed'] is False
    assert row['episodes'][0]['recovery_actions'][0]['same_arguments'] is False
    assert 'private' not in json.dumps(row)
    left,right = sample(),sample(True)
    right['repetition']=2
    feedback = training_feedback({'samples':[left,right]})
    assert feedback['contrasts'][0]['successful_samples'] == ['R1:2']
    assert feedback['contrasts'][0]['failed_samples'] == ['R1:1']


def test_duplicate_no_progress_and_unfinished_failure_remain_visible():
    item=sample()
    steps=item['turns'][0]['result']['plan']['steps']
    steps[1].update(args=steps[0]['args'],result={'products':[]})
    trace=recovery_trace(item)
    assert trace['repeated_without_new_records']==1
    assert trace['episodes'][-1]['recovery_actions']==[]


def test_recovery_dataset_separates_each_family_across_groups():
    data=load_dataset(ROOT/'eval/rrsi_recovery_tasks.json')
    spec=RSISpec.model_validate_json((ROOT/'research/experiments/rrsi-recovery-v2/evolution-spec.json').read_text())
    spec.validate_dataset(data)
    assert len(data['cases'])==18
    groups={k:{c['scenario']['kind'] for c in data['cases'] if c['id'] in ids}
            for k,ids in [('train',spec.train_cases),('val',spec.validation_cases),('test',spec.test_cases)]}
    assert groups['train']==groups['val']==groups['test']
    assert len(groups['train'])==6


@pytest.mark.parametrize('kind,count,code',[('tool_timeout',2,'TOOL_TIMEOUT'),('repeated_tool_failure',4,'TOOL_ERROR')])
def test_fault_injection_is_bounded_and_does_not_call_real_tool_on_failure(kind,count,code,monkeypatch):
    import src.direct_executor as direct
    from src.runtime import ToolSpec,tool_spec
    from src.errors import AgentError
    from eval.recovery_faults import recovery_fault
    original=direct._READ_ONLY_TOOLS['search_products']
    schema=tool_spec('search_products',original).schema
    calls=[]
    monkeypatch.setitem(direct._READ_ONLY_TOOLS,'search_products',ToolSpec(lambda **kwargs:calls.append(kwargs) or {'products':[]},schema))
    state={}
    with recovery_fault(kind,state):
        fn=direct._READ_ONLY_TOOLS['search_products'].function
        for _ in range(count):
            with pytest.raises(AgentError) as error: fn(query='x')
            assert error.value.code==code
        assert calls==[]
        fn(query='x')
        assert len(calls)==1
    assert state['fired'] and len(state['faults'])==count


def test_invalid_arguments_fault_restores_validator(monkeypatch):
    import src.adaptive as adaptive
    from eval.recovery_faults import recovery_fault
    from types import SimpleNamespace
    from src.errors import AgentError
    fn=lambda *a,**kw:'ok'
    monkeypatch.setattr(adaptive,'_validate_tool',fn)
    with recovery_fault('invalid_arguments',{}):
        with pytest.raises(AgentError): adaptive._validate_tool(SimpleNamespace(tool='search_products'))
        assert adaptive._validate_tool(SimpleNamespace(tool='search_products'))=='ok'
    assert adaptive._validate_tool is fn


@pytest.fixture
def client(tmp_path,monkeypatch):
    import api
    import src.rrsi_api as module
    monkeypatch.setattr(module,'ARTIFACT_ROOT',tmp_path)
    monkeypatch.setenv('LINGCART_ENGINEERING_KEY','rsi-test')
    return TestClient(api.app,headers={'X-Engineering-Key':'rsi-test'}),tmp_path


def test_research_api_requires_auth_and_real_files(client):
    import api
    http,root=client
    assert http.get('/engineering/rrsi/runs').json()=={'runs':[]}
    assert TestClient(api.app).get('/engineering/rrsi/runs').status_code==403
    assert http.get('/engineering/rrsi/runs/missing').status_code==404
    (root/'escape').symlink_to(ROOT/'research/rrsi/planning-smoke-01',target_is_directory=True)
    assert http.get('/engineering/rrsi/runs/escape').status_code==404


def test_research_api_preserves_partial_and_negative_results(client):
    http,root=client
    run=root/'partial';run.mkdir()
    (run/'spec.json').write_text(json.dumps({'target_module':'recovery'}))
    assert http.get('/engineering/rrsi/runs/partial').json()['partial'] is True
    (run/'evolution-report.json').write_text(json.dumps({'status':'COMPLETED','execution_source':'test_harness',
       'history':[{'round':0,'candidate':0,'status':'INVALID_PROPOSAL','parent_sha256':'seed'}]}))
    data=http.get('/engineering/rrsi/runs/partial').json()
    assert data['candidates'][0]['status']=='INVALID_PROPOSAL'
    assert data['candidates'][0]['samples']==[]
    assert data['comparison'] is None
    (run/'seed-validation.json').write_text('invalid json')
    data=http.get('/engineering/rrsi/runs/partial').json()
    assert data['partial'] and data['warnings']


def test_history_cannot_select_arbitrary_paths(client):
    http,root=client
    run=root/'hostile';run.mkdir()
    (run/'spec.json').write_text('{}')
    (run/'evolution-report.json').write_text(json.dumps({'history':[{'round':'../../secret','candidate':0,'status':'ACCEPTED','parent_sha256':'seed'}]}))
    data=http.get('/engineering/rrsi/runs/hostile').json()
    assert data['candidates']==[] and data['partial']


def test_api_marks_identical_arm_programs(client):
    http,root=client
    run=root/'same-program';run.mkdir()
    (run/'spec.json').write_text('{}')
    (run/'ablation-summary.json').write_text(json.dumps({'status':'COMPLETED','observed_improvement':False}))
    (run/'ablation-contract.json').write_text(json.dumps({'arms':{'fixed':'seed','evolved':'seed','baseline':'adaptive-v1'}}))
    data=http.get('/engineering/rrsi/runs/same-program').json()
    assert data['comparison']['identical_arms']==[['fixed','evolved']]
