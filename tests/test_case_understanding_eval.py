from copy import deepcopy
from eval.run_case_understanding_eval import check_sample, DATASET
from eval.run_live_eval import load_dataset
from eval.experiment_contract import ExperimentSpec
from pathlib import Path


def test_frozen_datasets_and_two_independent_contracts():
    dataset = load_dataset(DATASET)
    assert len([c for c in dataset['cases'] if c['split'] == 'test']) == 3
    for name in ('after-sales-source-refs-v1','after-sales-control-intent-v1'):
        spec = ExperimentSpec.from_path(Path('research/experiments') / name / 'experiment-spec.json')
        assert spec.dataset == 'eval/after_sales_understanding_tasks.json'


def test_judge_requires_current_committed_evidence_and_fresh_confirmation():
    case = {'turns':[{'expected_material_types':['SERIAL_LABEL']},{}],
            'expected_final_status':'WAITING_APPROVAL','check_fresh_approval':True}
    turn = {'failures':[], 'passed':True,'observed_tools':[],
            'result':{'run_id':'r','status':'WAITING_APPROVAL','pending_action':{'action_id':'old'}}}
    sample = {'turns':[deepcopy(turn),deepcopy(turn)]}
    check_sample(sample,case)
    assert not sample['passed']
    assert len(sample['turns'][0]['failures']) == len(sample['turns'][1]['failures']) == 1
    sample = {'turns':[deepcopy(turn),deepcopy(turn)]}
    sample['turns'][0]['observed_tools'] = [{'result':{'case':{'evidence':[{
        'evidence_type':'SERIAL_LABEL','source_ref':'agent_runs:r:request','status':'ACTIVE'}]}}}]
    sample['turns'][1]['result']['pending_action']['action_id'] = 'new'
    check_sample(sample,case)
    assert sample['passed']
