import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval.experiment_contract import ExperimentSpec


ROOT = Path(__file__).resolve().parents[1]


def test_template_and_first_experiment_are_machine_validated():
    template = ExperimentSpec.from_path(ROOT / 'research/templates/experiment-spec.json')
    experiment = ExperimentSpec.from_path(
        ROOT / 'research/experiments/objective-context-v1/experiment-spec.json')

    assert template.tuning_split == 'dev' and template.evaluation_split == 'test'
    assert experiment.dataset == 'eval/adaptive_tasks.json'
    assert experiment.candidate_profile == 'adaptive-objective-context-v1'


def test_contract_rejects_missing_guard_and_unreviewed_runtime_change():
    payload = json.loads((ROOT / 'research/templates/experiment-spec.json').read_text())
    payload['guard_metrics'] = payload['guard_metrics'][1:]
    with pytest.raises(ValidationError, match='保护指标'):
        ExperimentSpec.model_validate_json(json.dumps(payload))

    payload = json.loads((ROOT / 'research/templates/experiment-spec.json').read_text())
    payload['changes'] = [{'kind': 'runtime', 'slot': None, 'implementation_id': None,
                           'paths': ['src/runtime.py']}]
    with pytest.raises(ValidationError, match='兼容性审查'):
        ExperimentSpec.model_validate_json(json.dumps(payload))


def test_contract_rejects_reversed_guards_and_requires_a_cost_metric():
    payload = json.loads((ROOT / 'research/templates/experiment-spec.json').read_text())
    payload['guard_metrics'][0]['objective'] = 'minimize'
    with pytest.raises(ValidationError, match='保护指标的方向'):
        ExperimentSpec.model_validate_json(json.dumps(payload))

    payload = json.loads((ROOT / 'research/templates/experiment-spec.json').read_text())
    payload['primary_metrics'] = [{"name": "mean_success_rate", "objective": "maximize",
                                   "max_regression": 0, "min_improvement": 0}]
    payload['guard_metrics'] = [rule for rule in payload['guard_metrics']
                                if rule['name'] != 'latency_ms.p95']
    with pytest.raises(ValidationError, match='成本或延迟'):
        ExperimentSpec.model_validate_json(json.dumps(payload))
