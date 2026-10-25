import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval.experiment_contract import ExperimentSpec, MetricRule


ROOT = Path(__file__).resolve().parents[1]


def test_template_and_first_experiment_are_machine_validated():
    template = ExperimentSpec.from_path(ROOT / 'research/templates/experiment-spec.json')
    experiment = ExperimentSpec.from_path(
        ROOT / 'research/experiments/objective-context-v1/experiment-spec.json')

    assert template.tuning_split == 'dev' and template.evaluation_split == 'test'
    assert experiment.dataset == 'eval/adaptive_tasks.json'
    assert experiment.candidate_profile == 'adaptive-objective-context-v1'
    assert all(rule.minimum_value is None and rule.maximum_value is None
               for rule in (*template.primary_metrics, *template.guard_metrics))


def test_metric_rule_accepts_absolute_range_and_rejects_inversion():
    rule = MetricRule(
        name='policy_autonomy.case_resolution_rate',
        objective='maximize',
        max_regression=0,
        min_improvement=0.5,
        minimum_value=1.0,
        maximum_value=1.0,
    )

    assert rule.minimum_value == rule.maximum_value == 1.0
    with pytest.raises(ValidationError, match='绝对下限'):
        MetricRule(
            name='mean_success_rate',
            objective='maximize',
            minimum_value=1.0,
            maximum_value=0.75,
        )


def test_liveplan_experiment_freezes_one_steering_mechanism_and_cost_guard():
    experiment = ExperimentSpec.from_path(
        ROOT / 'research/experiments/liveplan-v1/experiment-spec.json')

    assert experiment.baseline_profile == 'adaptive-v1'
    assert experiment.candidate_profile == 'adaptive-liveplan-v1'
    assert [(change.kind, change.slot, change.implementation_id) for change in experiment.changes] == [
        ('mechanism', 'steering', 'liveplan-online-steering'),
    ]
    assert {'model_attempts', 'tool_calls'} <= {rule.name for rule in experiment.guard_metrics}


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


def test_pg_contract_freezes_context_only_and_disjoint_dev_selection():
    from eval.run_live_eval import load_dataset
    from src.pg_learning import EvolutionSpec
    folder = ROOT / 'research/experiments/pg-context-v1'
    experiment = ExperimentSpec.from_path(folder / 'experiment-spec.json')
    evolution = EvolutionSpec.model_validate_json((folder / 'evolution-spec.json').read_text())
    evolution.validate_dataset(load_dataset(ROOT / evolution.dataset))
    assert experiment.candidate_profile == 'adaptive-pg-context-v1'
    assert [(c.slot, c.implementation_id) for c in experiment.changes] == [('context', 'procedural-graph-context')]
    assert len(set(evolution.train_cases) | set(evolution.validation_cases)) == 10
    assert experiment.repeats == evolution.repeats == 3


def test_pg_guidance_contract_experiment_freezes_v2_localization_as_baseline():
    experiment = ExperimentSpec.from_path(
        ROOT / 'research/experiments/pg-guidance-adoption-v1/experiment-spec.json')
    assert experiment.baseline_profile == 'adaptive-pg-context-v2'
    assert experiment.candidate_profile == 'adaptive-pg-context-v3'
    assert [(change.kind, change.slot, change.implementation_id) for change in experiment.changes] == [
        ('mechanism', 'context', 'procedural-graph-context'),
    ]
    assert {'model_attempts', 'tool_calls'} <= {rule.name for rule in experiment.guard_metrics}


def test_pg_state_condition_experiment_uses_v3_baseline_and_fresh_holdout():
    from eval.run_live_eval import load_dataset

    experiment = ExperimentSpec.from_path(
        ROOT / 'research/experiments/pg-state-conditioned-planning-v1/experiment-spec.json')
    dataset = load_dataset(ROOT / experiment.dataset)
    assert experiment.baseline_profile == 'adaptive-pg-context-v3'
    assert experiment.candidate_profile == 'adaptive-pg-context-v4'
    assert dataset['split_counts'] == {'dev': 2, 'test': 4}
    assert all(case['id'].startswith('PGS-D') for case in dataset['cases'] if case['split'] == 'dev')
    assert all(case['id'].startswith('PGS-H') for case in dataset['cases'] if case['split'] == 'test')
    assert 'PG13' not in {case['id'] for case in dataset['cases']}


def test_pg_state_condition_v2_pins_same_graph_and_all_non_context_mechanisms():
    from src.harness.registry import get_runner, get_product_runner
    from src.errors import AgentError

    experiment = ExperimentSpec.from_path(
        ROOT / 'research/experiments/pg-state-conditioned-planning-v2/experiment-spec.json')
    baseline = get_runner(experiment.baseline_profile)
    candidate = get_runner(experiment.candidate_profile)
    left, right = baseline.provenance(), candidate.provenance()
    assert left['engine'] == right['engine']
    assert [m for m in left['mechanisms'] if m['slot'] != 'context'] == [
        m for m in right['mechanisms'] if m['slot'] != 'context']
    lconfig = next(m['configuration'] for m in left['mechanisms'] if m['slot'] == 'context')
    rconfig = next(m['configuration'] for m in right['mechanisms'] if m['slot'] == 'context')
    assert {key: rconfig[key] for key in lconfig} == lconfig
    assert set(rconfig) - set(lconfig) == {'applicability_strategy', 'path_ranking_strategy'}
    assert lconfig['graph_path'] == 'research/graphs/pg-seed-v2.json'
    assert len(lconfig['graph_sha256']) == 64
    assert baseline.build_context('test').applicability_strategy == 'none'
    assert candidate.build_context('test').applicability_strategy == 'typed-evidence-v1'
    assert experiment.dataset == 'eval/pg_state_condition_holdout_tasks.json'
    assert experiment.repeats == 3
    for profile_id in (experiment.baseline_profile, experiment.candidate_profile):
        with pytest.raises(AgentError, match='仅限研究'):
            get_product_runner(profile_id)


def test_preplay_contract_freezes_v4_and_only_adds_bounded_transition_search():
    from src.harness.registry import get_runner, get_product_runner
    from src.errors import AgentError

    experiment = ExperimentSpec.from_path(
        ROOT / 'research/experiments/pg-preplay-v6/experiment-spec.json')
    baseline = get_runner(experiment.baseline_profile)
    candidate = get_runner(experiment.candidate_profile)
    left, right = baseline.provenance(), candidate.provenance()
    assert left['engine'] == right['engine']
    assert [m for m in left['mechanisms'] if m['slot'] != 'context'] == [
        m for m in right['mechanisms'] if m['slot'] != 'context']
    lconfig = next(m['configuration'] for m in left['mechanisms'] if m['slot'] == 'context')
    rconfig = next(m['configuration'] for m in right['mechanisms'] if m['slot'] == 'context')
    assert {key: rconfig[key] for key in lconfig} == lconfig
    assert set(rconfig) - set(lconfig) == {
        'preplay_strategy', 'preplay_max_depth', 'preplay_beam_width', 'preplay_max_paths'}
    assert lconfig['graph_sha256'] == rconfig['graph_sha256']
    assert candidate.build_context('test').preplay_strategy == 'bounded-transition-search-v1'
    assert experiment.repeats == 3 and experiment.evaluation_split == 'test'
    for profile_id in (experiment.baseline_profile, experiment.candidate_profile):
        with pytest.raises(AgentError, match='仅限研究'):
            get_product_runner(profile_id)


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


def test_adaptive_contract_accepts_a_versioned_challenge_task_set():
    payload = json.loads((ROOT / 'research/experiments/ace-context-v1/experiment-spec.json').read_text())
    payload['dataset'] = 'eval/ace_challenge_tasks.json'

    spec = ExperimentSpec.model_validate_json(json.dumps(payload))

    assert spec.dataset == 'eval/ace_challenge_tasks.json'

    frozen = ExperimentSpec.from_path(
        ROOT / 'research/experiments/ace-context-v1/challenge-experiment-spec.json')
    assert frozen.dataset == 'eval/ace_challenge_tasks.json'
    assert frozen.repeats == 2


def test_ace_efficiency_contract_uses_fresh_dev_and_test_boundaries():
    frozen = ExperimentSpec.from_path(
        ROOT / 'research/experiments/ace-context-v1/efficiency-experiment-spec.json')
    dataset = json.loads((ROOT / frozen.dataset).read_text())

    assert frozen.dataset == 'eval/ace_efficiency_tasks.json'
    assert dataset['split_counts'] == {'dev': 3, 'test': 4}
    assert {case['split'] for case in dataset['cases']} == {'dev', 'test'}
    assert 'ACE-C01' not in {case['id'] for case in dataset['cases']}
    assert all(case['id'].startswith('ACE-E-D') for case in dataset['cases'] if case['split'] == 'dev')
    assert all(case['id'].startswith('ACE-E-H') for case in dataset['cases'] if case['split'] == 'test')
