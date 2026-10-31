from copy import deepcopy

import pytest

from eval.run_pg_effectiveness import aggregate, run


def block(case):
    return dict(status='COMPLETED', execution_source='live_provider', live_verified=True, providers=['fixed'],
        manifest={**{key: 'fixed' for key in ('dataset_sha256', 'evaluation_protocol_sha256',
            'business_protocol_sha256', 'python_runtime', 'dependency_lock_sha256', 'limits',
            'purpose_models', 'source_tree_sha256')}, 'harness_profile': {'profile_id': 'fixed'},
            'split': 'test', 'repeats': 2, 'selected_cases': [dict(id=case, split='test', turns=1)]},
        samples=[dict(case_id=case, repetition=rep, split='test', passed=rep == 1, adaptive_verified=True,
                      model_attempts=5, tool_calls=2, safety_violations=0, protocol_violations=0,
                      latency_ms=1000, turns=[]) for rep in (1, 2)])


def test_aggregation_preserves_failures_and_source_reports():
    reports = [block('A'), block('B')]
    original = deepcopy(reports)
    result = aggregate(reports, ('A', 'B'), 2)
    assert result['metrics']['mean_success_rate'] == .5
    assert result['metrics']['all_repeats_success_rate'] == 0
    assert len(result['block_reports_sha256']) == 2
    assert result['adaptive_verified_samples'] == 4
    assert reports == original


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'split', 'model', 'profile', 'source', 'incomplete'])
def test_aggregation_rejects_incomplete_or_incomparable_reports(defect):
    reports = [block('A'), block('B')]
    if defect == 'missing': reports[1]['samples'].pop()
    if defect == 'duplicate': reports[1]['samples'].append(deepcopy(reports[1]['samples'][0]))
    if defect == 'split': reports[1]['samples'][0]['split'] = 'dev'
    if defect == 'model': reports[1]['manifest']['purpose_models'] = 'changed'
    if defect == 'profile': reports[1]['manifest']['harness_profile']['profile_id'] = 'changed'
    if defect == 'source': reports[1]['execution_source'] = 'test_harness'
    if defect == 'incomplete': reports[1]['status'] = 'FAILED'
    with pytest.raises(ValueError):
        aggregate(reports, ('A', 'B'), 2)


def test_default_validation_does_not_run_models():
    result = run()
    assert result['model_calls'] == 0 and result['test_samples'] == 24


def test_frozen_product_constraints_are_feasible_after_two_candidates_disappear():
    from eval.run_live_eval import load_dataset
    from eval.run_pg_effectiveness import ROOT
    from src.backend.catalog import product_fixtures
    from src.adaptive_decision import Constraint, predicate_holds
    dataset = load_dataset(ROOT / 'eval/pg_effectiveness_20261031_tasks.json')
    for case in dataset['cases']:
        expected = case['adaptive_expect']
        targets = expected.get('recommendation_targets') or [expected['product_constraints']]
        for target in targets:
            matches = [p for p in product_fixtures() if all(
                predicate_holds(p, Constraint.model_validate(c)) for c in target)]
            needed = max(o.get('quantity', 1) for o in expected['objectives'])
            if case.get('scenario', {}).get('kind') in {'price_change', 'stock_change'}:
                needed += 2
            assert len(matches) >= needed, case['id']
