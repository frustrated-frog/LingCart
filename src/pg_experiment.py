"""PG 挑战集、冻结实验计划与三组对照判据；不触碰生产 Runtime。"""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from eval.run_live_eval import load_dataset, summarize, evaluate_gate
from src.pg_learning import EvolutionSpec, comparable
from src.procedural_graph import ProceduralGraph, canonical, fingerprint

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'eval/pg_challenge_tasks.json'
SEED = ROOT / 'research/graphs/pg-seed-v1.json'
ARMS = ('baseline', 'seed', 'evolved')


def challenge_dataset():
    dataset = load_dataset(DATASET)
    families, requests = set(), set()
    for case in dataset['cases']:
        partition = case.get('learning_partition')
        if partition not in {'train', 'validation', 'test'} or case['split'] != ('test' if partition == 'test' else 'dev'):
            raise ValueError('PG 分组必须将训练、验证和测试隔离')
        family = case.get('scenario_family')
        request = canonical(case['turns'])
        if not family or family in families or request in requests:
            raise ValueError('PG 情境族与请求不能重复或跨组')
        families.add(family)
        requests.add(request)
    if Counter(c['learning_partition'] for c in dataset['cases']) != {'train': 6, 'validation': 6, 'test': 6}:
        raise ValueError('PG v1 必须完整冻结 6/6/6 个情境')
    return dataset


def plan(preset):
    if preset not in {'smoke', 'full'}:
        raise ValueError('未知实验配置')
    dataset = challenge_dataset()
    groups = {part: [c['id'] for c in dataset['cases'] if c['learning_partition'] == part]
              for part in ('train', 'validation', 'test')}
    spec = EvolutionSpec.model_validate_json((ROOT / 'research/experiments/pg-challenge-v1/evolution-spec.json').read_text())
    spec.validate_dataset(dataset)
    if (list(spec.train_cases) != groups['train'] or list(spec.validation_cases) != groups['validation']
            or spec.dataset != DATASET.relative_to(ROOT).as_posix()):
        raise ValueError('PG 训练配置与冻结挑战集不一致')
    if preset == 'smoke':
        spec = spec.model_copy(update={'repeats': 1, 'iterations': 1})
    repeats, iterations = spec.repeats, spec.iterations
    samples = repeats * (len(groups['validation']) + iterations * (len(groups['train']) + len(groups['validation']))
                         + 3 * len(groups['test']))
    return dict(preset=preset, spec=spec.model_dump(mode='json'), groups=groups,
                max_samples=samples, max_model_attempts=samples * spec.max_model_calls_per_run + iterations * 2,
                dataset_sha256=fingerprint(dataset), seed_sha256=ProceduralGraph.from_path(SEED).fingerprint,
                cost_guard=dict(max_extra_model_attempts_per_sample=2, max_tool_call_regression=0))


def behavior_metrics(report):
    samples = report.get('samples', [])
    recovery = exposed = repeated = 0
    triggered = 0
    for sample in samples:
        sample_exposed = sample_triggered = False
        for turn in sample.get('turns', []):
            result = turn.get('result', {})
            sample_triggered |= bool(result.get('evaluation_environment', {}).get('fired'))
            seen = set()
            for event in result.get('events', []):
                sample_exposed |= (event.get('type') == 'mechanism_applied'
                                   and event.get('mechanism_id') == 'procedural-graph-context')
                if event.get('type') != 'tool_result':
                    continue
                value = event.get('result')
                invalid = isinstance(value, dict) and (value.get('error') or value.get('success') is False
                                                       or value.get('products') == [])
                if invalid:
                    key = canonical([event.get('tool'), event.get('args'), value])
                    repeated += key in seen
                    seen.add(key)
        triggered += sample_triggered
        recovery += sample_triggered and sample['passed']
        exposed += sample_exposed
    metrics = summarize(samples, report.get('manifest', {}).get('repeats', 1))
    return {**metrics, 'triggered_samples': triggered, 'recovered_samples': recovery,
            'recovery_rate': recovery / triggered if triggered else None,
            'pg_exposed_samples': exposed, 'pg_exposure_rate': exposed / len(samples) if samples else None,
            'repeated_ineffective_calls': repeated}


def test_contract(candidate, baseline, repeats):
    import json
    contract = json.loads((ROOT / 'research/experiments/pg-context-v1/experiment-spec.json').read_text())
    contract.update(experiment_id='pg-challenge-' + candidate['profile_id'][-24:],
        candidate_profile=candidate['profile_id'], baseline_profile=baseline['profile_id'],
        dataset=DATASET.relative_to(ROOT).as_posix(), repeats=repeats)
    context = next(m for m in candidate['mechanisms'] if m['slot'] == 'context')
    contract['changes'][0]['paths'] = list(context['source_files'])
    return contract


def comparison(reports, frozen, contracts):
    """以原始样本重算指标。冒烟、替身、缺项或未演化均不宣称收益。"""
    from eval.run_task_eval import dataset_fingerprint
    expected = Counter((case, repetition) for case in frozen['groups']['test']
                       for repetition in range(1, frozen['spec']['repeats'] + 1))
    failures = []
    for arm in ARMS:
        report = reports[arm]
        manifest = report.get('manifest', {})
        if (report.get('status') != 'COMPLETED' or manifest.get('split') != 'test'
                or manifest.get('dataset_sha256') != dataset_fingerprint(challenge_dataset())
                or Counter((s.get('case_id'), s.get('repetition')) for s in report.get('samples', [])) != expected
                or any(s.get('split') != 'test' or not s.get('adaptive_verified') for s in report.get('samples', []))):
            failures.append(f'{arm}: 样本不完整或数据身份不符')
        if not comparable(reports['baseline'], report):
            failures.append(f'{arm}: 数据、源码、模型或预算不可比')
    gates = {}
    for arm, base in (('seed', 'baseline'), ('evolved', 'seed')):
        candidate = deepcopy(reports[arm])
        contract = contracts.get(arm)
        if contract is None:
            gates[arm] = dict(passed=False, failures=['没有保留新的图结构，演化组仍使用种子图'])
            continue
        candidate['manifest']['experiment_spec'] = contract
        errors = evaluate_gate(candidate, reports[base])
        allowance = frozen['cost_guard']['max_extra_model_attempts_per_sample'] * len(expected)
        if candidate['metrics']['model_attempts'] > reports[base]['metrics']['model_attempts'] + allowance:
            errors.append('模型调用成本超过每样本额外 2 次的预设上限')
        gates[arm] = dict(passed=not errors, failures=errors)
    live = all(r.get('execution_source') == 'live_provider' and r.get('live_verified') for r in reports.values())
    valid = live and not failures and all(g['passed'] for g in gates.values())
    improved = reports['evolved']['metrics']['mean_success_rate'] > reports['seed']['metrics']['mean_success_rate']
    conclusion = ('SMOKE_ONLY' if frozen['preset'] == 'smoke' else 'TEST_HARNESS' if not live else
                  'INCONCLUSIVE' if not valid else 'OBSERVED_IMPROVEMENT' if improved else 'NO_REGRESSION')
    return dict(conclusion=conclusion, gates=gates, failures=failures, promotion_status='candidate',
                evidence_valid=valid, statistical_significance=None)
