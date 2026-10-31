"""冻结开发集学习与三组独立留出评测；默认只检查配置，--live 才调用模型。"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from eval.experiment_contract import ExperimentSpec
from eval.run_live_eval import evaluate, load_dataset, summarize
from eval.run_pg_evolution import graph_profile
from src.harness import get_registry, get_runner
from src.pg_experiment import behavior_metrics, comparison
from src.pg_learning import EvolutionSpec, ModelGraphRefiner, comparable, evolve, write_artifact
from src.procedural_graph import ProceduralGraph, fingerprint

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'research/experiments/pg-effectiveness-20261031'
ARMS = ('baseline', 'seed', 'evolved')


def aggregate(reports, cases, repeats):
    """只合并同臂、同环境完整区块；拒绝遗漏、重复或混入开发样本。"""
    if not reports:
        raise ValueError('缺少原始报告')
    first = reports[0]
    samples = [s for report in reports for s in report['samples']]
    expected = Counter((case, rep) for case in cases for rep in range(1, repeats + 1))
    if Counter((s['case_id'], s['repetition']) for s in samples) != expected:
        raise ValueError('样本遗漏、重复或未冻结')
    for report in reports:
        if (not comparable(first, report) or report.get('status') != 'COMPLETED'
                or report.get('execution_source') != first.get('execution_source')
                or report['manifest']['harness_profile'] != first['manifest']['harness_profile']
                or report['manifest']['split'] != 'test' or report['manifest']['repeats'] != repeats
                or any(s['split'] != 'test' for s in report['samples'])):
            raise ValueError('原始区块不具备同组可比性')
    result = deepcopy(first)
    result['samples'] = deepcopy(samples)
    result['manifest']['selected_cases'] = [case for report in reports
                                           for case in report['manifest']['selected_cases']]
    result['block_reports_sha256'] = [fingerprint(report) for report in reports]
    result['live_verified'] = all(report.get('live_verified') is True for report in reports)
    result['adaptive_verified_samples'] = sum(s.get('adaptive_verified') is True for s in samples)
    result['metrics'] = summarize(result['samples'], repeats)
    return result


def run(*, live=False, output=None):
    plan = json.loads((EXPERIMENT / 'evaluation-plan.json').read_text())
    spec = EvolutionSpec.model_validate_json((EXPERIMENT / 'evolution-spec.json').read_text())
    contract_template = ExperimentSpec.from_path(EXPERIMENT / 'experiment-spec.json')
    dataset_path = ROOT / spec.dataset
    dataset = load_dataset(dataset_path)
    spec.validate_dataset(dataset)
    test_cases = tuple(c['id'] for c in dataset['cases'] if c['split'] == 'test')
    if list(test_cases) != plan['test_cases'] or contract_template.repeats != plan['test_repeats']:
        raise ValueError('冻结计划与任务集或重复次数不一致')
    graph = ProceduralGraph.from_path(ROOT / plan['seed_graph'])
    if not live:
        return dict(status='VALIDATED', model_calls=0, test_samples=len(test_cases) * plan['test_repeats'] * 3,
                    dataset_sha256=fingerprint(dataset), seed_sha256=graph.fingerprint)
    if output is None or output.exists() or not output.resolve().is_relative_to(EXPERIMENT):
        raise ValueError('live 输出必须为本实验目录内尚不存在的目录')
    output.mkdir(parents=True)
    write_artifact(output / 'frozen-inputs.json', dict(plan=plan, evolution=spec.model_dump(mode='json'),
        contract=contract_template.model_dump(mode='json'), dataset_sha256=fingerprint(dataset),
        driver_sha256=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()))
    profiles = {}

    def profile(current):
        if current.fingerprint not in profiles:
            path = output / 'graphs' / f'{current.fingerprint}.json'
            write_artifact(path, current.model_dump(mode='json'))
            item = graph_profile(current, path, online_profile=spec.online_profile)
            write_artifact(path.with_name(f'{current.fingerprint}-profile.json'), item.model_dump(mode='json'))
            get_registry().register_profile(item)
            profiles[current.fingerprint] = item.profile_id
        return profiles[current.fingerprint]

    def execute(profile_id, cases, split, repeats, label):
        def on_sample(sample):
            write_artifact(output / 'samples' / f'{label}-{sample["case_id"]}-{sample["repetition"]}.json', sample)
            print(json.dumps(dict(stage=label, case=sample['case_id'], repetition=sample['repetition'],
                passed=sample['passed'], model_attempts=sample['model_attempts'],
                statuses=[t['result']['status'] for t in sample['turns']]), ensure_ascii=False), flush=True)
        return evaluate(dataset, repeats=repeats, split=split, case_ids=cases, execution_mode='adaptive',
            harness_profile=profile_id, dataset_path=dataset_path, on_sample=on_sample,
            max_model_calls_per_run=spec.max_model_calls_per_run,
            max_prompt_tokens_per_run=spec.max_prompt_tokens_per_run,
            max_completion_tokens_per_run=spec.max_completion_tokens_per_run,
            max_seconds_per_run=spec.max_seconds_per_run,
            max_provider_attempts_per_call=spec.max_provider_attempts_per_call)

    retained, evolution = evolve(graph, spec,
        evaluator=lambda current, cases, label: execute(profile(current), cases, 'dev', spec.repeats, label),
        refiner=ModelGraphRefiner(evolution_policy=spec.evolution_policy,
            max_prompt_tokens=spec.max_refiner_prompt_tokens,
            max_completion_tokens=spec.max_refiner_completion_tokens,
            max_provider_attempts=spec.max_provider_attempts_per_call), output_dir=output / 'learning')
    print(json.dumps(dict(stage='selection', changed=retained.fingerprint != graph.fingerprint,
        results=[dict(status=h['status'], reason=h['reason']) for h in evolution['history']]), ensure_ascii=False), flush=True)
    ids = dict(baseline=plan['baseline_profile'], seed=profile(graph), evolved=profile(retained))
    contracts = {}
    for arm, base in (('seed', 'baseline'), ('evolved', 'seed')):
        if ids[arm] == ids[base]:
            continue
        contract = contract_template.model_dump(mode='json')
        contract.update(experiment_id=f'{plan["experiment_id"]}-{arm}', baseline_profile=ids[base], candidate_profile=ids[arm])
        context = next(m for m in get_runner(ids[arm]).provenance()['mechanisms'] if m['slot'] == 'context')
        contract['changes'][0]['paths'] = context['source_files']
        ExperimentSpec.model_validate_json(json.dumps(contract))
        contracts[arm] = contract
    frozen = dict(preset='exploratory', groups=dict(test=list(test_cases)), spec=dict(repeats=plan['test_repeats']),
                  dataset_sha256=fingerprint(dataset), profiles=ids, contracts=contracts,
                  cost_guard=dict(max_extra_model_attempts_per_sample=spec.max_extra_model_attempts_per_sample))
    write_artifact(output / 'test-frozen.json', frozen)  # 在首次 test 调用前冻结动态图身份和契约。
    blocks = {arm: [] for arm in ARMS}
    for index, case in enumerate(test_cases):
        order = ARMS[index % 3:] + ARMS[:index % 3]
        for arm in order:
            report = execute(ids[arm], (case,), 'test', plan['test_repeats'], f'test-{arm}')
            write_artifact(output / 'raw' / f'{case}-{arm}.json', report)
            blocks[arm].append(report)
    reports = {arm: aggregate(blocks[arm], test_cases, plan['test_repeats']) for arm in ARMS}
    for arm, report in reports.items():
        write_artifact(output / f'{arm}-report.json', report)
    metrics = {}
    for arm, report in reports.items():
        metrics[arm] = behavior_metrics(report)
        metrics[arm]['false_completion_samples'] = sum(not s['passed'] and any(
            t['result']['status'] == 'SUCCEEDED' and not t['passed'] for t in s['turns']) for s in report['samples'])
    result = dict(status='COMPLETED', comparison=comparison(reports, frozen, contracts), metrics=metrics,
                  retained_graph_changed=retained.fingerprint != graph.fingerprint,
                  limitations=['4个test任务各重复2次，不能作统计显著性结论', '只覆盖虚构目录单轮选购与预设扰动',
                               '未评估支付/退款、多轮长程任务、ACE或LivePlan', '同图第三组只测随机波动，不证明演化收益'])
    write_artifact(output / 'summary.json', result)
    return dict(status=result['status'], output=str(output), comparison=result['comparison'],
                rates={arm: values['mean_success_rate'] for arm, values in metrics.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(live=args.live, output=args.output.resolve() if args.output else None), ensure_ascii=False, indent=2))
