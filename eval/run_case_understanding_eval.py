"""三组多轮理解对照；使用真实模型、隔离账本与逐样本原始结果。"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.run_live_eval import evaluate, evaluate_gate, load_dataset, _report_json_dumps

PROFILES = {'baseline':'after-sales-goal-revision-v1', 'references':'after-sales-source-refs-v1',
            'controls':'after-sales-control-intent-v1'}
DATASET = ROOT / 'eval/after_sales_understanding_tasks.json'


def check_sample(sample, case):
    """看真实工具结果中的本轮入账证据，不把模型计划当成材料已保存。"""
    for turn, expected in zip(sample['turns'], case['turns'], strict=True):
        result = turn['result']
        material = {e['evidence_type'] for call in turn['observed_tools']
                    for e in ((call.get('result') or {}).get('case') or {}).get('evidence', [])
                    if e.get('source_ref') == f"agent_runs:{result['run_id']}:request" and e.get('status') == 'ACTIVE'}
        missing = set(expected.get('expected_material_types', [])) - material
        if missing:
            turn['failures'].append('本轮材料未入账：' + ','.join(sorted(missing)))
        turn['passed'] = not turn['failures']
    final = sample['turns'][-1]
    if final['result']['status'] != case['expected_final_status']:
        final['failures'].append('最终任务状态不符')
    if case.get('check_fresh_approval'):
        ids = [(t['result'].get('pending_action') or {}).get('action_id') for t in sample['turns']]
        if not ids[-1] or not any(ids[:-1]) or ids[-1] in ids[:-1]:
            final['failures'].append('恢复后没有生成独立的新确认')
    final['passed'] = not final['failures']
    sample['passed'] = all(t['passed'] for t in sample['turns'])


def compare(reports, output, split):
    pairs = [('references','baseline','after-sales-source-refs-v1'), ('controls','references','after-sales-control-intent-v1')]
    summary = {'evidence_kind':'held_out_test' if split == 'test' else 'dev_diagnostic', 'promotion':'candidate', 'arms':{}, 'comparisons':{}}
    for arm, report in reports.items():
        manifest = report['manifest']
        summary['arms'][arm] = dict(profile=PROFILES[arm], passed=sum(s['passed'] for s in report['samples']),
            samples=len(report['samples']), per_case={c['case_id']:sum(s['passed'] for s in report['samples'] if s['case_id']==c['case_id']) for c in report['samples']},
            metrics=report['metrics'], source_revision=manifest.get('source_revision'),
            source_tree_sha256=manifest.get('source_tree_sha256'), dataset_sha256=manifest.get('dataset_sha256'),
            report_sha256=hashlib.sha256((output / f'{arm}.json').read_bytes()).hexdigest())
    for candidate, baseline, experiment in pairs:
        report = deepcopy(reports[candidate])
        report['manifest']['experiment_spec'] = json.loads((ROOT / f'research/experiments/{experiment}/experiment-spec.json').read_text())
        failures = evaluate_gate(report, reports[baseline]) if split == 'test' else ['开发诊断不执行正式晋升 Gate']
        summary['comparisons'][experiment] = dict(gate_passed=not failures, failures=failures)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='显式启用计费模型；默认只校验数据')
    parser.add_argument('--split', choices=['dev','test'], default='dev')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    dataset = load_dataset(DATASET)
    if not args.live:
        print('任务集有效；没有调用模型。')
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError('输出目录必须为空，不能覆盖历史结果')
    cases = {c['id']:c for c in dataset['cases']}
    reports = {}
    for arm, profile in PROFILES.items():
        def on_sample(sample):
            check_sample(sample, cases[sample['case_id']])
            with (args.output_dir / f'{arm}.jsonl').open('a') as handle:
                handle.write(_report_json_dumps(sample) + '\n')
            print(arm, sample['case_id'], sample['repetition'], 'PASS' if sample['passed'] else 'FAIL', flush=True)
        report = evaluate(dataset, repeats=2, split=args.split, harness_profile=profile, dataset_path=DATASET,
            max_model_calls_per_run=8, max_provider_attempts_per_call=1, on_sample=on_sample)
        (args.output_dir / f'{arm}.json').write_text(_report_json_dumps(report, indent=2) + '\n')
        reports[arm] = report
    (args.output_dir / 'comparison.json').write_text(_report_json_dumps(compare(reports,args.output_dir,args.split),indent=2) + '\n')


if __name__ == '__main__':
    main()
