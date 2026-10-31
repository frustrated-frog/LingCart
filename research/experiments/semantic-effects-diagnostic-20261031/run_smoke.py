"""已见任务诊断回归：每题一次，保存全部结果，不做机制收益比较。"""
import json
from pathlib import Path

from eval.run_live_eval import ROOT, evaluate, load_dataset, _report_json_dumps


def main():
    destination = Path(__file__).resolve().parent / 'run-01'
    destination.mkdir(exist_ok=False)
    selections = (
        ('hub', 'eval/pg_effectiveness_20261031_tasks.json', 'PGE-T01', 'test', 'adaptive-v1', 'adaptive'),
        ('case', 'eval/after_sales_autonomy_tasks.json', 'ASA-D01', 'dev', 'after-sales-policy-execution-v1', 'auto'),
        ('unknown', 'eval/after_sales_evidence_boundary_tasks.json', 'ASE-D01', 'dev', 'after-sales-evidence-boundary-v2', 'auto'),
    )
    limits = dict(max_model_calls_per_run=16, max_prompt_tokens_per_run=256000,
                  max_completion_tokens_per_run=12000, max_seconds_per_run=90,
                  max_provider_attempts_per_call=1)
    plan = dict(purpose='seen_case_diagnostic_not_effectiveness_evaluation', repeats=1,
                selections=selections, requested_limits=limits,
                limit_policy='evaluate intersects each limit with existing runtime defaults')
    (destination / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    for label, dataset, case, split, profile, mode in selections:
        path = ROOT / dataset
        report = evaluate(load_dataset(path), dataset_path=path, case_ids=(case,),
                          split=split, repeats=1, execution_mode=mode, harness_profile=profile, **limits)
        (destination / f'{label}.json').write_text(_report_json_dumps(report, indent=2) + '\n')
        print(json.dumps(dict(label=label, status=report['status'],
                              passed=[sample['passed'] for sample in report['samples']],
                              model_attempts=[sample['model_attempts'] for sample in report['samples']]),
                         ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
