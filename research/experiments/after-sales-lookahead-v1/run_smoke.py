"""同版 V5 / 前瞻候选三轮开发诊断；每次输出新目录，保留所有负结果。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from eval.run_live_eval import evaluate, load_dataset, _report_json_dumps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset_path = ROOT / 'eval/after_sales_lookahead_dev_tasks.json'
    dataset = load_dataset(dataset_path)
    # 先在隔离库核验政策夹具，避免把配置错误当成 Agent 失败或浪费模型请求。
    from eval.run_task_eval import _runtime, _isolated_db
    from src.after_sales.policy import AfterSalesPolicyEngine
    from src.backend.seed import DEFAULT_AFTER_SALES_POLICY_VERSION_ID
    runtime = _runtime()
    for case in dataset['cases']:
        with _isolated_db(runtime, case['fixture']):
            with runtime['tools']._session() as session:
                policy = session.get(runtime['db'].PolicyVersion, DEFAULT_AFTER_SALES_POLICY_VERSION_ID)
                AfterSalesPolicyEngine._validated_rules(policy)
    for name, profile in [('baseline', 'after-sales-active-clarification-v5'), ('candidate', 'after-sales-lookahead-v1')]:
        report = evaluate(dataset, dataset_path=dataset_path, repeats=1, split='dev',
                          harness_profile=profile, max_model_calls_per_run=8,
                          max_provider_attempts_per_call=1)
        (args.output / f'{name}.json').write_text(_report_json_dumps(report, indent=2) + '\n')
        print(_report_json_dumps(dict(profile=profile, status=report['status'],
                                     passed=[s['passed'] for s in report['samples']],
                                     failures=[t['failures'] for s in report['samples'] for t in s['turns']])), flush=True)


if __name__ == '__main__':
    main()
