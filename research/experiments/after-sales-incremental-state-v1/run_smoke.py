"""固定两项开发诊断，各重复两次；精确检查入账材料类型，保留每个负结果。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from eval.run_live_eval import evaluate, load_dataset, _report_json_dumps


def material_checks(sample, case):
    checks=[]
    for turn, expected in zip(sample['turns'], case['turns']):
        result=turn['result']
        expected_types=expected.get('expected_material_types')
        if expected_types is None:
            continue
        actual=set()
        for event in result.get('events', []):
            value=event.get('result') if event['type']=='tool_result' else None
            if not isinstance(value, dict): continue
            for evidence in (value.get('case') or {}).get('evidence', []):
                if evidence['source_ref']==f"agent_runs:{result['run_id']}:request" and evidence['status']=='ACTIVE':
                    actual.add(evidence['evidence_type'])
        checks.append(dict(run_id=result['run_id'], expected_types=sorted(expected_types), actual_types=sorted(actual),
                           passed=actual==set(expected_types)))
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    dataset_path=ROOT/'eval/after_sales_incremental_dev_tasks.json'
    dataset=load_dataset(dataset_path)
    from eval.run_task_eval import _runtime, _isolated_db
    from src.after_sales.policy import AfterSalesPolicyEngine
    from src.backend.seed import DEFAULT_AFTER_SALES_POLICY_VERSION_ID
    runtime=_runtime()
    for case in dataset['cases']:
        with _isolated_db(runtime,case['fixture']):
            with runtime['tools']._session() as session:
                policy=session.get(runtime['db'].PolicyVersion,DEFAULT_AFTER_SALES_POLICY_VERSION_ID)
                AfterSalesPolicyEngine._validated_rules(policy)
    for name,profile in [('baseline','after-sales-lookahead-v1'),('candidate','after-sales-incremental-state-v1')]:
        def persist_sample(sample):
            path=args.output/f"{name}-{sample['case_id']}-{sample['repetition']}.json"
            path.write_text(_report_json_dumps(sample,indent=2)+'\n')
            print(name,sample['case_id'],sample['repetition'],sample['passed'],flush=True)
        report=evaluate(dataset,dataset_path=dataset_path,repeats=2,split='dev',harness_profile=profile,
                        max_model_calls_per_run=8,max_provider_attempts_per_call=1,on_sample=persist_sample)
        cases={case['id']:case for case in dataset['cases']}
        diagnostics=[]
        for sample in report['samples']:
            checks=material_checks(sample,cases[sample['case_id']])
            final=sample['turns'][-1]['result']
            final_ok=final['status']=='WAITING_APPROVAL' and not final.get('pending_clarification')
            diagnostics.append(dict(case_id=sample['case_id'],repetition=sample['repetition'],material_checks=checks,
                final_confirmation_without_stale_question=final_ok,
                complete=sample['passed'] and all(c['passed'] for c in checks) and final_ok))
        report['diagnostic_checks']=diagnostics
        (args.output/f'{name}.json').write_text(_report_json_dumps(report,indent=2)+'\n')
        print(json.dumps(dict(profile=profile,complete=sum(d['complete'] for d in diagnostics),samples=len(diagnostics))),flush=True)


if __name__=='__main__': main()
