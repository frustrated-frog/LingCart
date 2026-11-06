"""从持久轨迹投影恢复过程；观察改善不等同于任务成功或因果归因。"""
from collections import defaultdict
from src.procedural_graph import fingerprint


def recovery_trace(sample):
    episodes, timeline = [], []
    for turn_index, turn in enumerate(sample.get('turns', [])):
        result = turn.get('result', {})
        steps = (result.get('plan') or {}).get('steps', [])
        events = result.get('events', [])
        sequences = {e['step_id']: e.get('sequence') for e in events
                     if e.get('type') == 'step_completed' and e.get('step_id')}
        seen, rows = set(), []
        for step in steps:
            body = step.get('result') if isinstance(step.get('result'), dict) else {}
            products = body.get('products') if isinstance(body.get('products'), list) else []
            identities = {p.get('product_id') for p in products if isinstance(p, dict) and p.get('product_id')}
            new_count = len(identities - seen)
            seen |= identities
            code = (step.get('error') or {}).get('code')
            failed = step.get('status') in {'FAILED', 'REJECTED', 'BUSINESS_ERROR'}
            empty = step.get('tool') in {'search_products', 'discover_products', 'compare_products'} and body.get('products') == []
            kind = ('empty_results' if empty else 'verification_rejected' if failed and step.get('tool') in {'select_product','complete_objectives'} else 'tool_failure' if failed else None)
            row = dict(turn=turn_index, step_id=step['step_id'], sequence=sequences.get(step['step_id']),
                       tool=step.get('tool'), status=step.get('status'), failure_kind=kind,
                       error_code=code, new_records=new_count, observation_available=bool(body),
                       action_identity=fingerprint([step.get('tool'), step.get('args')]))
            rows.append(row)
        for index, row in enumerate(rows):
            timeline.append({k:v for k,v in row.items() if k != 'action_identity'})
            if not row['failure_kind']:
                continue
            following = rows[index+1:index+4]
            actions = [dict(tool=x['tool'], step_id=x['step_id'], sequence=x['sequence'], status=x['status'],
                            same_arguments=x['action_identity'] == row['action_identity'],
                            new_records=x['new_records']) for x in following]
            episodes.append(dict(turn=turn_index, failure_sequence=row['sequence'], step_id=row['step_id'],
                failure_kind=row['failure_kind'], error_code=row['error_code'], recovery_actions=actions,
                observation_improved=any(x['new_records'] > 0 for x in following),
                outcome='TASK_SUCCEEDED' if sample.get('passed') is True else 'TASK_NOT_SUCCEEDED'))
        for event in events:
            if event.get('type') == 'decision_rejected':
                sequence = event.get('sequence')
                following = [x for x in rows if sequence is not None and x['sequence'] is not None and x['sequence'] > sequence][:3]
                episodes.append(dict(turn=turn_index, failure_sequence=sequence, step_id=None,
                    failure_kind='decision_rejected', error_code=event.get('code'),
                    recovery_actions=[dict(tool=x['tool'], step_id=x['step_id'], sequence=x['sequence'],
                        status=x['status'], same_arguments=None, new_records=x['new_records']) for x in following],
                    observation_improved=any(x['new_records'] > 0 for x in following),
                    outcome='TASK_SUCCEEDED' if sample.get('passed') is True else 'TASK_NOT_SUCCEEDED'))
    return dict(episodes=episodes, timeline=timeline, task_passed=sample.get('passed') is True,
                verified_objectives=sum(row.get('status') == 'VERIFIED'
                    for t in sample.get('turns', [])[-1:] for row in (t.get('result', {}).get('plan') or {}).get('objective_ledger', [])),
                repeated_without_new_records=sum(a['same_arguments'] is True and a['new_records'] == 0
                    for e in episodes for a in e['recovery_actions']),
                note='新增记录仅统计后续三步首次出现的商品编号数量，不覆盖其他信息增益；任务成功单独判定，不代表因果收益')


def training_feedback(report):
    groups = defaultdict(list)
    samples = []
    for sample in report['samples']:
        row = dict(sample_ref=f"{sample['case_id']}:{sample['repetition']}", passed=sample['passed'],
                   model_attempts=sample['model_attempts'], recovery=recovery_trace(sample))
        # 原始事件中只选取类型与已定义状态，不携带指令、请求、工具参数或业务事实。
        row['module_triggers'] = [dict(module=e.get('module'), trigger=e.get('trigger'), status=e.get('status'))
            for t in sample.get('turns', []) for e in t.get('result', {}).get('events', []) if e.get('type') == 'rrsi_module_applied']
        row['actions'] = [{k:e[k] for k in ('type','tool','status','code') if k in e}
            for t in sample.get('turns', []) for e in t.get('result', {}).get('events', [])
            if e.get('type') in {'tool_result','decision_rejected'}]
        samples.append(row)
        groups[sample['case_id']].append(row)
    contrasts = []
    for case, rows in groups.items():
        success = [r['sample_ref'] for r in rows if r['passed']]
        failure = [r['sample_ref'] for r in rows if not r['passed']]
        if success and failure:
            contrasts.append(dict(case_id=case, successful_samples=success, failed_samples=failure,
                                  interpretation='同任务不同重复的关联对照，不是因果识别'))
    return dict(schema_version=1, samples=samples, contrasts=contrasts)
