"""固定任务及预算的四臂留出对照；不把结果传回学习器。"""
from pathlib import Path
from src.pg_learning import comparable, write_artifact
from src.rrsi_learning import measurement
from src.rrsi_policy import HarnessProgram


def run_ablation(seed, retained, spec, *, evaluator, output_dir: Path, source='live_provider'):
    control = seed.model_dump(mode='json')
    target = spec.target_module
    # 保留候选的调用模式与触发策略，只还原学习到的指令，检查额外分析是否解释结果。
    control[target].update(mode=getattr(retained, target).mode, trigger=getattr(retained, target).trigger)
    import json
    schedule = HarnessProgram.model_validate_json(json.dumps(control))
    arms = dict(baseline=None, fixed=seed, schedule_control=schedule, evolved=retained)
    write_artifact(output_dir / 'ablation-contract.json', dict(schema_version=1,
        dataset=spec.dataset, cases=spec.test_cases, repeats=spec.repeats,
        arms={name:p.fingerprint if p else 'adaptive-v1' for name,p in arms.items()},
        max_model_calls_per_run=spec.max_model_calls_per_run, max_seconds_per_run=spec.max_seconds_per_run,
        max_token_regression=spec.beta0, metric='task_success_with_cost_guard', max_success_regression=0,
        promotion_status='candidate', note='各臂总预算一致，实际模型调用量仍可能不同；对照是关联证据'))
    reports, stats = {}, {}
    result = dict(status='INVALID', measurements=stats, observed_improvement=False, promotion_status='candidate')
    try:
        for name, program in arms.items():
            raw = evaluator(program, spec.test_cases, 'ablation-' + name, split='test')
            write_artifact(output_dir / f'ablation-{name}.json', raw)
            reports[name] = raw
            if program is None and raw.get('manifest', {}).get('harness_profile', {}).get('profile_id') != 'adaptive-v1':
                raise ValueError('RSI_BASELINE_MISMATCH')
            stats[name] = measurement(raw, program, spec.test_cases, spec.repeats, split='test', source=source,
                                      target_module=target if program else None)
        if any(not comparable(reports['baseline'], report) for report in reports.values()):
            raise ValueError('RSI_ENVIRONMENT_CHANGED')
        candidate = stats['evolved']
        guards = all(candidate['score'] >= stats[a]['score'] and candidate['pass_at_1'] >= stats[a]['pass_at_1']
                     and candidate['stable'] >= stats[a]['stable'] and candidate['tokens'] <= stats[a]['tokens'] * (1 + spec.beta0)
                     for a in ('baseline','fixed','schedule_control'))
        improvement = any(candidate['score'] > stats[a]['score'] or candidate['tokens'] < stats[a]['tokens']
                          for a in ('fixed','schedule_control'))
        result.update(status='COMPLETED', observed_improvement=guards and improvement
            and not candidate['safety'] and not candidate['protocol'] and retained.fingerprint != seed.fingerprint,
            code='MEASURED_NOT_PROMOTED')
    except ValueError as error:
        result['code'] = str(error) if str(error).startswith('RSI_') else 'RSI_ABLATION_INVALID'
    write_artifact(output_dir / 'ablation-summary.json', result)
    return result
