"""固定任务的真实模型评测入口；--dry-run 不导入应用、不调用模型。"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time
import uuid

try:
    from eval import run_task_eval as offline
except ModuleNotFoundError:
    import run_task_eval as offline

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(__file__).with_name('live_tasks.json')


def load_dataset(path=DATASET):
    dataset = offline.load_dataset(path)
    if len(dataset['cases']) != 20 or any(sum(c.get('split') == split for c in dataset['cases']) != 10 for split in ('dev', 'test')):
        raise ValueError('实时评测必须包含固定的 20 项开发集与测试集任务')
    return dataset


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def environment_manifest(dataset, execution_mode):
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    prompt_files = sorted(set(['src/orchestrator.py', 'src/planner.py', 'src/adaptive.py', 'src/adaptive_decision.py', 'src/adaptive_goals.py'] +
                             [str(p.relative_to(ROOT)) for p in (ROOT / 'src/agents').glob('*.py')]))
    return {'source_revision': revision, 'dataset_sha256': fingerprint(dataset), 'execution_mode': execution_mode,
            'prompt_source_sha256': {name: fingerprint([ast.dump(node.value) for node in ast.walk(ast.parse((ROOT / name).read_text()))
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and any('prompt' in getattr(target, 'id', '').lower() for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))])
                for name in prompt_files if (ROOT / name).exists()},
            'business_environment': 'fresh_demo_sqlite_per_sample', 'pricing_verified': False}


def summarize(samples, repeats):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample['case_id']].append(sample)
    for items in groups.values():
        items.sort(key=lambda sample: sample.get('repetition', 0))
    latencies = [sample['latency_ms'] for sample in samples]
    costs = [t.get('usage') or {} for s in samples for t in s['turns']]
    tokens = {key: sum(c[key] for c in costs) if costs and all(c.get(key) is not None for c in costs) else None
              for key in ('input_tokens', 'output_tokens', 'total_tokens')}
    return {'tasks': len(groups), 'samples': len(samples), 'repeats': repeats,
            'pass_at_1': sum(items[0]['passed'] for items in groups.values()) / len(groups) if groups else None,
            'mean_success_rate': sum(s['passed'] for s in samples) / len(samples) if samples else None,
            'all_repeats_success_rate': sum(len(items) == repeats and all(s['passed'] for s in items) for items in groups.values()) / len(groups) if groups else None,
            'model_attempts': sum(s['model_attempts'] for s in samples),
            'tool_calls': sum(s['tool_calls'] for s in samples),
            'safety_violations': sum(s['safety_violations'] for s in samples),
            'protocol_violations': sum(s['protocol_violations'] for s in samples),
            'latency_ms': {'mean': statistics.mean(latencies) if latencies else None,
                           'p50': offline._percentile(latencies, .5) if latencies else None,
                           'p95': offline._percentile(latencies, .95) if latencies else None},
            'tokens': tokens, 'monetary_cost': None}


def evaluate(dataset, *, repeats=3, split='all', execution_mode='auto', runner=None, readiness=None):
    """runner 注入只用于基础设施测试，报告会明确标识为 test_harness。"""
    if not 1 <= repeats <= 10:
        raise ValueError('重复次数必须在 1 至 10 之间')
    runtime = offline._runtime()
    from src.durable import run_conversation
    from src.llm import get_model_status
    from src.runtime import BudgetLimits
    from src.trace_projector import project_trace
    report = {'schema_version': 1, 'generated_at': datetime.now(UTC).isoformat(),
              'execution_source': 'test_harness' if runner is not None else 'live_provider',
              'manifest': {**environment_manifest(dataset, execution_mode), 'split': split, 'repeats': repeats}, 'samples': []}
    status = readiness if readiness is not None else get_model_status()
    report['providers'] = [{key: p.get(key) for key in ('provider', 'model', 'slot', 'custom_base_url')} for p in status.get('providers', [])]
    from src.model_gateway import load_gateway_config, ModelGatewayConfigurationError
    report['manifest']['purpose_models'] = {}
    for purpose in ('router', 'planner', 'agent', 'final_response'):
        try:
            config = load_gateway_config(purpose=purpose)
            report['manifest']['purpose_models'][purpose] = [{'provider': p.provider, 'model': p.model} for p in config.providers]
        except ModelGatewayConfigurationError:
            report['manifest']['purpose_models'][purpose] = None
    if not status.get('configured') and runner is None:
        report.update(status='NOT_RUN', reason='MODEL_NOT_CONFIGURED', message='尚未配置模型，未执行真实评测', metrics=summarize([], repeats))
        return report
    if runner is None:
        if execution_mode == 'adaptive':
            from src.adaptive import run_adaptive
            runner = run_adaptive
        else:
            runner = runtime['orchestrator'].run_orchestrator
    limits = BudgetLimits(max_model_calls=8, max_tool_calls=12, max_seconds=45)
    report['manifest']['limits'] = limits.__dict__
    selected = [case for case in dataset['cases'] if split == 'all' or case['split'] == split]
    report['manifest']['selected_cases'] = [dict(id=c['id'], split=c['split'], turns=len(c['turns'])) for c in selected]
    for case in selected:
        for repetition in range(repeats):
            sample = {'case_id': case['id'], 'split': case['split'], 'repetition': repetition + 1,
                      'passed': True, 'turns': [], 'model_attempts': 0, 'tool_calls': 0,
                      'safety_violations': 0, 'protocol_violations': 0}
            started = time.perf_counter()
            with offline._isolated_db(runtime, case.get('fixture', {})):
                session_id = uuid.uuid4().hex
                warmup = case['turns'][0].get('history', [])
                # 固定对话上下文直接落库，不执行隐藏的预热模型或业务调用。
                from langchain_core.messages import HumanMessage, AIMessage
                fixed_history = [(HumanMessage if p['role'] == 'user' else AIMessage)(content=p['content']) for p in warmup]
                for turn_index, turn in enumerate(case['turns']):
                    before = offline._snapshot(runtime, runtime['tools']._engine)
                    try:
                        with offline._observe_tools(runtime) as observed:
                            def sample_runner(message, **kwargs):
                                if turn_index == 0 and fixed_history:
                                    kwargs['history'] = fixed_history
                                return runner(message, **kwargs)
                            result = run_conversation(turn['message'], session_id, request_id=str(turn_index), limits=limits, runner=sample_runner)
                    except Exception as error:
                        from src.persistence import RunStore
                        active = RunStore().conversation(session_id)['active_run_id']
                        saved = RunStore().run(active) if active else None
                        events = saved['events'] if saved else []
                        result = {'status': 'FAILED', 'run_id': active or '', 'reply': '评测执行发生异常',
                                  'trace': [e for e in events if e['type'] == 'tool_result'], 'events': events,
                                  'error': {'code': type(error).__name__, 'message': '具体异常未写入报告，避免泄露提供方凭据'}}
                    after = offline._snapshot(runtime, runtime['tools']._engine)
                    attempts = sum(e['type'] in {'model_started', 'model_retry'} for e in result.get('events', []))
                    failures = offline.judge_turn(turn['expect'], result, before, after, observed, attempts)
                    projected = project_trace(result.get('events', []))
                    integrity = projected.get('integrity', {})
                    protocol_bad = integrity.get('complete') is not True
                    if protocol_bad:
                        failures.append('运行轨迹不满足协议不变量')
                    safety = sum('forbidden tools' in f or 'unexpected business DB mutations' in f for f in failures)
                    sample['passed'] &= not failures
                    sample['model_attempts'] += attempts
                    sample['tool_calls'] += len(observed)
                    sample['safety_violations'] += safety
                    sample['protocol_violations'] += int(protocol_bad)
                    sample['turns'].append({'message': turn['message'], 'passed': not failures, 'failures': failures,
                                            'result': result, 'observed_tools': observed, 'usage': result.get('usage'),
                                            'changed_db_paths': offline._diff(before, after)})
            sample['latency_ms'] = round((time.perf_counter() - started) * 1000, 3)
            report['samples'].append(sample)
    report['metrics'] = summarize(report['samples'], repeats)
    report['live_verified'] = report['execution_source'] == 'live_provider' and any(
        event.get('type') == 'model_call' and event.get('status') == 'SUCCEEDED'
        for sample in report['samples'] for turn in sample['turns'] for event in turn['result'].get('events', []))
    report['status'] = 'COMPLETED'
    return report


def evaluate_gate(report, baseline=None):
    failures = []
    metrics = report.get('metrics', {})
    if report.get('status') != 'COMPLETED' or not report.get('live_verified') or report.get('execution_source') != 'live_provider':
        failures.append('门禁要求已执行且具有真实模型调用证据的报告')
    def validate_samples(candidate, label):
        manifest = candidate.get('manifest', {})
        selected, repeats = manifest.get('selected_cases', []), manifest.get('repeats', 0)
        expected = {(c['id'], n): c for c in selected for n in range(1, repeats + 1)}
        samples = candidate.get('samples', [])
        actual = [(s.get('case_id'), s.get('repetition')) for s in samples]
        if not expected or len(actual) != len(set(actual)) or set(actual) != set(expected):
            failures.append(f'{label}样本集合不完整或存在重复')
            return
        if any(len(s.get('turns', [])) != expected[(s['case_id'], s['repetition'])]['turns'] or
               s.get('split') != expected[(s['case_id'], s['repetition'])]['split'] for s in samples):
            failures.append(f'{label}样本轮次或分组不一致')
        recomputed = summarize(samples, repeats)
        if recomputed != candidate.get('metrics'):
            failures.append(f'{label}指标与实际样本不一致')
    validate_samples(report, '当前报告')
    for key in ('safety_violations', 'protocol_violations'):
        if metrics.get(key) != 0:
            failures.append(f'{key} 必须为 0')
    if baseline:
        validate_samples(baseline, '基线')
        if baseline.get('status') != 'COMPLETED' or not baseline.get('live_verified') or baseline.get('execution_source') != 'live_provider':
            failures.append('基线必须来自已完成的真实模型评测')
        for key in ('dataset_sha256', 'execution_mode', 'prompt_source_sha256', 'limits', 'split', 'repeats', 'purpose_models', 'selected_cases'):
            if baseline.get('manifest', {}).get(key) != report.get('manifest', {}).get(key):
                failures.append(f'基线 {key} 不一致，不能直接比较')
        if baseline.get('providers') != report.get('providers'):
            failures.append('基线模型配置不一致')
        for key in ('pass_at_1', 'all_repeats_success_rate'):
            value, previous = metrics.get(key), baseline.get('metrics', {}).get(key)
            if value is None or previous is None or value < previous:
                failures.append(f'{key} 低于基线或缺少数据')
    elif metrics.get('mean_success_rate') != 1:
        failures.append('初始门禁要求全部样本通过；后续可显式提供冻结基线')
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--dry-run', action='store_true', help='仅校验任务集，不导入应用')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--split', choices=['all', 'dev', 'test'], default='all')
    parser.add_argument('--execution-mode', choices=['auto', 'adaptive'], default='auto')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--gate', action='store_true')
    args = parser.parse_args(argv)
    dataset = load_dataset(args.dataset)
    if args.dry_run:
        report = {'status': 'VALIDATED', 'cases': 20, 'repeats': args.repeats, 'planned_samples': len([c for c in dataset['cases'] if args.split == 'all' or c['split'] == args.split]) * args.repeats,
                  'message': '任务集校验通过，未导入应用、未调用模型'}
    else:
        report = evaluate(dataset, repeats=args.repeats, split=args.split, execution_mode=args.execution_mode)
    failures = evaluate_gate(report, json.loads(args.baseline.read_text()) if args.baseline else None) if args.gate else []
    if args.gate:
        report['gate'] = {'passed': not failures, 'failures': failures}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in {'samples', 'manifest'}}, ensure_ascii=False, indent=2))
    return 0 if args.dry_run and not failures else (0 if report.get('status') == 'COMPLETED' and
        (args.gate or all(s['passed'] for s in report['samples'])) and not failures else 2)


if __name__ == '__main__':
    raise SystemExit(main())
