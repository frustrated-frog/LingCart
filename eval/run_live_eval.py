"""固定任务的真实模型评测入口；--dry-run 不导入应用、不调用模型。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, date, datetime
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import uuid

try:
    from eval import run_task_eval as offline
    from eval.experiment_contract import ExperimentSpec
except ModuleNotFoundError:
    import run_task_eval as offline
    from experiment_contract import ExperimentSpec

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(__file__).with_name('live_tasks.json')
KERNEL_BOUNDARIES = {
    'state_and_side_effects': {
        'src/runtime.py', 'src/direct_executor.py', 'src/tools.py', 'src/backend/db.py',
    },
    'persistence_and_recovery': {
        'src/runtime_state.py', 'src/persistence.py', 'src/durable.py', 'src/recovery.py',
        'src/session_control.py',
    },
    'model_call_protocol': {
        'src/model_gateway.py', 'src/llm.py', 'src/native_protocol.py', 'src/context_budget.py',
    },
    'catalog_semantic_binding': {
        'src/catalog_concepts.py', 'src/constraint_bindings.py',
    },
    'harness_assembly': {
        'src/harness/contracts.py', 'src/harness/profile.py', 'src/harness/registry.py',
    },
}
KERNEL_FILES = frozenset().union(*KERNEL_BOUNDARIES.values())
ADAPTIVE_EXPECT_KEYS = frozenset({
    'objectives', 'product_constraints', 'minimum_recommendations',
    'recommendation_targets', 'require_goal_coverage',
})
CONSTRAINT_KEYS = frozenset({'description', 'source_quote', 'field', 'op', 'value'})
CONSTRAINT_OPERATORS = frozenset({'eq', 'ne', 'contains', 'not_contains', 'lte', 'gte', 'in', 'not_in'})


def _report_json_default(value):
    """Serialize known protocol values without silently stringifying arbitrary objects."""
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    model_dump = getattr(value, 'model_dump', None)
    if callable(model_dump):
        return model_dump(mode='json')
    raise TypeError(f'Object of type {value.__class__.__name__} is not JSON serializable')


def _report_json_dumps(value, **kwargs):
    return json.dumps(value, ensure_ascii=False, default=_report_json_default, **kwargs)


def _validate_constraint(value, label):
    if not isinstance(value, dict) or set(value) != CONSTRAINT_KEYS:
        raise ValueError(f'{label} 必须完整声明 description/source_quote/field/op/value')
    if any(not isinstance(value[key], str) or not value[key].strip()
           for key in ('description', 'source_quote', 'field')):
        raise ValueError(f'{label} 的文字字段不能为空')
    if value['op'] not in CONSTRAINT_OPERATORS:
        raise ValueError(f'{label} 使用了未知约束操作符 {value["op"]!r}')


def _validate_adaptive_cases(dataset):
    if dataset.get('suite') != 'adaptive':
        return
    for case in dataset['cases']:
        label = case['id']
        expected = case.get('adaptive_expect')
        if not isinstance(expected, dict) or set(expected) - ADAPTIVE_EXPECT_KEYS:
            raise ValueError(f'{label}: adaptive_expect 缺失或包含未知字段')
        objectives = expected.get('objectives')
        if not isinstance(objectives, list) or not objectives or any(
                not isinstance(item, dict) or not isinstance(item.get('kind'), str) or not item['kind']
                for item in objectives):
            raise ValueError(f'{label}: objectives 必须声明至少一个带 kind 的目标')
        constraints = expected.get('product_constraints', [])
        targets = expected.get('recommendation_targets', [])
        if not isinstance(constraints, list) or not isinstance(targets, list) or any(
                not isinstance(target, list) or not target for target in targets):
            raise ValueError(f'{label}: 商品约束与独立推荐目标必须使用列表')
        for index, constraint in enumerate(constraints):
            _validate_constraint(constraint, f'{label}: product_constraints[{index}]')
        for target_index, target in enumerate(targets):
            for constraint_index, constraint in enumerate(target):
                _validate_constraint(constraint, f'{label}: recommendation_targets[{target_index}][{constraint_index}]')
        minimum = expected.get('minimum_recommendations', 1)
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
            raise ValueError(f'{label}: minimum_recommendations 必须是正整数')
        scenario = case.get('scenario')
        if scenario is None:
            continue
        if not isinstance(scenario, dict) or scenario.get('kind') not in {'steer', 'price_change', 'stock_change', 'empty_search'}:
            raise ValueError(f'{label}: scenario.kind 无效')
        if scenario['kind'] == 'empty_search':
            if set(scenario) != {'kind'}:
                raise ValueError(f'{label}: empty_search 只能声明 kind')
        elif scenario['kind'] == 'steer':
            if set(scenario) != {'kind', 'message'} or not isinstance(scenario['message'], str) or not scenario['message'].strip():
                raise ValueError(f'{label}: steer 场景必须且只能声明非空 message')
        elif (set(scenario) != {'kind', 'updates'} or not isinstance(scenario['updates'], dict)
              or not scenario['updates'] or set(scenario['updates']) - {'price', 'stock'}):
            raise ValueError(f'{label}: 环境变化必须且只能更新 price/stock')


def load_dataset(path=DATASET):
    dataset = offline.load_dataset(path)
    _validate_adaptive_cases(dataset)
    if any(case.get('autonomy_expect') for case in dataset['cases']):
        try:
            from eval.after_sales_autonomy_checks import validate_dataset as validate_after_sales_dataset
        except ModuleNotFoundError:
            from after_sales_autonomy_checks import validate_dataset as validate_after_sales_dataset
        validate_after_sales_dataset(dataset)
    expected = dataset.get('split_counts')
    if expected is None:
        expected = {'dev': 10, 'test': 10}
    if (not isinstance(expected, dict) or not expected
            or any(split not in {'dev', 'test'} or not isinstance(count, int) or isinstance(count, bool) or count < 1
                   for split, count in expected.items())):
        raise ValueError('实时评测 split_counts 必须声明正整数 dev/test 数量')
    actual = {split: sum(case.get('split') == split for case in dataset['cases'])
              for split in ('dev', 'test')}
    undeclared = [case['id'] for case in dataset['cases'] if case.get('split') not in expected]
    if undeclared or sum(expected.values()) != len(dataset['cases']) or any(
            actual.get(split, 0) != count for split, count in expected.items()):
        raise ValueError('实时评测任务数量必须与数据集冻结的 split_counts 一致')
    return dataset


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _files_fingerprint(paths):
    return fingerprint({str(path.relative_to(ROOT)): path.read_text() for path in paths if path.exists()})


def _source_files_manifest(paths):
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.exists()}


def _repo_relative(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def _prompt_fingerprint(path):
    """Fingerprint every literal that can become prompt text, including f-strings."""
    import ast

    values = [node.value for node in ast.walk(ast.parse(path.read_text()))
              if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    return fingerprint(values)


def environment_manifest(dataset, execution_mode, dataset_path=DATASET):
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    prompt_files = sorted(set(['src/orchestrator.py', 'src/dialogue_understanding.py', 'src/adaptive.py', 'src/liveplan.py',
                               'src/adaptive_decision.py', 'src/adaptive_goals.py', 'src/request_audit.py',
                               'src/catalog_concepts.py', 'src/constraint_bindings.py', 'src/native_protocol.py'] +
                             [str(p.relative_to(ROOT)) for p in (ROOT / 'src/agents').glob('*.py')]))
    evaluator_files = [ROOT / name for name in ('eval/run_live_eval.py', 'eval/run_task_eval.py',
                                                'eval/adaptive_checks.py', 'eval/experiment_contract.py')]
    business_files = [ROOT / name for name in ('src/tools.py', 'src/backend/db.py', 'src/backend/seed.py')]
    source_files = sorted([*(ROOT / 'src').rglob('*.py'), *(ROOT / 'eval').glob('*.py'),
                           *(ROOT / 'configs/harness').glob('*.json')])
    file_hashes = _source_files_manifest(source_files)
    dependency_lock = ROOT / 'requirements.lock'
    boundary_hashes = {name: fingerprint({path: file_hashes.get(path) for path in sorted(paths)})
                       for name, paths in KERNEL_BOUNDARIES.items()}
    return {'source_revision': revision, 'dataset_path': _repo_relative(dataset_path),
            'dataset_sha256': fingerprint(dataset), 'execution_mode': execution_mode,
            'source_tree_sha256': fingerprint(file_hashes), 'source_files_sha256': file_hashes,
            'kernel_source_sha256': fingerprint({name: file_hashes.get(name) for name in sorted(KERNEL_FILES)}),
            'kernel_boundaries_sha256': boundary_hashes,
            'prompt_source_sha256': {name: _prompt_fingerprint(ROOT / name)
                for name in prompt_files if (ROOT / name).exists()},
            'evaluation_protocol_sha256': _files_fingerprint(evaluator_files),
            'business_protocol_sha256': _files_fingerprint(business_files),
            'business_environment': 'fresh_demo_sqlite_per_sample', 'pricing_verified': False,
            'python_runtime': {'implementation': platform.python_implementation(),
                               'version': platform.python_version()},
            'dependency_lock_sha256': (hashlib.sha256(dependency_lock.read_bytes()).hexdigest()
                                       if dependency_lock.exists() else None),
            'container_image_digest': os.getenv('LINGCART_IMAGE_DIGEST')}


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
    events = [event for sample in samples for turn in sample.get('turns', [])
              for event in (turn.get('result') or {}).get('events', [])]
    drifted_samples = [sample for sample in samples if any(
        event.get('type') == 'trajectory_drift_detected'
        for turn in sample.get('turns', []) for event in (turn.get('result') or {}).get('events', []))]
    autonomy = [sample.get('autonomy') for sample in samples if sample.get('autonomy')]
    resolvable = [item for item in autonomy if item.get('expected_autonomous_actions', 0) > 0]
    approval = [item for item in autonomy if item.get('approval_boundary_expected')]
    return {'tasks': len(groups), 'samples': len(samples), 'repeats': repeats,
            'pass_at_1': sum(items[0]['passed'] for items in groups.values()) / len(groups) if groups else None,
            'mean_success_rate': sum(s['passed'] for s in samples) / len(samples) if samples else None,
            'all_repeats_success_rate': sum(len(items) == repeats and all(s['passed'] for s in items) for items in groups.values()) / len(groups) if groups else None,
            'model_attempts': sum(s['model_attempts'] for s in samples),
            'tool_calls': sum(s['tool_calls'] for s in samples),
            'safety_violations': sum(s['safety_violations'] for s in samples),
            'protocol_violations': sum(s['protocol_violations'] for s in samples),
            'trajectory_control': {
                'drift_detections': sum(event.get('type') == 'trajectory_drift_detected' for event in events),
                'advisor_calls': sum(event.get('type') == 'model_call'
                                     and event.get('purpose') == 'liveplan_advisor' for event in events),
                'intervention_activations': sum(event.get('type') == 'intervention_applied'
                                                and event.get('outcome') == 'ACTIVATED' for event in events),
                'action_rejections': sum(event.get('type') == 'intervention_applied'
                                         and event.get('outcome') == 'ACTION_REJECTED' for event in events),
                'drifted_samples': len(drifted_samples),
                'recovered_samples': sum(sample.get('passed') is True for sample in drifted_samples),
                'recovery_rate': (sum(sample.get('passed') is True for sample in drifted_samples)
                                  / len(drifted_samples) if drifted_samples else 0),
            },
            'policy_autonomy': {
                'case_resolution_rate': (sum(item.get('resolved') is True for item in autonomy)
                                         / len(autonomy) if autonomy else 0),
                'autonomous_resolution_rate': (sum(item.get('resolved') is True for item in resolvable)
                                               / len(resolvable) if resolvable else 0),
                'avoidable_pause_rate': (sum(item.get('avoidable_pause') is True for item in resolvable)
                                         / len(resolvable) if resolvable else 0),
                'approval_boundary_rate': (sum(item.get('approval_boundary_preserved') is True for item in approval)
                                           / len(approval) if approval else 1),
                'unauthorized_write_count': sum(int(item.get('unauthorized_write_count', 0)) for item in autonomy),
                'duplicate_action_count': sum(int(item.get('duplicate_action_count', 0)) for item in autonomy),
            },
            'latency_ms': {'mean': statistics.mean(latencies) if latencies else None,
                           'p50': offline._percentile(latencies, .5) if latencies else None,
                           'p95': offline._percentile(latencies, .95) if latencies else None},
            'tokens': tokens, 'monetary_cost': None}


def evaluate(dataset, *, repeats=3, split='all', case_ids=(), execution_mode='auto', harness_profile=None,
             experiment_spec=None, dataset_path=DATASET, runner=None, readiness=None,
             max_model_calls_per_run=6, max_prompt_tokens_per_run=None,
             max_completion_tokens_per_run=None, max_seconds_per_run=None,
             max_provider_attempts_per_call=None, on_sample=None):
    """runner 注入只用于基础设施测试，报告会明确标识为 test_harness。"""
    injected_runner = runner is not None
    if not 1 <= repeats <= 10:
        raise ValueError('重复次数必须在 1 至 10 之间')
    if not isinstance(max_model_calls_per_run, int) or isinstance(max_model_calls_per_run, bool) or max_model_calls_per_run < 1:
        raise ValueError('每个 Run 的模型调用上限必须是正整数')
    suite = dataset.get('suite', 'support')
    if suite == 'adaptive' and execution_mode != 'adaptive':
        raise ValueError('自主专项任务必须使用 adaptive 入口')
    available_ids = {case['id'] for case in dataset['cases']}
    unknown_ids = set(case_ids) - available_ids
    if unknown_ids:
        raise ValueError('不存在的评测案例：' + '、'.join(sorted(unknown_ids)))
    if experiment_spec is not None and case_ids:
        raise ValueError('正式实验契约必须运行完整 evaluation split，不能只运行案例子集')
    # 在隔离数据库的临时环境之外加载配置，避免 patch.dict 恢复时丢弃密钥。
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    runtime = offline._runtime()
    from src.durable import run_conversation
    from src.llm import get_model_status
    from dataclasses import replace
    from src.runtime import BudgetLimits
    from src.trace_projector import project_trace
    report = {'schema_version': 2, 'generated_at': datetime.now(UTC).isoformat(),
              'execution_source': 'test_harness' if injected_runner else 'live_provider',
              'manifest': {**environment_manifest(dataset, execution_mode, dataset_path),
                           'suite': suite, 'split': split, 'repeats': repeats}, 'samples': []}
    if not injected_runner:
        from src.harness import get_runner
        profile_id = harness_profile or ('adaptive-v1' if execution_mode == 'adaptive' else 'baseline-v1')
        runner = get_runner(profile_id)
        if runner.execution_mode != execution_mode:
            raise ValueError(f'Harness Profile {profile_id} 的执行模式不是 {execution_mode}')
        report['manifest']['harness_profile'] = runner.provenance()
    else:
        selected = getattr(runner, 'harness_runner', runner)
        report['manifest']['harness_profile'] = getattr(selected, 'provenance', lambda: {
            'profile_id': 'injected-test-runner', 'profile_version': 'unversioned', 'profile_sha256': None})()
    if experiment_spec is not None:
        experiment_spec = (experiment_spec if isinstance(experiment_spec, ExperimentSpec) else
                           ExperimentSpec.model_validate_json(json.dumps(experiment_spec, ensure_ascii=False)))
        actual_profile = report['manifest']['harness_profile'].get('profile_id')
        if actual_profile != experiment_spec.candidate_profile:
            raise ValueError(f'实验候选要求 {experiment_spec.candidate_profile}，实际运行 {actual_profile}')
        if report['manifest']['dataset_path'] != experiment_spec.dataset or suite != experiment_spec.suite:
            raise ValueError('实验任务集或 suite 与契约不一致')
        if repeats != experiment_spec.repeats:
            raise ValueError(f'实验必须按契约重复 {experiment_spec.repeats} 次')
        if split != experiment_spec.evaluation_split:
            raise ValueError(f'实验结论必须使用未参与调参的 {experiment_spec.evaluation_split} 分组')
        report['manifest']['experiment_spec'] = experiment_spec.model_dump(mode='json')
    else:
        report['manifest']['experiment_spec'] = None
    status = readiness if readiness is not None else get_model_status()
    report['providers'] = [{key: p.get(key) for key in ('provider', 'model', 'slot', 'custom_base_url', 'thinking_mode', 'reasoning_effort', 'strict_tools')} for p in status.get('providers', [])]
    from src.model_gateway import load_gateway_config, ModelGatewayConfigurationError
    report['manifest']['purpose_models'] = {}
    for purpose in ('router', 'dialogue_understanding', 'evidence_verification', 'goal', 'planner', 'agent',
                    'audit', 'final_response', 'liveplan_advisor'):
        try:
            config = load_gateway_config(purpose=purpose)
            report['manifest']['purpose_models'][purpose] = [{'provider': p.provider, 'model': p.model,
                'thinking_mode': p.thinking_mode, 'reasoning_effort': p.reasoning_effort, 'strict_tools': p.strict_tools} for p in config.providers]
            for item in report['manifest']['purpose_models'][purpose]:
                item['gateway_max_attempts'] = config.max_attempts
        except ModelGatewayConfigurationError:
            report['manifest']['purpose_models'][purpose] = None
    if not status.get('configured') and not injected_runner:
        report.update(status='NOT_RUN', reason='MODEL_NOT_CONFIGURED', message='尚未配置模型，未执行真实评测', metrics=summarize([], repeats))
        return report
    limits = BudgetLimits.adaptive() if suite == 'adaptive' else BudgetLimits(max_seconds=45)
    limit_updates = {'max_model_calls': min(limits.max_model_calls, max_model_calls_per_run)}
    for name, value in (
        ('max_prompt_tokens', max_prompt_tokens_per_run),
        ('max_completion_tokens', max_completion_tokens_per_run),
        ('max_seconds', max_seconds_per_run),
    ):
        if value is not None:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} 必须是正数')
            limit_updates[name] = min(getattr(limits, name), value)
    limits = replace(limits, **limit_updates)
    report['manifest']['limits'] = {**limits.__dict__,
        'max_provider_attempts_per_call': max_provider_attempts_per_call}
    selected = [case for case in dataset['cases']
                if (split == 'all' or case['split'] == split) and (not case_ids or case['id'] in case_ids)]
    if not selected:
        raise ValueError('所选 split 与案例过滤条件没有可执行样本')
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
                            sample_runner.harness_runner = getattr(runner, 'harness_runner', runner)
                            sample_runner.execution_mode = getattr(runner, 'execution_mode', execution_mode)
                            from contextlib import nullcontext
                            from eval.adaptive_checks import scenario
                            from src.llm import model_attempt_limit
                            with model_attempt_limit(max_provider_attempts_per_call):
                                with (scenario(case, session_id) if suite == 'adaptive' else nullcontext({})) as environment:
                                    result = run_conversation(turn['message'], session_id, request_id=str(turn_index), limits=limits, runner=sample_runner)
                            if suite == 'adaptive':
                                result['evaluation_environment'] = environment
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
                    environment = result.get('evaluation_environment') or {}
                    checked_after = after
                    if environment.get('environment_paths'):
                        # 环境故障注入有单独记录；Agent的其它业务变化仍由原门禁核验。
                        from copy import deepcopy
                        checked_after = deepcopy(after)
                        for path in environment['environment_paths']:
                            _, identity, field = path.split('.')
                            checked_after['products'][identity][field] = before['products'][identity][field]
                    from eval.adaptive_checks import observed_with_faults
                    checked_observed = observed_with_faults(observed, environment)
                    failures = offline.judge_turn(turn['expect'], result, before, checked_after, checked_observed, attempts)
                    if case.get('autonomy_expect'):
                        try:
                            from eval.after_sales_autonomy_checks import judge
                        except ModuleNotFoundError:
                            from after_sales_autonomy_checks import judge

                        autonomy, autonomy_failures = judge(
                            runtime, runtime['tools']._engine, case['autonomy_expect'], result, checked_observed,
                        )
                        sample['autonomy'] = autonomy
                        failures += autonomy_failures
                    if suite == 'adaptive':
                        from eval.adaptive_checks import judge
                        effective = turn['message'] + ('\n运行中补充要求：\n' + case['scenario']['message'] if case.get('scenario', {}).get('kind') == 'steer' else '')
                        failures += judge({**case, '_effective_request': effective}, result, after)
                        if case.get('scenario') and not environment.get('fired'):
                            failures.append('专项环境扰动没有实际触发')
                        sample['adaptive_verified'] = sample.get('adaptive_verified', True) and result.get('autonomous_executed', False)
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
            if on_sample is not None:
                on_sample(sample)
    report['metrics'] = summarize(report['samples'], repeats)
    if suite == 'adaptive':
        report['adaptive_verified_samples'] = sum(s.get('adaptive_verified') is True for s in report['samples'])
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
    if report.get('manifest', {}).get('suite') == 'adaptive' and any(not s.get('adaptive_verified') for s in report.get('samples', [])):
        failures.append('自主专项不能包含未执行模型自主决策的样本')
    for key in ('safety_violations', 'protocol_violations'):
        if metrics.get(key) != 0:
            failures.append(f'{key} 必须为 0')
    if baseline:
        validate_samples(baseline, '基线')
        if baseline.get('status') != 'COMPLETED' or not baseline.get('live_verified') or baseline.get('execution_source') != 'live_provider':
            failures.append('基线必须来自已完成的真实模型评测')
        for key in ('dataset_path', 'dataset_sha256', 'suite', 'evaluation_protocol_sha256', 'business_protocol_sha256',
                    'business_environment', 'limits', 'split', 'repeats', 'purpose_models', 'selected_cases',
                    'python_runtime', 'dependency_lock_sha256', 'container_image_digest'):
            if baseline.get('manifest', {}).get(key) != report.get('manifest', {}).get(key):
                failures.append(f'基线 {key} 不一致，不能直接比较')
        if baseline.get('providers') != report.get('providers'):
            failures.append('基线模型配置不一致')
        change_keys = ('execution_mode', 'prompt_source_sha256', 'source_tree_sha256', 'harness_profile')
        changed = [key for key in change_keys
                   if baseline.get('manifest', {}).get(key) != report.get('manifest', {}).get(key)]
        experiment_spec = report.get('manifest', {}).get('experiment_spec')
        if changed and not experiment_spec:
            failures.append('候选实现改变了 ' + '、'.join(changed) + '，但没有结构化实验契约')
        if experiment_spec:
            failures.extend(_validate_experiment_contract(report, baseline, experiment_spec))
        else:
            for key in ('pass_at_1', 'all_repeats_success_rate'):
                value, previous = metrics.get(key), baseline.get('metrics', {}).get(key)
                if value is None or previous is None or value < previous:
                    failures.append(f'{key} 低于基线或缺少数据')
    else:
        if report.get('manifest', {}).get('experiment_spec'):
            failures.append('论文对照 Gate 必须同时提供冻结基线报告和实验契约')
        if metrics.get('mean_success_rate') != 1:
            failures.append('单版本质量 Gate 要求全部样本通过')
    return failures


def _metric_value(metrics, name):
    value = metrics
    for part in name.split('.'):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _profile_parts(manifest):
    profile = manifest.get('harness_profile') or {}
    mechanisms = {item.get('slot'): item for item in profile.get('mechanisms', []) if item.get('slot')}
    return profile, mechanisms


def _validate_experiment_contract(report, baseline, payload):
    from pydantic import ValidationError
    try:
        spec = ExperimentSpec.model_validate_json(json.dumps(payload, ensure_ascii=False))
    except ValidationError as error:
        return ['实验契约无效：' + '; '.join('.'.join(map(str, item['loc'])) + ' ' + item['msg']
                                            for item in error.errors(include_url=False)[:6])]
    failures = []
    current_manifest, baseline_manifest = report.get('manifest', {}), baseline.get('manifest', {})
    current_profile, current_mechanisms = _profile_parts(current_manifest)
    baseline_profile, baseline_mechanisms = _profile_parts(baseline_manifest)
    if baseline_profile.get('profile_id') != spec.baseline_profile:
        failures.append('实验契约的 baseline_profile 与基线报告不一致')
    if current_profile.get('profile_id') != spec.candidate_profile:
        failures.append('实验契约的 candidate_profile 与候选报告不一致')
    if current_manifest.get('split') != spec.evaluation_split or baseline_manifest.get('split') != spec.evaluation_split:
        failures.append('论文实验必须在契约声明的未调参分组上对照')
    if (current_manifest.get('dataset_path') != spec.dataset or
            baseline_manifest.get('dataset_path') != spec.dataset or
            current_manifest.get('suite') != spec.suite or baseline_manifest.get('suite') != spec.suite):
        failures.append('实验任务集或 suite 与契约不一致')
    if current_manifest.get('repeats') != spec.repeats or baseline_manifest.get('repeats') != spec.repeats:
        failures.append('实验重复次数与契约不一致')
    if not current_manifest.get('dependency_lock_sha256') or not baseline_manifest.get('dependency_lock_sha256'):
        failures.append('论文对照报告必须记录可重建的依赖锁指纹')

    declarations = list(spec.changes)
    changed_slots = {slot for slot in set(current_mechanisms) | set(baseline_mechanisms)
                     if current_mechanisms.get(slot) != baseline_mechanisms.get(slot)}
    mechanism_declarations = {(item.slot, item.implementation_id): item for item in declarations
                              if item.kind == 'mechanism'}
    declared_slots = set(mechanism_declarations)
    for slot in changed_slots:
        candidate = current_mechanisms.get(slot) or {}
        if (slot, candidate.get('id')) not in declared_slots:
            failures.append(f'机制变更未声明：{slot}/{candidate.get("id")}')
    for slot, implementation_id in declared_slots:
        candidate = current_mechanisms.get(slot) or {}
        if slot not in changed_slots or candidate.get('id') != implementation_id:
            failures.append(f'契约声明的机制未成为实际差异：{slot}/{implementation_id}')
        elif not set(candidate.get('source_files') or ()) <= set(mechanism_declarations[(slot, implementation_id)].paths):
            failures.append(f'机制源码未完整纳入变更范围：{slot}/{implementation_id}')
        else:
            missing = []
            for sample in report.get('samples', []):
                events = [event for turn in sample.get('turns', [])
                          for event in (turn.get('result') or {}).get('events', [])]
                if not any(event.get('type') == 'mechanism_applied' and event.get('slot') == slot
                           and event.get('mechanism_id') == implementation_id for event in events):
                    missing.append(f"{sample.get('case_id')}#{sample.get('repetition')}")
            if missing:
                failures.append(f'候选机制没有在全部样本中产生执行证据：{slot}/{implementation_id} '
                                + '、'.join(missing[:5]))

    engine_changed = current_profile.get('engine') != baseline_profile.get('engine')
    engine_declarations = {item.implementation_id: item for item in declarations if item.kind == 'engine'}
    declared_engines = set(engine_declarations)
    current_engine = (current_profile.get('engine') or {}).get('id')
    if engine_changed and current_engine not in declared_engines:
        failures.append(f'引擎变更未声明：{current_engine}')
    if not engine_changed and declared_engines:
        failures.append('契约声明了引擎变更，但实际引擎相同')
    if engine_changed and current_engine in engine_declarations:
        current_sources = set((current_profile.get('engine') or {}).get('source_files') or ())
        if not current_sources <= set(engine_declarations[current_engine].paths):
            failures.append(f'引擎源码未完整纳入变更范围：{current_engine}')

    base_files = baseline_manifest.get('source_files_sha256')
    current_files = current_manifest.get('source_files_sha256')
    if not isinstance(base_files, dict) or not isinstance(current_files, dict):
        failures.append('结构化实验缺少逐文件源码指纹')
        changed_files = set()
    else:
        for label, manifest, file_hashes in (('基线', baseline_manifest, base_files),
                                             ('候选', current_manifest, current_files)):
            if manifest.get('source_tree_sha256') != fingerprint(file_hashes):
                failures.append(f'{label}源码树指纹与逐文件指纹不一致')
            expected_kernel = fingerprint({name: file_hashes.get(name) for name in sorted(KERNEL_FILES)})
            if manifest.get('kernel_source_sha256') != expected_kernel:
                failures.append(f'{label} Stable Kernel 指纹不一致')
            expected_boundaries = {name: fingerprint({path: file_hashes.get(path) for path in sorted(paths)})
                                   for name, paths in KERNEL_BOUNDARIES.items()}
            if manifest.get('kernel_boundaries_sha256') != expected_boundaries:
                failures.append(f'{label} Stable Kernel 职责边界指纹不一致')
        changed_files = {name for name in set(base_files) | set(current_files)
                         if base_files.get(name) != current_files.get(name)}
        allowed_files = {path for item in declarations for path in item.paths}
        unapproved = changed_files - allowed_files
        if unapproved:
            failures.append('存在未声明的源码变更：' + '、'.join(sorted(unapproved)))
        changed_kernel = changed_files & KERNEL_FILES
        runtime_files = {path for item in declarations if item.kind == 'runtime' for path in item.paths}
        if changed_kernel and (not spec.compatibility_review or not changed_kernel <= runtime_files):
            boundaries = [name for name, paths in KERNEL_BOUNDARIES.items() if changed_kernel & paths]
            failures.append('Stable Kernel 变更缺少独立兼容性审查：' +
                            '、'.join(sorted(changed_kernel)) + f' （责任域：{"/".join(boundaries)}）')

    changed_prompts = {name for name in set(baseline_manifest.get('prompt_source_sha256', {})) |
                       set(current_manifest.get('prompt_source_sha256', {}))
                       if baseline_manifest.get('prompt_source_sha256', {}).get(name) !=
                       current_manifest.get('prompt_source_sha256', {}).get(name)}
    prompt_files = {path for item in declarations if item.kind == 'prompt' for path in item.paths}
    if changed_prompts - prompt_files:
        failures.append('Prompt 变更未按文件声明：' + '、'.join(sorted(changed_prompts - prompt_files)))

    primary_names = {rule.name for rule in spec.primary_metrics}
    for rule in (*spec.primary_metrics, *spec.guard_metrics):
        current = _metric_value(report.get('metrics', {}), rule.name)
        previous = _metric_value(baseline.get('metrics', {}), rule.name)
        if current is None or previous is None:
            failures.append(f'实验指标 {rule.name} 缺少可比较数值')
            continue
        if rule.minimum_value is not None and current < rule.minimum_value:
            failures.append(
                f'实验指标 {rule.name} 当前值 {current:g} 低于绝对下限 {rule.minimum_value:g}'
            )
        if rule.maximum_value is not None and current > rule.maximum_value:
            failures.append(
                f'实验指标 {rule.name} 当前值 {current:g} 高于绝对上限 {rule.maximum_value:g}'
            )
        improvement = current - previous if rule.objective == 'maximize' else previous - current
        if improvement < -rule.max_regression:
            failures.append(f'实验指标 {rule.name} 退化 {abs(improvement):g}，超过允许值 {rule.max_regression:g}')
        if improvement < rule.min_improvement:
            label = '主指标' if rule.name in primary_names else '保护指标'
            failures.append(f'{label} {rule.name} 改善 {improvement:g}，未达到 {rule.min_improvement:g}')
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--dry-run', action='store_true', help='仅校验任务集，不导入应用')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--split', choices=['all', 'dev', 'test'], default='all')
    parser.add_argument('--case', action='append', default=[], help='仅执行指定案例；可重复传入，仅用于 Smoke')
    parser.add_argument('--max-samples', type=int, default=2, help='实时执行允许的最大样本数，默认 2')
    parser.add_argument('--max-model-calls-per-run', type=int, default=6,
                        help='每个 Run 最多允许的模型调用数；只能收紧 Runtime 预算，默认 6')
    parser.add_argument('--execution-mode', choices=['auto', 'adaptive'], default='auto')
    parser.add_argument('--harness-profile', help='使用已注册的 Harness Profile，例如 baseline-v1')
    parser.add_argument('--experiment-spec', type=Path, help='机器可校验的 JSON 实验契约')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--gate', action='store_true')
    args = parser.parse_args(argv)
    dataset = load_dataset(args.dataset)
    experiment_spec = ExperimentSpec.from_path(args.experiment_spec) if args.experiment_spec else None
    available_ids = {case['id'] for case in dataset['cases']}
    unknown_ids = set(args.case) - available_ids
    if unknown_ids:
        parser.error('不存在的评测案例：' + '、'.join(sorted(unknown_ids)))
    selected = [case for case in dataset['cases']
                if (args.split == 'all' or case['split'] == args.split)
                and (not args.case or case['id'] in args.case)]
    planned_samples = len(selected) * args.repeats
    if not selected:
        parser.error('所选 split 与案例过滤条件没有可执行样本')
    if args.max_samples < 1:
        parser.error('--max-samples 必须大于 0')
    if args.max_model_calls_per_run < 1:
        parser.error('--max-model-calls-per-run 必须大于 0')
    if args.gate and experiment_spec and args.baseline is None:
        parser.error('论文对照 Gate 必须通过 --baseline 提供冻结基线报告')
    if experiment_spec and args.case:
        parser.error('正式实验契约必须运行完整 evaluation split，不能使用 --case')
    if not args.dry_run and planned_samples > args.max_samples:
        parser.error(f'计划执行 {planned_samples} 个实时样本，超过 --max-samples={args.max_samples}；请显式提高预算')
    if args.dry_run:
        if dataset.get('suite') == 'adaptive' and args.execution_mode != 'adaptive':
            parser.error('自主专项任务必须使用 adaptive')
        if experiment_spec and (args.split != experiment_spec.evaluation_split or
                                args.harness_profile != experiment_spec.candidate_profile or
                                _repo_relative(args.dataset) != experiment_spec.dataset or
                                dataset.get('suite', 'support') != experiment_spec.suite or
                                args.repeats != experiment_spec.repeats):
            parser.error('实验 dry-run 的候选 Profile、任务集、suite、重复次数和评测分组必须与契约一致')
        report = {'status': 'VALIDATED', 'suite': dataset.get('suite', 'support'), 'execution_mode': args.execution_mode,
                  'experiment_id': experiment_spec.experiment_id if experiment_spec else None,
                  'cases': len(selected), 'dataset_cases': len(dataset['cases']),
                  'repeats': args.repeats, 'planned_samples': planned_samples,
                  'message': '任务集校验通过，未导入应用、未调用模型'}
    else:
        report = evaluate(dataset, repeats=args.repeats, split=args.split, case_ids=tuple(args.case),
                          execution_mode=args.execution_mode,
                          harness_profile=args.harness_profile, experiment_spec=experiment_spec,
                          dataset_path=args.dataset,
                          max_model_calls_per_run=args.max_model_calls_per_run)
    failures = evaluate_gate(report, json.loads(args.baseline.read_text()) if args.baseline else None) if args.gate else []
    if args.gate:
        report['gate'] = {'passed': not failures, 'failures': failures}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(_report_json_dumps(report, indent=2) + '\n')
    print(_report_json_dumps({key: value for key, value in report.items() if key not in {'samples', 'manifest'}}, indent=2))
    return 0 if args.dry_run and not failures else (0 if report.get('status') == 'COMPLETED' and
        (args.gate or all(s['passed'] for s in report['samples'])) and not failures else 2)


if __name__ == '__main__':
    raise SystemExit(main())
