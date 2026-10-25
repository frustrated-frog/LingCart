import copy
import json
from datetime import UTC, datetime

import pytest

from eval import run_live_eval as live


def test_live_report_serializer_normalizes_temporal_tool_values():
    encoded = live._report_json_dumps({
        'observed_at': datetime(2026, 10, 26, 2, 48, tzinfo=UTC),
    })

    assert json.loads(encoded) == {'observed_at': '2026-10-26T02:48:00+00:00'}
    with pytest.raises(TypeError, match='not JSON serializable'):
        live._report_json_dumps({'unexpected': object()})


def test_live_dataset_has_frozen_split_and_no_scripted_model():
    dataset = live.load_dataset()
    assert len(dataset['cases']) == 20
    assert {c['split'] for c in dataset['cases']} == {'dev', 'test'}
    assert all('model' not in case for case in dataset['cases'])


def test_live_dry_run_does_not_import_application(monkeypatch, capsys):
    monkeypatch.setattr(live.offline, '_runtime', lambda: (_ for _ in ()).throw(AssertionError('不能导入应用')))
    assert live.main(['--dry-run']) == 0
    assert '未调用模型' in capsys.readouterr().out


def test_live_case_filter_and_sample_budget_prevent_accidental_batch_calls(monkeypatch, capsys):
    first = live.load_dataset()['cases'][0]['id']
    assert live.main(['--dry-run', '--case', first, '--repeats', '2']) == 0
    assert json.loads(capsys.readouterr().out)['planned_samples'] == 2

    monkeypatch.setattr(live, 'evaluate', lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError('超过样本预算时不应启动实时评测')))
    with pytest.raises(SystemExit):
        live.main(['--repeats', '1'])


def test_live_model_call_budget_can_only_reduce_runtime_limit():
    dataset = live.load_dataset()
    dataset['cases'] = [dataset['cases'][0]]

    def runner(_message, **kwargs):
        return {'status': 'FAILED', 'reply': '预算测试', 'trace': [], 'events': [],
                'error': {'code': 'TEST', 'message': '预算测试'}}

    report = live.evaluate(dataset, repeats=1, runner=runner, readiness={'configured': False},
                           max_model_calls_per_run=99)

    assert report['manifest']['limits']['max_model_calls'] == 8


def test_unconfigured_provider_is_not_a_successful_live_baseline():
    report = live.evaluate(live.load_dataset(), readiness={'configured': False})
    assert report['status'] == 'NOT_RUN'
    assert report['samples'] == []
    assert live.evaluate_gate(report)


def test_dotenv_is_loaded_before_temporary_runtime_environment(monkeypatch):
    import os
    import dotenv
    from unittest.mock import patch

    monkeypatch.delenv('LLM_MODEL', raising=False)
    def load(path):
        assert path == live.ROOT / '.env'
        monkeypatch.setenv('LLM_MODEL', 'test-flash')

    original = live.offline._runtime
    def isolated_runtime():
        assert os.environ['LLM_MODEL'] == 'test-flash'
        with patch.dict(os.environ, {'SUPPORT_AGENT_DB_URL': 'sqlite:///:memory:'}):
            runtime = original()
        assert os.environ['LLM_MODEL'] == 'test-flash'
        return runtime

    monkeypatch.setattr(dotenv, 'load_dotenv', load)
    monkeypatch.setattr(live.offline, '_runtime', isolated_runtime)
    report = live.evaluate(live.load_dataset(), readiness={'configured': False})
    assert report['status'] == 'NOT_RUN'
    assert os.environ['LLM_MODEL'] == 'test-flash'


def test_stability_requires_all_repetitions_and_preserves_unknown_usage():
    def sample(passed):
        return dict(case_id='L01', passed=passed, latency_ms=1, model_attempts=1, tool_calls=2,
                    safety_violations=0, protocol_violations=0, turns=[{'usage': {}}])
    metrics = live.summarize([sample(True), sample(False), sample(True)], 3)
    assert metrics['pass_at_1'] == 1
    assert metrics['mean_success_rate'] == 2 / 3
    assert metrics['all_repeats_success_rate'] == 0
    assert metrics['tokens']['total_tokens'] is None
    assert metrics['monetary_cost'] is None


def test_summary_projects_liveplan_control_evidence_without_inventing_recovery():
    sample = dict(
        case_id='L01', passed=True, latency_ms=1, model_attempts=1, tool_calls=0,
        safety_violations=0, protocol_violations=0,
        turns=[{'usage': {}, 'result': {'events': [
            {'type': 'trajectory_drift_detected'},
            {'type': 'model_call', 'purpose': 'liveplan_advisor'},
            {'type': 'intervention_applied', 'outcome': 'ACTIVATED'},
            {'type': 'intervention_applied', 'outcome': 'ACTION_REJECTED'},
        ]}}],
    )

    control = live.summarize([sample], 1)['trajectory_control']

    assert control == {
        'drift_detections': 1,
        'advisor_calls': 1,
        'intervention_activations': 1,
        'action_rejections': 1,
        'drifted_samples': 1,
        'recovered_samples': 1,
        'recovery_rate': 1,
    }

    no_drift = live.summarize([{**sample, 'turns': [{'usage': {}, 'result': {'events': []}}]}], 1)
    assert no_drift['trajectory_control']['recovery_rate'] == 0
    assert no_drift['trajectory_control']['recovered_samples'] == 0


def test_incompatible_model_or_dataset_cannot_be_compared():
    baseline = {'status': 'COMPLETED', 'live_verified': True, 'manifest': {'dataset_sha256': 'a'},
                'providers': [{'model': 'one'}], 'metrics': {'safety_violations': 0, 'protocol_violations': 0,
                                                         'pass_at_1': 1, 'all_repeats_success_rate': 1}}
    candidate = copy.deepcopy(baseline)
    candidate['providers'] = [{'model': 'two'}]
    assert '基线模型配置不一致' in live.evaluate_gate(candidate, baseline)
    candidate = copy.deepcopy(baseline)
    candidate['live_verified'] = False
    assert live.evaluate_gate(candidate, baseline)


def _comparable_live_report(profile_id='adaptive-v1', *, model_attempts=1):
    from src.harness import get_runner

    provenance = get_runner(profile_id).provenance()
    context_id = next((item['id'] for item in provenance['mechanisms'] if item['slot'] == 'context'), None)
    sample = dict(case_id='L01', split='test', repetition=1, passed=True, latency_ms=1,
                  model_attempts=model_attempts, tool_calls=0, safety_violations=0, protocol_violations=0,
                  adaptive_verified=True, turns=[{'usage': {}, 'result': {'events': [
                      {'type': 'mechanism_applied', 'slot': 'context', 'mechanism_id': context_id},
                  ]}}])
    source_files = {'src/context_engine.py': 'same', 'src/runtime.py': 'same'}
    kernel_sha = live.fingerprint({name: source_files.get(name) for name in sorted(live.KERNEL_FILES)})
    boundary_sha = {name: live.fingerprint({path: source_files.get(path) for path in sorted(paths)})
                    for name, paths in live.KERNEL_BOUNDARIES.items()}
    return {'status': 'COMPLETED', 'live_verified': True, 'execution_source': 'live_provider',
            'manifest': {'dataset_path': 'eval/adaptive_tasks.json', 'dataset_sha256': 'dataset',
                         'suite': 'adaptive', 'execution_mode': 'adaptive',
                         'evaluation_protocol_sha256': 'evaluator', 'business_protocol_sha256': 'business',
                         'business_environment': 'fresh_demo_sqlite_per_sample',
                         'source_tree_sha256': live.fingerprint(source_files),
                         'source_files_sha256': source_files, 'kernel_source_sha256': kernel_sha,
                         'kernel_boundaries_sha256': boundary_sha,
                         'prompt_source_sha256': {'src/adaptive_decision.py': 'a'},
                         'harness_profile': provenance,
                         'limits': {'max_seconds': 45}, 'split': 'test', 'repeats': 1,
                         'purpose_models': {'agent': [{'provider': 'test', 'model': 'same'}]},
                         'python_runtime': {'implementation': 'CPython', 'version': '3.11.0'},
                         'dependency_lock_sha256': 'lock', 'container_image_digest': None,
                         'selected_cases': [{'id': 'L01', 'split': 'test', 'turns': 1}],
                         'experiment_spec': None},
            'providers': [{'provider': 'test', 'model': 'same'}], 'samples': [sample],
            'metrics': live.summarize([sample], 1)}


def _experiment_spec(*, minimum=0, paths=None):
    return {'schema_version': 1, 'experiment_id': 'context-test-v1', 'baseline_profile': 'adaptive-v1',
            'candidate_profile': 'adaptive-objective-context-v1',
            'dataset': 'eval/adaptive_tasks.json', 'suite': 'adaptive', 'repeats': 1,
            'tuning_split': 'dev',
            'evaluation_split': 'test',
            'changes': [{'kind': 'mechanism', 'slot': 'context', 'implementation_id': 'objective-first-context',
                         'paths': paths or ['src/context_engine.py', 'src/harness/mechanisms.py']}],
            'primary_metrics': [{'name': 'model_attempts', 'objective': 'minimize',
                                 'max_regression': 0, 'min_improvement': minimum}],
            'guard_metrics': [
                {'name': 'pass_at_1', 'objective': 'maximize', 'max_regression': 0, 'min_improvement': 0},
                {'name': 'all_repeats_success_rate', 'objective': 'maximize', 'max_regression': 0, 'min_improvement': 0},
                {'name': 'safety_violations', 'objective': 'minimize', 'max_regression': 0, 'min_improvement': 0},
                {'name': 'protocol_violations', 'objective': 'minimize', 'max_regression': 0, 'min_improvement': 0}],
            'compatibility_review': None}


def test_gate_allows_structured_mechanism_intervention():
    baseline = _comparable_live_report()
    candidate = _comparable_live_report('adaptive-objective-context-v1')
    candidate['manifest']['experiment_spec'] = _experiment_spec()
    assert live.evaluate_gate(candidate, baseline) == []


def test_gate_rejects_undeclared_intervention_and_changed_evaluator():
    baseline = _comparable_live_report()
    candidate = copy.deepcopy(baseline)
    candidate['manifest']['harness_profile'] = _comparable_live_report(
        'adaptive-objective-context-v1')['manifest']['harness_profile']
    failures = live.evaluate_gate(candidate, baseline)
    assert any('没有结构化实验契约' in failure for failure in failures)

    candidate = copy.deepcopy(baseline)
    candidate['manifest']['evaluation_protocol_sha256'] = 'changed-evaluator'
    candidate['manifest']['harness_profile'] = _comparable_live_report(
        'adaptive-objective-context-v1')['manifest']['harness_profile']
    candidate['manifest']['experiment_spec'] = _experiment_spec()
    failures = live.evaluate_gate(candidate, baseline)
    assert any('evaluation_protocol_sha256' in failure for failure in failures)


def test_gate_rejects_unapproved_source_and_kernel_changes():
    baseline = _comparable_live_report()
    candidate = _comparable_live_report('adaptive-objective-context-v1')
    candidate['manifest']['experiment_spec'] = _experiment_spec()
    candidate['manifest']['source_tree_sha256'] = 'changed'
    candidate['manifest']['source_files_sha256']['src/runtime.py'] = 'changed'
    failures = live.evaluate_gate(candidate, baseline)
    assert any('未声明的源码变更' in failure for failure in failures)
    assert any('Stable Kernel' in failure for failure in failures)


def test_gate_applies_declared_cost_metric_threshold():
    baseline = _comparable_live_report(model_attempts=1)
    candidate = _comparable_live_report('adaptive-objective-context-v1', model_attempts=2)
    candidate['manifest']['experiment_spec'] = _experiment_spec(minimum=1)
    failures = live.evaluate_gate(candidate, baseline)
    assert any('model_attempts' in failure and ('退化' in failure or '未达到' in failure) for failure in failures)


def test_gate_requires_declared_mechanism_source_files_and_manifest_integrity():
    baseline = _comparable_live_report()
    candidate = _comparable_live_report('adaptive-objective-context-v1')
    candidate['manifest']['experiment_spec'] = _experiment_spec(paths=['src/context_engine.py'])
    failures = live.evaluate_gate(candidate, baseline)
    assert any('机制源码未完整纳入变更范围' in failure for failure in failures)

    candidate = _comparable_live_report('adaptive-objective-context-v1')
    candidate['manifest']['experiment_spec'] = _experiment_spec()
    candidate['manifest']['source_tree_sha256'] = 'forged'
    failures = live.evaluate_gate(candidate, baseline)
    assert any('源码树指纹与逐文件指纹不一致' in failure for failure in failures)


def test_gate_rejects_a_declared_mechanism_without_runtime_application_evidence():
    baseline = _comparable_live_report()
    candidate = _comparable_live_report('adaptive-objective-context-v1')
    candidate['manifest']['experiment_spec'] = _experiment_spec()
    candidate['samples'][0]['turns'][0]['result']['events'] = []

    failures = live.evaluate_gate(candidate, baseline)

    assert any('没有在全部样本中产生执行证据' in failure for failure in failures)


def test_experiment_gate_requires_a_baseline_report():
    candidate = _comparable_live_report('adaptive-objective-context-v1')
    candidate['manifest']['experiment_spec'] = _experiment_spec()

    failures = live.evaluate_gate(candidate)

    assert any('必须同时提供冻结基线' in failure for failure in failures)


def test_cli_rejects_experiment_gate_without_baseline_before_model_execution(monkeypatch):
    import pytest

    monkeypatch.setattr(live, 'evaluate', lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError('缺少基线时不应启动评测')))
    spec = live.ROOT / 'research/experiments/objective-context-v1/experiment-spec.json'
    with pytest.raises(SystemExit):
        live.main(['--gate', '--experiment-spec', str(spec)])


def test_declared_harness_assembly_change_still_requires_kernel_review():
    baseline = _comparable_live_report()
    candidate = _comparable_live_report('adaptive-objective-context-v1')
    candidate['manifest']['experiment_spec'] = _experiment_spec(paths=[
        'src/context_engine.py', 'src/harness/mechanisms.py', 'src/harness/registry.py'])
    files = candidate['manifest']['source_files_sha256']
    files['src/harness/registry.py'] = 'changed'
    candidate['manifest']['source_tree_sha256'] = live.fingerprint(files)
    candidate['manifest']['kernel_source_sha256'] = live.fingerprint(
        {name: files.get(name) for name in sorted(live.KERNEL_FILES)})
    candidate['manifest']['kernel_boundaries_sha256'] = {
        name: live.fingerprint({path: files.get(path) for path in sorted(paths)})
        for name, paths in live.KERNEL_BOUNDARIES.items()}

    failures = live.evaluate_gate(candidate, baseline)

    assert any('Stable Kernel' in failure and 'harness_assembly' in failure for failure in failures)


def test_kernel_boundaries_cover_recovery_model_gateway_and_harness_assembly():
    assert 'src/recovery.py' in live.KERNEL_BOUNDARIES['persistence_and_recovery']
    assert 'src/model_gateway.py' in live.KERNEL_BOUNDARIES['model_call_protocol']
    assert 'src/harness/registry.py' in live.KERNEL_BOUNDARIES['harness_assembly']


def test_live_dataset_uses_frozen_declared_split_counts(tmp_path):
    dataset = live.load_dataset()
    dataset['cases'] = [dict(dataset['cases'][0], split='test')]
    dataset['split_counts'] = {'test': 1}
    path = tmp_path / 'challenge_tasks.json'
    path.write_text(json.dumps(dataset, ensure_ascii=False), encoding='utf-8')

    loaded = live.load_dataset(path)

    assert loaded['split_counts'] == {'test': 1}
    dataset['split_counts'] = {'test': 2}
    path.write_text(json.dumps(dataset, ensure_ascii=False), encoding='utf-8')
    with pytest.raises(ValueError, match='split_counts'):
        live.load_dataset(path)


def test_adaptive_dataset_rejects_unknown_expectations_before_live_execution(tmp_path):
    dataset = json.loads((live.ROOT / 'eval/ace_challenge_tasks.json').read_text(encoding='utf-8'))
    dataset['cases'][0]['adaptive_expect']['product_constraints'][0]['operator'] = 'lte'
    path = tmp_path / 'invalid_adaptive_tasks.json'
    path.write_text(json.dumps(dataset, ensure_ascii=False), encoding='utf-8')

    with pytest.raises(ValueError, match='product_constraints'):
        live.load_dataset(path)


def test_injected_runner_uses_real_business_evidence_and_is_never_labeled_live():
    run_orchestrator = live.offline._runtime()["orchestrator"].run_orchestrator
    dataset = live.load_dataset()
    dataset['cases'] = [dataset['cases'][0]]
    def runner(message, **kwargs):
        return run_orchestrator('查询订单 1002 的状态', **kwargs)
    report = live.evaluate(dataset, repeats=2, runner=runner, readiness={'configured': False})
    assert report['execution_source'] == 'test_harness'
    assert report['metrics']['mean_success_rate'] == 1
    assert not report['live_verified']
    assert any(item['tool'] == 'get_order_status'
               for item in report['samples'][0]['turns'][0]['observed_tools'])
    assert live.evaluate_gate(report)


def test_gate_rejects_missing_samples_and_nonlive_source():
    report = {'status': 'COMPLETED', 'live_verified': True, 'execution_source': 'test_harness',
              'manifest': {'repeats': 3, 'selected_cases': [{'id': 'L01', 'split': 'test', 'turns': 1}]},
              'samples': [], 'metrics': {'safety_violations': 0, 'protocol_violations': 0, 'mean_success_rate': 1}}
    failures = live.evaluate_gate(report)
    assert any('样本集合' in failure for failure in failures)
    assert any('真实模型' in failure for failure in failures)


def test_gate_without_baseline_reports_not_run(monkeypatch, capsys):
    monkeypatch.setattr(live, 'evaluate', lambda *_a, **_k: {'status': 'NOT_RUN', 'metrics': {}, 'samples': []})
    first = live.load_dataset()['cases'][0]['id']
    assert live.main(['--gate', '--case', first, '--repeats', '1', '--max-samples', '1']) == 2
    assert 'NOT_RUN' in capsys.readouterr().out


def test_dataset_order_expectations_match_requests():
    case = live.load_dataset()['cases'][12]
    assert '1009' in case['turns'][0]['message']


def test_dataset_requires_ten_cases_per_split(tmp_path):
    import json
    import pytest
    dataset = live.load_dataset()
    dataset['cases'][0]['split'] = 'test'
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(dataset))
    with pytest.raises(ValueError):
        live.load_dataset(path)


def test_exception_keeps_physical_business_calls():
    runtime = live.offline._runtime()
    dataset = live.load_dataset()
    dataset['cases'] = [dataset['cases'][0]]
    def runner(message, **kwargs):
        runtime['tools'].escalate_to_human('测试异常证据', '离线评测')
        raise RuntimeError('测试异常')
    report = live.evaluate(dataset, repeats=1, runner=runner, readiness={'configured': False})
    assert report['metrics']['tool_calls'] >= 1
    assert report['metrics']['safety_violations'] >= 1
    assert any(item['tool'] == 'escalate_to_human'
               for item in report['samples'][0]['turns'][0]['observed_tools'])
