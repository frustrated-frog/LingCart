import copy

from eval import run_live_eval as live


def test_live_dataset_has_frozen_split_and_no_scripted_model():
    dataset = live.load_dataset()
    assert len(dataset['cases']) == 20
    assert {c['split'] for c in dataset['cases']} == {'dev', 'test'}
    assert all('model' not in case for case in dataset['cases'])


def test_live_dry_run_does_not_import_application(monkeypatch, capsys):
    monkeypatch.setattr(live.offline, '_runtime', lambda: (_ for _ in ()).throw(AssertionError('不能导入应用')))
    assert live.main(['--dry-run']) == 0
    assert '未调用模型' in capsys.readouterr().out


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

    sample = dict(case_id='L01', split='test', repetition=1, passed=True, latency_ms=1,
                  model_attempts=model_attempts, tool_calls=0, safety_violations=0, protocol_violations=0,
                  adaptive_verified=True, turns=[{'usage': {}}])
    source_files = {'src/context_engine.py': 'same', 'src/runtime.py': 'same'}
    kernel_sha = live.fingerprint({name: source_files.get(name) for name in sorted(live.KERNEL_FILES)})
    return {'status': 'COMPLETED', 'live_verified': True, 'execution_source': 'live_provider',
            'manifest': {'dataset_path': 'eval/adaptive_tasks.json', 'dataset_sha256': 'dataset',
                         'suite': 'adaptive', 'execution_mode': 'adaptive',
                         'evaluation_protocol_sha256': 'evaluator', 'business_protocol_sha256': 'business',
                         'business_environment': 'fresh_demo_sqlite_per_sample',
                         'source_tree_sha256': live.fingerprint(source_files),
                         'source_files_sha256': source_files, 'kernel_source_sha256': kernel_sha,
                         'prompt_source_sha256': {'src/adaptive_decision.py': 'a'},
                         'harness_profile': get_runner(profile_id).provenance(),
                         'limits': {'max_seconds': 45}, 'split': 'test', 'repeats': 1,
                         'purpose_models': {'agent': [{'provider': 'test', 'model': 'same'}]},
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
    candidate = copy.deepcopy(baseline)
    candidate['manifest'].update(harness_profile=_comparable_live_report(
        'adaptive-objective-context-v1')['manifest']['harness_profile'], experiment_spec=_experiment_spec())
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
    assert report['samples'][0]['turns'][0]['observed_tools'][0]['tool'] == 'get_order_status'
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
    assert live.main(['--gate']) == 2
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
    assert report['metrics']['tool_calls'] == 1
    assert report['metrics']['safety_violations'] >= 1
    assert report['samples'][0]['turns'][0]['observed_tools'][0]['tool'] == 'escalate_to_human'
