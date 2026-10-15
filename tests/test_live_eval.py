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
