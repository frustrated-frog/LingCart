import json

import pytest
from langchain_core.messages import AIMessage

from eval import run_evidence_semantic_eval as evaluator
from tests.evidence_fixtures import response_for


def test_default_is_dry_run_without_importing_a_model(monkeypatch, capsys):
    monkeypatch.setattr(evaluator, 'evaluate', lambda *_a, **_k: pytest.fail('dry-run must not evaluate'))
    evaluator.main([])
    result = json.loads(capsys.readouterr().out)
    assert result['mode'] == 'dry_run'
    assert result['max_model_attempts'] == 42


def test_frozen_dataset_has_separate_splits_and_rejects_future_withdrawal():
    data, spec = evaluator.load_inputs()
    assert len(data.cases) == 10
    assert spec.scope == 'evidence_verification_component'
    invalid = data.cases[0].model_dump()
    invalid['turns'][0]['withdraw_turn'] = 1
    with pytest.raises(ValueError, match='已经提交'):
        evaluator.EvidenceCase.model_validate(invalid)


def test_scripted_oracle_verifies_measurements_and_keeps_raw_outputs():
    data, spec = evaluator.load_inputs()
    oracle = {t.message: t for c in data.cases for t in c.turns}
    class Oracle:
        def invoke(self, prompt):
            inputs = json.loads(prompt.split('\nEvidence input:\n')[-1])
            turn = oracle[inputs['latest_request']]
            return AIMessage(content=json.dumps(response_for(inputs, turn.expected_statuses[0],
                                                             withdrawal=turn.expect_withdrawal), ensure_ascii=False))
    saved = []
    report = evaluator.evaluate(data, spec, split='test', model_factory=Oracle,
                                execution_source='test_harness', on_sample=saved.append)
    baseline, candidate = report['metrics'][spec.baseline], report['metrics'][spec.candidate]
    assert candidate['turn_accuracy'] == 1
    assert candidate['false_acceptances'] == candidate['unauthorized_withdrawals'] == 0
    assert baseline['false_acceptances'] > 0
    assert baseline['unauthorized_withdrawals'] > 0
    assert report['gate']['passed']
    assert report['gate']['promotion_eligible'] is False
    assert report['execution_source'] == 'test_harness'
    assert len(saved) == len(report['samples'])
    assert all(s['raw_model_responses'] for s in saved if s['variant'] == spec.candidate)
    assert candidate['total_tokens'] is None  # 无 usage 的模型替身不能伪造费用数据。


def test_bad_model_output_is_a_retained_negative_result():
    data, spec = evaluator.load_inputs()
    class Invalid:
        def invoke(self, _prompt):
            return AIMessage(content='{"invented":true}')
    report = evaluator.evaluate(data, spec, split='dev', model_factory=Invalid, execution_source='test_harness')
    assert report['gate']['passed'] is False
    assert 'protocol_errors' in report['gate']['failures']
    assert report['metrics'][spec.candidate]['protocol_errors'] == 21
    assert report['samples'][-1]['raw_model_responses'] == ['{"invented":true}']


def test_live_requires_fresh_explicit_output_before_calling_models(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, 'evaluate', lambda *_a, **_k: pytest.fail('must not evaluate'))
    with pytest.raises(SystemExit):
        evaluator.main(['--live'])
    existing = tmp_path / 'result.json'
    existing.write_text('original')
    with pytest.raises(SystemExit):
        evaluator.main(['--live', '--output', str(existing)])
    assert existing.read_text() == 'original'
