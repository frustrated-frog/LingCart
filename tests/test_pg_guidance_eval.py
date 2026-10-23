from eval.run_pg_guidance_eval import evaluate, main


def test_guidance_contract_gate_is_deterministic_and_model_free():
    result = evaluate()
    assert result['gate_passed'] is True
    assert result['model_calls'] == result['metrics']['guidance_model_calls'] == 0
    assert result['metrics']['contract_pass_rate'] == 1
    assert result['metrics']['blocked_exact_repeats'] == 1
    assert {row['review_status'] for row in result['cases']} == {'ALIGNED', 'BLOCKED', 'DIVERGED', 'UNKNOWN'}


def test_guidance_contract_cli_gate_passes():
    assert main(['--gate']) == 0
