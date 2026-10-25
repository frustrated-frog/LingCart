from eval.run_pg_state_condition_eval import evaluate, main


def test_state_condition_gate_is_deterministic_safe_and_model_free():
    result = evaluate()
    assert result["gate_passed"] is True
    assert result["model_calls"] == result["metrics"]["model_calls"] == 0
    assert result["metrics"]["contract_pass_rate"] == 1
    assert result["metrics"]["unsafe_selected_edges"] == 0
    assert result["metrics"]["unknown_evidence_abstentions"] == 1


def test_state_condition_cli_gate_passes():
    assert main(["--gate"]) == 0
