from eval.run_pg_localization_eval import evaluate


def test_offline_localization_contract_passes_without_model_calls():
    result = evaluate()
    assert result["gate_passed"] is True
    assert result["model_calls"] == 0
    assert result["metrics"]["candidate_anchor_accuracy"] == 1
    assert result["metrics"]["candidate_full_graph_fallbacks"] == 0
    assert result["metrics"]["baseline_full_graph_fallbacks"] > 0
