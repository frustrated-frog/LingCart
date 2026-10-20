import json

import pytest

from eval import run_liveplan_monitor_eval as monitor_eval


def test_frozen_test_split_has_perfect_monitor_gate_without_model_calls():
    report = monitor_eval.evaluate(split="test")

    assert report["status"] == "PASSED"
    assert report["metrics"]["cases"] == 8
    assert report["metrics"]["exact_match_rate"] == 1
    assert report["metrics"]["precision"] == 1
    assert report["metrics"]["recall"] == 1
    assert report["metrics"]["model_calls"] == 0


def test_dataset_validation_rejects_split_count_drift(tmp_path):
    payload = monitor_eval.load_dataset()
    payload["split_counts"]["test"] += 1
    path = tmp_path / "monitor-cases.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="split_counts"):
        monitor_eval.load_dataset(path)


def test_cli_gate_can_write_a_replayable_report_without_model_calls(tmp_path, capsys):
    output = tmp_path / "monitor-report.json"

    assert monitor_eval.main(["--split", "dev", "--gate", "--output", str(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "PASSED"
    assert report["metrics"]["cases"] == 4
    assert report["metrics"]["model_calls"] == 0
    assert '"status": "PASSED"' in capsys.readouterr().out
