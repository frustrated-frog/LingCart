import json
from pathlib import Path

import pytest

from eval.run_pg_preplay_eval import evaluate, main


def test_preplay_contract_gate_is_deterministic_and_model_free():
    result = evaluate()
    assert result["gate_passed"] is True
    assert result["metrics"] == {
        "cases": 8,
        "contract_pass_rate": 1.0,
        "unsafe_predictions": 0,
        "abstentions": 2,
        "model_calls": 0,
    }


def test_preplay_contract_rejects_unknown_suite(tmp_path: Path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"schema_version": 1, "suite": "other"}))
    with pytest.raises(ValueError, match="评测协议"):
        evaluate(path)


def test_preplay_cli_gate_returns_success(capsys):
    assert main(["--gate"]) == 0
    assert json.loads(capsys.readouterr().out)["gate_passed"] is True
