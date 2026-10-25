import json
from datetime import datetime
from pathlib import Path

import pytest

from eval.after_sales_autonomy_checks import _latest_case
from eval.experiment_contract import ExperimentSpec
from eval.run_live_eval import load_dataset, summarize
from src.backend import db


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval/after_sales_autonomy_tasks.json"
SPEC = ROOT / "research/experiments/after-sales-autonomy-v1/experiment-spec.json"


def test_after_sales_autonomy_dataset_and_contract_are_frozen():
    dataset = load_dataset(DATASET)
    spec = ExperimentSpec.from_path(SPEC)

    assert dataset["split_counts"] == {"dev": 2, "test": 4}
    assert spec.dataset == "eval/after_sales_autonomy_tasks.json"
    assert spec.baseline_profile == "after-sales-plan-observer-v0"
    assert spec.candidate_profile == "after-sales-policy-execution-v1"
    assert [(item.kind, item.slot, item.implementation_id) for item in spec.changes] == [
        ("mechanism", "planning", "policy-aware-case-execution"),
    ]
    assert {rule.name for rule in spec.primary_metrics} == {
        "policy_autonomy.autonomous_resolution_rate",
        "policy_autonomy.avoidable_pause_rate",
    }


def test_after_sales_autonomy_dataset_rejects_unfrozen_expectations(tmp_path):
    payload = json.loads(DATASET.read_text())
    payload["cases"][0]["autonomy_expect"].pop("expected_pause_status")
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload, ensure_ascii=False))

    with pytest.raises(ValueError, match="autonomy_expect"):
        load_dataset(path)


def test_latest_case_uses_persisted_open_time(tmp_path):
    engine = db.get_engine(f"sqlite:///{tmp_path / 'autonomy-judge.db'}")
    db.init_db(engine)
    with db.get_session_factory(engine)() as session:
        for case_id, opened_at in (
            ("older", datetime(2026, 1, 1)),
            ("newer", datetime(2026, 1, 2)),
        ):
            session.add(db.AfterSalesCase(
                case_id=case_id,
                request_key=f"request-{case_id}",
                case_type="REFUND",
                order_id=1001,
                order_item_id=None,
                requested_resolution="REFUND",
                reason_code="NOT_AS_DESCRIBED",
                reason_text="测试评测器案件排序",
                status="OPEN",
                revision=1,
                origin_session_id=None,
                context={},
                opened_at=opened_at,
                updated_at=opened_at,
            ))
        session.commit()

    latest, evidence, decision = _latest_case({"db": db}, engine)

    assert latest.case_id == "newer"
    assert evidence == []
    assert decision is None


def test_live_summary_reports_policy_autonomy_without_inventing_cost_savings():
    samples = [
        {"case_id": "one", "repetition": 1, "passed": True, "latency_ms": 10,
         "model_attempts": 1, "tool_calls": 2, "safety_violations": 0, "protocol_violations": 0,
         "turns": [], "autonomy": {"expected_autonomous_actions": 1, "resolved": True,
                                      "avoidable_pause": False, "approval_boundary_expected": True,
                                      "approval_boundary_preserved": True, "unauthorized_write_count": 0,
                                      "duplicate_action_count": 0}},
        {"case_id": "two", "repetition": 1, "passed": False, "latency_ms": 20,
         "model_attempts": 1, "tool_calls": 1, "safety_violations": 0, "protocol_violations": 0,
         "turns": [], "autonomy": {"expected_autonomous_actions": 1, "resolved": False,
                                      "avoidable_pause": True, "approval_boundary_expected": False,
                                      "approval_boundary_preserved": True, "unauthorized_write_count": 0,
                                      "duplicate_action_count": 0}},
    ]

    metrics = summarize(samples, repeats=1)

    assert metrics["policy_autonomy"] == {
        "autonomous_resolution_rate": 0.5,
        "avoidable_pause_rate": 0.5,
        "approval_boundary_rate": 1.0,
        "unauthorized_write_count": 0,
        "duplicate_action_count": 0,
    }
