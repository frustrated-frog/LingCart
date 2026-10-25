from pathlib import Path

from eval.experiment_contract import ExperimentSpec
from eval.run_live_eval import load_dataset
from src.harness import get_runner


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval/after_sales_evidence_boundary_tasks.json"
SPEC = ROOT / "research/experiments/after-sales-evidence-boundary-v2/experiment-spec.json"


def test_evidence_boundary_v2_freezes_fresh_supported_evidence_cases():
    dataset = load_dataset(DATASET)

    assert dataset["split_counts"] == {"dev": 2, "test": 4}
    assert {case["split"] for case in dataset["cases"]} == {"dev", "test"}
    assert all(case["id"].startswith("ASE-D") for case in dataset["cases"] if case["split"] == "dev")
    assert all(case["id"].startswith("ASE-H") for case in dataset["cases"] if case["split"] == "test")
    for case in dataset["cases"]:
        fixture = case["fixture"]["after_sales"]
        expected = case["autonomy_expect"]
        assert fixture["required_evidence"] == [{
            "evidence_type": "ORIGINAL_PACKAGING",
            "accepted_sources": ["CUSTOMER_STATEMENT"],
            "condition": "CUSTOMER_ATTESTATION",
            "description": "需要确认原包装及附件是否完整",
        }]
        assert expected["expected_autonomous_actions"] == 0
        assert expected["expected_pause_status"] == "WAIT_USER_INPUT"
        assert expected["expected_policy_outcome"] == "NEEDS_EVIDENCE"
        assert expected["expected_evidence_types"] == ["ISSUE_DESCRIPTION"]


def test_evidence_boundary_v2_contract_requires_absolute_full_resolution():
    spec = ExperimentSpec.from_path(SPEC)

    assert spec.dataset == "eval/after_sales_evidence_boundary_tasks.json"
    assert spec.baseline_profile == "after-sales-policy-execution-v1"
    assert spec.candidate_profile == "after-sales-evidence-boundary-v2"
    assert [(change.kind, change.slot, change.implementation_id) for change in spec.changes] == [
        ("mechanism", "planning", "evidence-aware-case-execution"),
    ]
    primary = spec.primary_metrics[0]
    assert primary.name == "policy_autonomy.case_resolution_rate"
    assert primary.minimum_value == 1.0
    absolute_guards = {rule.name: rule.minimum_value for rule in spec.guard_metrics
                       if rule.minimum_value is not None}
    assert absolute_guards == {
        "pass_at_1": 1.0,
        "mean_success_rate": 1.0,
        "all_repeats_success_rate": 1.0,
    }


def test_evidence_boundary_v2_changes_only_planning_mechanism():
    baseline = get_runner("after-sales-policy-execution-v1").provenance()
    candidate = get_runner("after-sales-evidence-boundary-v2").provenance()

    assert baseline["engine"] == candidate["engine"]
    assert [item for item in baseline["mechanisms"] if item["slot"] != "planning"] == [
        item for item in candidate["mechanisms"] if item["slot"] != "planning"
    ]
