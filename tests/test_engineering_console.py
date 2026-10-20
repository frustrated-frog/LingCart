import json

import pytest

from src import engineering_console, tools
from src.backend.db import get_engine, get_session_factory
from src.backend.seed import seed
from src.engineering_console import _report_mismatches, dataset_catalog, experiment_catalog, profile_catalog


@pytest.fixture(autouse=True)
def isolated_db():
    engine = get_engine("sqlite:///:memory:")
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)


def test_registry_view_model_exposes_one_slot_candidate_change():
    profiles = {item["profile_id"]: item for item in profile_catalog()}
    baseline = profiles["adaptive-v1"]
    candidate = profiles["adaptive-objective-context-v1"]
    changed = [slot for slot in candidate["mechanisms"]
               if candidate["mechanisms"].get(slot) != baseline["mechanisms"].get(slot)]
    assert changed == ["context"]
    assert candidate["research"]["status"] == "candidate"
    assert candidate["execution_sha256"]

    liveplan = profiles["adaptive-liveplan-v1"]
    liveplan_changed = [slot for slot in liveplan["mechanisms"]
                        if liveplan["mechanisms"].get(slot) != baseline["mechanisms"].get(slot)]
    assert liveplan_changed == ["steering"]
    assert liveplan["research"]["status"] == "candidate"


def test_dataset_view_model_reports_real_case_and_split_counts():
    datasets = {item["dataset_id"]: item for item in dataset_catalog()}
    assert datasets["offline-core-v1"]["case_count"] == 60
    assert datasets["live-support-v1"]["splits"] == {"dev": 10, "test": 10}
    assert datasets["adaptive-agent-v1"]["turn_count"] == 21


def test_experiment_view_does_not_invent_a_comparable_report(monkeypatch, tmp_path):
    folder = tmp_path / "research" / "experiments" / "empty-v1"
    folder.mkdir(parents=True)
    spec = {"experiment_id": "empty-v1", "baseline_profile": "adaptive-v1",
            "candidate_profile": "candidate-v1"}
    (folder / "experiment-spec.json").write_text(json.dumps(spec))
    monkeypatch.setattr(engineering_console, "ROOT", tmp_path)
    monkeypatch.setattr(engineering_console, "_current_experiment_identity", lambda _spec: {})

    experiment = experiment_catalog()[0]
    assert experiment["experiment_id"] == "empty-v1"
    assert experiment["status"] == "AWAITING_REPORT"
    assert experiment["report"] is None


def test_experiment_view_surfaces_a_rejected_report(monkeypatch, tmp_path):
    folder = tmp_path / "research" / "experiments" / "rejected-v1"
    output = tmp_path / "output" / "nested"
    folder.mkdir(parents=True)
    output.mkdir(parents=True)
    spec = {"experiment_id": "rejected-v1", "baseline_profile": "adaptive-v1",
            "candidate_profile": "candidate-v1"}
    (folder / "experiment-spec.json").write_text(json.dumps(spec))
    current = {"experiment_spec": spec, "harness_profile": {}}
    report = {"status": "COMPLETED", "generated_at": "2026-10-18T00:00:00Z",
              "manifest": current, "metrics": {"pass_at_1": 0.8},
              "gate": {"passed": False, "failures": ["正确率退化"]}}
    (output / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(engineering_console, "ROOT", tmp_path)
    monkeypatch.setattr(engineering_console, "_current_experiment_identity", lambda _spec: current)

    experiment = experiment_catalog()[0]
    assert experiment["status"] == "CURRENT_REJECTED"
    assert experiment["report"]["gate"]["failures"] == ["正确率退化"]
    assert experiment["report_path"] == "output/nested/report.json"


def test_report_identity_detects_stale_source_profile_model_and_dataset():
    current = {
        "experiment_spec": {"experiment_id": "x"}, "dataset_path": "eval/tasks.json",
        "dataset_sha256": "dataset-new", "suite": "adaptive", "split": "test", "repeats": 3,
        "source_tree_sha256": "source-new", "evaluation_protocol_sha256": "eval",
        "business_protocol_sha256": "business", "dependency_lock_sha256": "lock",
        "purpose_models": {"agent": [{"provider": "openai", "model": "new"}]},
        "harness_profile": {"profile_id": "candidate", "profile_sha256": "profile-new",
                            "execution_sha256": "execution-new"},
    }
    report = {"manifest": {**current, "dataset_sha256": "dataset-old", "source_tree_sha256": "source-old",
                            "purpose_models": {"agent": [{"provider": "openai", "model": "old"}]},
                            "harness_profile": {"profile_id": "candidate", "profile_sha256": "profile-old",
                                                "execution_sha256": "execution-old"}}}
    mismatches = _report_mismatches(report, current)
    assert {"dataset_sha256", "source_tree_sha256", "purpose_models",
            "harness_profile.profile_sha256", "harness_profile.execution_sha256"} <= set(mismatches)
