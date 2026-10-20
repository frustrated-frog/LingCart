import json
from pathlib import Path

import pytest

from eval import run_semantic_live_eval as semantic_eval
from eval.eval_journal import EvaluationJournal
from eval.semantic_contracts import (
    AcceptanceStage,
    SemanticExperimentSpec,
    StageBudget,
    StageExecution,
    StageSelection,
    fingerprint,
)


ROOT = Path(__file__).resolve().parents[1]


def _dataset():
    return semantic_eval.load_dataset(ROOT / "eval/semantic_cases.json")


def _case(case_id="trusted-timezone-normalization"):
    return next(case for case in _dataset().cases if case.id == case_id)


def _result(*, timezone="America/New_York", status="SUCCEEDED", reason="COMPLETED",
            max_model_calls=2):
    run_id = "semantic-runner-test"
    events = [
        {"run_id": run_id, "sequence": 1, "type": "model_started",
         "purpose": "dialogue_understanding", "agent": "intake"},
        {"run_id": run_id, "sequence": 2, "type": "model_call", "status": "SUCCEEDED",
         "purpose": "dialogue_understanding", "provider": "test", "model": "fixture"},
        {"run_id": run_id, "sequence": 3, "type": "tool_call", "tool": "read_clock",
         "args": {"timezone": timezone}, "execution_id": "clock-execution",
         "action_fingerprint": "clock-fingerprint", "policy": {"effect": "READ"}},
        {"run_id": run_id, "sequence": 4, "type": "tool_result", "tool": "read_clock",
         "args": {"timezone": timezone},
         "result": {"timezone": timezone, "data_source": "RUNTIME_CLOCK", "observed_at": "now"},
         "status": "SUCCEEDED", "execution_id": "clock-execution",
         "action_fingerprint": "clock-fingerprint", "policy": {"effect": "READ"}},
        {"run_id": run_id, "sequence": 5, "type": "verification",
         "goal_outcomes": [{"objective_id": "clock", "outcome": "SATISFIED", "step_ids": ["step-1"]}],
         "all_goals_satisfied": status == "SUCCEEDED", "side_effects_executed": False},
        {"run_id": run_id, "sequence": 6, "type": "terminal", "status": status, "reason": reason},
    ]
    verification = events[-2] | {"type": None, "run_id": None, "sequence": None}
    verification = {key: value for key, value in verification.items() if value is not None}
    return {
        "run_id": run_id,
        "status": status,
        "terminal_reason": reason,
        "understanding": {
            "objectives": [{"id": "clock", "kind": "SYSTEM_TIME", "timezone": timezone}],
            "commands": [],
        },
        "plan": {
            "steps": [{"step_id": "step-1", "tool": "read_clock", "args": {"timezone": timezone},
                       "capability_id": "system.clock.read", "objective_id": "clock",
                       "objective_ids": ["clock"], "status": "SUCCEEDED"}],
            "verification": verification,
        },
        "events": events,
        "usage": {"total_tokens": 42},
        "budget": {
            "limits": {"max_model_calls": max_model_calls},
            "consumed": {"model_calls": 1},
        },
    }


def _stage(*, modes=("auto",), cap=2, repeats=1, gate_type="smoke"):
    return AcceptanceStage(
        id="fixture-stage",
        gate_type=gate_type,
        claim_scope="selected_cases_only" if gate_type == "smoke" else "frozen_stage",
        selection=StageSelection(case_ids=("trusted-timezone-normalization",), all_cases=False),
        executions=tuple(StageExecution(
            mode=mode, profile_id="adaptive-v1" if mode == "adaptive" else "baseline-v1"
        ) for mode in modes),
        repeats=repeats,
        budget=StageBudget(max_executions=len(modes) * repeats, max_model_calls_per_run=cap),
        requires=(),
    )


def test_default_cli_uses_frozen_single_smoke_and_never_evaluates(monkeypatch, capsys):
    monkeypatch.setattr(semantic_eval, "evaluate", lambda *_args, **_kwargs: pytest.fail("dry run cannot execute"))

    semantic_eval.main([])

    output = capsys.readouterr().out
    assert '"mode": "dry_run"' in output
    assert '"stage_id": "single-smoke"' in output
    assert '"trusted-timezone-normalization"' in output
    assert "未调用模型" in output


def test_case_selection_uses_validated_v2_contract_and_rejects_unknown_ids():
    selected = semantic_eval.select_cases(
        _dataset(),
        case_ids=["identity-command", "trusted-timezone-normalization", "identity-command"],
    )

    assert [case["id"] for case in selected] == ["identity-command", "trusted-timezone-normalization"]
    with pytest.raises(ValueError, match="未知语义评测案例"):
        semantic_eval.select_cases(_dataset(), case_ids=["missing"])
    grouped = semantic_eval.select_cases(_dataset(), groups=["conversation"])
    assert {case["id"] for case in grouped} == {"identity-command", "capability-paraphrase"}


def test_effective_model_budget_can_only_tighten_runtime_and_stage_limits():
    assert semantic_eval.effective_model_call_budget("auto", stage_cap=99, cli_cap=99)["effective"] == 8
    assert semantic_eval.effective_model_call_budget("adaptive", stage_cap=99, cli_cap=99)["effective"] == 16
    assert semantic_eval.effective_model_call_budget("adaptive", stage_cap=6, cli_cap=4)["effective"] == 4
    assert semantic_eval.effective_model_call_budget("adaptive", stage_cap=6, cli_cap=99)["effective"] == 6
    for value in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="正整数"):
            semantic_eval.effective_model_call_budget("auto", stage_cap=value)


def test_injected_evaluation_records_raw_protocol_profile_and_cost():
    seen_limits = []

    def runner(_message, *, history, limits):
        seen_limits.append(limits)
        assert history is None
        return _result(max_model_calls=limits.max_model_calls)

    stage = _stage(cap=2)
    report = semantic_eval.evaluate(
        _dataset(), cases=[_case().model_dump(mode="json")], repeats=1, modes=("auto",),
        max_model_calls_per_run=2,
        readiness={"configured": True, "providers": [{"provider": "test", "model": "fixture"}]},
        runners={"auto": runner}, stage=stage,
    )

    assert report["execution_source"] == "test_harness"
    assert report["metrics"]["pass_rate"] == 1
    assert report["metrics"]["model_calls"] == 1
    assert report["metrics"]["total_tokens"] == 42
    assert report["samples"][0]["raw_result"]["events"][3]["type"] == "tool_result"
    assert report["samples"][0]["model_providers"] == [{
        "purpose": "dialogue_understanding", "provider": "test", "model": "fixture",
    }]
    assert seen_limits[0].max_model_calls == 2


def test_auto_adaptive_semantic_mismatch_fails_both_samples():
    stage = _stage(modes=("auto", "adaptive"), cap=2)
    report = semantic_eval.evaluate(
        _dataset(), cases=[_case().model_dump(mode="json")], repeats=1,
        modes=("auto", "adaptive"), max_model_calls_per_run=2,
        readiness={"configured": True}, stage=stage,
        runners={
            "auto": lambda _message, *, history, limits: _result(
                timezone="America/New_York", max_model_calls=limits.max_model_calls),
            "adaptive": lambda _message, *, history, limits: _result(
                timezone="Europe/London", max_model_calls=limits.max_model_calls),
        },
    )

    assert report["metrics"]["auto_adaptive_consistency"] == 0
    assert not any(sample["passed"] for sample in report["samples"])
    assert all("AUTO_ADAPTIVE_SEMANTIC_MISMATCH" in sample["failure_codes"]
               for sample in report["samples"])


def test_history_is_passed_as_typed_messages():
    case = _case("history-order-pronoun")
    stage = AcceptanceStage(
        id="history-smoke", gate_type="smoke", claim_scope="selected_cases_only",
        selection=StageSelection(case_ids=(case.id,), all_cases=False),
        executions=(StageExecution(mode="auto", profile_id="baseline-v1"),),
        repeats=1, budget=StageBudget(max_executions=1, max_model_calls_per_run=2), requires=(),
    )

    def runner(_message, *, history, limits):
        assert [item.type for item in history] == ["human", "ai"]
        assert [item.content for item in history] == ["我想查订单 1002", "已经记下订单 1002。"]
        raise RuntimeError("stop after history assertion")

    report = semantic_eval.evaluate(
        _dataset(), cases=[case.model_dump(mode="json")], repeats=1, modes=("auto",),
        max_model_calls_per_run=2, readiness={"configured": True},
        runners={"auto": runner}, stage=stage,
    )

    assert report["status"] == "COMPLETED"
    assert report["samples"][0]["outcome"] == "EXCEPTION"


def test_gate_recomputes_raw_samples_and_rejects_injected_source_even_if_label_is_forged():
    stage = _stage(cap=2)
    report = semantic_eval.evaluate(
        _dataset(), cases=[_case().model_dump(mode="json")], repeats=1, modes=("auto",),
        max_model_calls_per_run=2,
        readiness={"configured": True, "providers": [{"provider": "test", "model": "fixture"}]},
        runners={"auto": lambda _message, *, history, limits: _result(max_model_calls=limits.max_model_calls)},
        stage=stage,
    )
    report["execution_source"] = "real_model"
    report["samples"][0]["passed"] = True
    report["metrics"]["pass_rate"] = 1

    receipt = semantic_eval.SmokeGate.evaluate(report, _dataset(), stage)

    assert not receipt.passed
    assert "PROFILE_IDENTITY_MISSING" in receipt.failure_codes


def test_smoke_report_cannot_be_promoted_by_acceptance_gate():
    with pytest.raises(ValueError, match="不能把 Smoke"):
        semantic_eval.AcceptanceGate.evaluate({}, _dataset(), _stage(gate_type="smoke"))


def test_sample_exception_is_journaled_before_evaluation_stops(tmp_path):
    case = _case()
    stage = _stage(cap=2)
    journal_path = tmp_path / "report.journal.jsonl"

    with EvaluationJournal(journal_path, resume=False) as journal:
        report = semantic_eval.evaluate(
            _dataset(), cases=[case.model_dump(mode="json")], repeats=1, modes=("auto",),
            max_model_calls_per_run=2, readiness={"configured": True},
            runners={"auto": lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("secret"))},
            stage=stage, journal=journal,
        )
        terminal = [record for record in journal.records if record["record_type"] == "SAMPLE_TERMINAL"]

    assert report["samples"][0]["outcome"] == "EXCEPTION"
    assert len(terminal) == 1
    assert terminal[0]["payload"]["sample"]["exception"]["safe_message"].startswith("模型或运行时")
    assert "secret" not in journal_path.read_text()


def test_dataset_and_stage_hashes_are_stable_but_semantic_change_is_detected():
    dataset = _dataset()
    first = dataset.model_dump(mode="json")
    reordered = json.loads(json.dumps(first, ensure_ascii=False))
    assert fingerprint(first) == fingerprint(reordered)

    reordered["cases"][0]["message"] += "。"
    assert fingerprint(first) != fingerprint(reordered)


def _accepted_experiment_report(profile_id, *, pass_rate, total_tokens, stage_sha, source_sha):
    manifest = {
        "dataset_sha256": "dataset",
        "acceptance_spec_sha256": "acceptance",
        "stage_contract_sha256": stage_sha,
        "case_sha256": {"case": "case-sha"},
        "selected_case_ids": ["case"],
        "repeats": 1,
        "executions": [{"mode": "auto", "profile_id": profile_id}],
        "profiles": {"auto": {"profile_id": profile_id, "profile_sha256": profile_id,
                                 "execution_sha256": profile_id}},
        "model_call_budgets": {"auto": {"effective": 2}},
        "model_configuration": {"providers": [{"provider": "test", "model": "same"}]},
        "oracle_protocol": {"id": "semantic-goal-oracle", "version": 1},
        "evaluator_protocol_sha256": "evaluator",
        "dependency_lock_sha256": "lock",
        "python_runtime": {"implementation": "CPython", "version": "3.11"},
        "business_environment": "local_demo_sqlite",
        "source_files_sha256": {"src/runtime.py": source_sha},
    }
    manifest["source_tree_sha256"] = fingerprint(manifest["source_files_sha256"])
    manifest["experiment_compatibility_sha256"] = fingerprint(
        semantic_eval._experiment_compatibility_payload(manifest)
    )
    report = {
        "status": "COMPLETED",
        "manifest": manifest,
        "samples": [{"case_sha256": "case-sha", "repeat": 1, "mode": "auto"}],
        "metrics": {"pass_rate": pass_rate, "total_tokens": total_tokens,
                    "latency_ms": {"mean": 10}},
    }
    report["gate"] = {
        "gate_type": "ACCEPTANCE",
        "passed": True,
        "checked_report_sha256": fingerprint(report),
    }
    return report


def _experiment_spec():
    return SemanticExperimentSpec.model_validate_json(json.dumps({
        "schema_version": 1,
        "experiment_id": "semantic-profile-v1",
        "baseline_profiles": ["baseline-v1"],
        "candidate_profiles": ["candidate-v1"],
        "primary_metrics": [{
            "name": "pass_rate", "objective": "maximize",
            "max_regression": 0, "min_improvement": 0.1,
        }],
        "guard_metrics": [{
            "name": "total_tokens", "objective": "minimize",
            "max_regression": 20, "min_improvement": 0,
        }],
    }))


def test_experiment_compatibility_allows_declared_profile_and_stage_difference():
    baseline = _accepted_experiment_report(
        "baseline-v1", pass_rate=.5, total_tokens=100, stage_sha="baseline-stage", source_sha="base",
    )
    candidate = _accepted_experiment_report(
        "candidate-v1", pass_rate=.8, total_tokens=110, stage_sha="candidate-stage", source_sha="candidate",
    )

    compatibility = semantic_eval.ExperimentCompatibilityGate.evaluate(
        baseline, candidate, stage_id="semantic-profile-v1",
    )
    admission = semantic_eval.ExperimentGate.evaluate(baseline, candidate, spec=_experiment_spec())

    assert compatibility.passed, compatibility
    assert compatibility.evidence["profile_difference_allowed"] is True
    assert admission.passed, admission
    comparisons = admission.evidence["effect"]["evidence"]["metric_comparisons"]
    assert {row["metric"] for row in comparisons} == {"pass_rate", "total_tokens"}


def test_experiment_gate_rejects_no_effect_and_forged_compatibility_identity():
    baseline = _accepted_experiment_report(
        "baseline-v1", pass_rate=.5, total_tokens=100, stage_sha="baseline-stage", source_sha="base",
    )
    candidate = _accepted_experiment_report(
        "candidate-v1", pass_rate=.5, total_tokens=100, stage_sha="candidate-stage", source_sha="candidate",
    )
    no_effect = semantic_eval.ExperimentGate.evaluate(baseline, candidate, spec=_experiment_spec())
    assert not no_effect.passed
    assert "EXPERIMENT_IMPROVEMENT_NOT_MET" in no_effect.failure_codes

    candidate["manifest"]["model_call_budgets"]["auto"]["effective"] = 3
    forged = semantic_eval.ExperimentCompatibilityGate.evaluate(
        baseline, candidate, stage_id="semantic-profile-v1",
    )
    assert not forged.passed
    assert "EXPERIMENT_IDENTITY_INVALID" in forged.failure_codes


def test_source_manifest_covers_runtime_closure_and_is_recomputable():
    manifest = semantic_eval._source_manifest()

    assert manifest["source_manifest_version"] == 2
    for path in (
        "src/catalog_concepts.py",
        "src/constraint_bindings.py",
        "src/catalog_metadata.py",
        "src/tools.py",
    ):
        assert path in manifest["source_files_sha256"]
    assert manifest["source_tree_sha256"] == fingerprint(manifest["source_files_sha256"])
