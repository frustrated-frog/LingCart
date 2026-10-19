import json

import pytest

from eval import run_semantic_live_eval as semantic_eval


def _dataset():
    return {
        "schema_version": 1,
        "cases": [
            {
                "id": "time",
                "message": "纽约现在几点",
                "must_use_semantic": True,
                "expected": {"objectives": ["SYSTEM_TIME"], "statuses": ["SUCCEEDED"]},
            },
            {
                "id": "identity",
                "message": "你是谁",
                "must_use_semantic": True,
                "expected": {"commands_any": ["IDENTITY"], "statuses": ["SUCCEEDED"]},
            },
        ],
    }


def _result(*, status="SUCCEEDED", reason="COMPLETED"):
    return {
        "status": status,
        "terminal_reason": reason,
        "understanding": {
            "objectives": [{"kind": "SYSTEM_TIME", "timezone": "America/New_York"}],
            "commands": [],
        },
        "plan": {"steps": [{"tool": "read_clock", "args": {"timezone": "America/New_York"}}]},
        "trace": [{"tool": "read_clock"}],
        "usage": {"total_tokens": 42},
        "budget": {"consumed": {"model_calls": 1}},
    }


def test_default_cli_is_a_single_case_dry_run_without_evaluation(monkeypatch, capsys):
    monkeypatch.setattr(semantic_eval, "load_cases", lambda _path: _dataset())
    monkeypatch.setattr(
        semantic_eval,
        "evaluate",
        lambda *_args, **_kwargs: pytest.fail("默认预检不能执行评测"),
    )

    semantic_eval.main([])

    output = capsys.readouterr().out
    assert '"mode": "dry_run"' in output
    assert '"selected_case_ids": [' in output
    assert '"time"' in output
    assert "未调用模型" in output


def test_live_cli_rejects_a_plan_above_budget_before_model_status(monkeypatch, tmp_path):
    monkeypatch.setattr(semantic_eval, "load_cases", lambda _path: _dataset())

    with pytest.raises(SystemExit) as error:
        semantic_eval.main([
            "--live", "--all-cases", "--mode", "both",
            "--output", str(tmp_path / "report.json"),
        ])

    assert error.value.code == 2


def test_live_cli_requires_an_immutable_output_path(monkeypatch):
    monkeypatch.setattr(semantic_eval, "load_cases", lambda _path: _dataset())

    with pytest.raises(SystemExit) as error:
        semantic_eval.main(["--live", "--case", "time"])

    assert error.value.code == 2


def test_case_selection_is_explicit_and_rejects_unknown_ids():
    selected = semantic_eval.select_cases(_dataset(), case_ids=["identity", "time", "identity"])

    assert [case["id"] for case in selected] == ["identity", "time"]
    with pytest.raises(ValueError, match="未知语义评测案例"):
        semantic_eval.select_cases(_dataset(), case_ids=["missing"])


def test_injected_evaluation_records_raw_protocol_and_cost(monkeypatch):
    from src.intake import TaskIntake

    monkeypatch.setattr(
        "src.intake.analyze_request",
        lambda message: TaskIntake(request=message, execution_mode="LLM_REQUIRED", reason="test"),
    )
    seen_limits = []

    def runner(_message, *, limits):
        seen_limits.append(limits)
        return _result()

    dataset = _dataset()
    report = semantic_eval.evaluate(
        dataset,
        cases=dataset["cases"][:1],
        repeats=1,
        modes=("auto",),
        max_model_calls_per_run=2,
        readiness={"configured": True, "providers": [{"provider": "test", "model": "fixture"}]},
        runners={"auto": runner},
    )

    assert report["execution_source"] == "test_harness"
    assert report["metrics"]["pass_rate"] == 1
    assert report["metrics"]["model_calls"] == 1
    assert report["metrics"]["total_tokens"] == 42
    assert report["samples"][0]["understanding"]["objectives"][0]["kind"] == "SYSTEM_TIME"
    assert report["samples"][0]["plan"]["steps"][0]["tool"] == "read_clock"
    assert seen_limits[0].max_model_calls == 2


def test_auto_adaptive_terminal_mismatch_fails_both_samples(monkeypatch):
    from src.intake import TaskIntake

    monkeypatch.setattr(
        "src.intake.analyze_request",
        lambda message: TaskIntake(request=message, execution_mode="LLM_REQUIRED", reason="test"),
    )
    dataset = _dataset()
    report = semantic_eval.evaluate(
        dataset,
        cases=dataset["cases"][:1],
        repeats=1,
        modes=("auto", "adaptive"),
        max_model_calls_per_run=2,
        readiness={"configured": True},
        runners={
            "auto": lambda _message, *, limits: _result(),
            "adaptive": lambda _message, *, limits: _result(status="PARTIAL", reason="MODEL_STOPPED"),
        },
    )

    assert report["metrics"]["auto_adaptive_consistency"] == 0
    assert not any(sample["passed"] for sample in report["samples"])
    assert all("Auto/Adaptive semantic mismatch" in sample["failures"] for sample in report["samples"])


def test_dataset_digest_is_stable_across_key_order():
    first = _dataset()
    second = json.loads(json.dumps(first, ensure_ascii=False))

    assert semantic_eval._dataset_digest(first) == semantic_eval._dataset_digest(second)
