"""Validate the offline evaluator itself and selected real task workflows."""

import copy
import json
import os
import subprocess
import sys

import pytest

from eval import run_task_eval as task_eval


def _case(case_id):
    return copy.deepcopy(next(case for case in task_eval.load_dataset()["cases"] if case["id"] == case_id))


def test_fixed_dataset_has_independent_tasks_and_required_coverage():
    dataset = task_eval.load_dataset()
    cases = dataset["cases"]
    assert 40 <= len(cases) <= 60
    assert len({case["id"] for case in cases}) == len(cases)
    assert len({case["description"] for case in cases}) == len(cases)
    assert {"zh", "en"} <= {case["language"] for case in cases}
    assert {"compound", "context", "clarification", "human", "unsupported",
            "rule_rejection", "payment_approval", "refund_approval", "shopping",
            "model_resilience"} <= {case["category"] for case in cases}
    assert sum(case["category"] == "order" for case in cases) < len(cases) // 4
    assert all(turn["expect"]["statuses"] for case in cases for turn in case["turns"])


@pytest.mark.parametrize("defect", ["duplicate_id", "unknown_status", "unknown_expectation", "unknown_tool"])
def test_dataset_validation_rejects_mistakes(defect):
    dataset = task_eval.load_dataset()
    if defect == "duplicate_id":
        dataset["cases"][1]["id"] = dataset["cases"][0]["id"]
    elif defect == "unknown_status":
        dataset["cases"][0]["turns"][0]["expect"]["statuses"] = ["DONE_MAYBE"]
    elif defect == "unknown_expectation":
        dataset["cases"][0]["turns"][0]["expect"]["max_modell_calls"] = 0
    else:
        dataset["cases"][0]["turns"][0]["expect"]["tools"] = [{"name": "fake_tool"}]
    with pytest.raises(ValueError):
        task_eval.validate_dataset(dataset)


def test_dry_run_validates_without_loading_application(monkeypatch, tmp_path, capsys):
    def forbidden_import():
        raise AssertionError("dry-run must not import the application")

    monkeypatch.setattr(task_eval, "_runtime", forbidden_import)
    output = tmp_path / "dry-run.json"
    assert task_eval.main(["--dry-run", "--output", str(output)]) == 0
    assert "No application imports" in capsys.readouterr().out
    report = json.loads(output.read_text())
    assert report["validated"] is True
    assert report["application_imported"] is False
    assert report["case_count"] == 60


@pytest.mark.parametrize("case_id", ["T01", "T09", "T21", "T22", "T33", "T35", "T36", "T43", "T48", "T53",
                                   "T55", "T56", "T57", "T58", "T59", "T60"])
def test_real_workflows_have_correct_tools_state_and_database(case_id):
    result = task_eval.evaluate_case(_case(case_id))
    assert result["passed"], json.dumps(result, ensure_ascii=False, indent=2)


def test_each_task_is_seeded_independently_and_original_tool_database_is_restored():
    runtime = task_eval._runtime()
    tools, db = runtime["tools"], runtime["db"]
    original_engine, original_factory = tools._engine, tools._SessionLocal
    caller_engine = db.get_engine("sqlite:///:memory:")
    try:
        tools.configure_db(caller_engine)
        caller_factory = tools._SessionLocal
        with caller_factory() as session:
            runtime["seed"](session)
            payment = session.query(db.Payment).filter_by(order_id=1003).one()
            payment.status, payment.method = "paid", "caller-method"
            session.commit()
        first = task_eval.evaluate_case(_case("T32"), runtime)
        second = task_eval.evaluate_case(_case("T32"), runtime)
        assert first["passed"] and second["passed"]
        assert tools._engine is caller_engine
        assert tools._SessionLocal is caller_factory
        with caller_factory() as session:
            payment = session.query(db.Payment).filter_by(order_id=1003).one()
            assert payment.status == "paid"
            assert payment.method == "caller-method"
    finally:
        tools._engine, tools._SessionLocal = original_engine, original_factory
        caller_engine.dispose()


def test_cli_does_not_open_or_modify_environment_database(tmp_path):
    # An invalid existing SQLite file would fail on import if the runner used
    # the env URL, and its exact bytes reveal any accidental replacement.
    db_path = tmp_path / "do-not-touch.db"
    sentinel = b"user database sentinel: this is deliberately not SQLite"
    db_path.write_bytes(sentinel)
    output = tmp_path / "report.json"
    env = {**os.environ, "SUPPORT_AGENT_DB_URL": f"sqlite:///{db_path}"}
    process = subprocess.run(
        [sys.executable, str(task_eval.ROOT / "eval/run_task_eval.py"),
         "--case", "T01", "--output", str(output)],
        env=env, cwd=task_eval.ROOT, capture_output=True, text=True, timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert db_path.read_bytes() == sentinel
    assert json.loads(output.read_text())["metrics"]["cost"]["estimated_amount"] is None


def test_judge_rejects_a_fabricated_success_trace_without_backend_execution():
    state = {"orders": {}, "payments": {}, "refunds": {}, "products": {}, "shipments": {}}
    result = {"run_id": "fabricated", "status": "SUCCEEDED", "reply": "已送达",
              "trace": [{"tool": "get_order_status", "args": {"order_id": 1001},
                         "result": {"order_id": 1001, "status": "delivered"}, "status": "SUCCEEDED"}],
              "events": [{"run_id": "fabricated", "sequence": 1, "type": "terminal"}]}
    failures = task_eval.judge_turn(_case("T01")["turns"][0]["expect"], result, state, state, [], 0)
    assert any("physical backend evidence" in failure for failure in failures)


def test_judge_checks_actual_database_changes_not_just_success_reply():
    before = {"payments": {"1003": {"status": "pending", "method": "card"}}}
    after = copy.deepcopy(before)
    result = {"run_id": "payment", "status": "SUCCEEDED", "reply": "付款成功",
              "trace": [], "events": [{"run_id": "payment", "sequence": 1, "type": "terminal"}]}
    expect = _case("T33")["turns"][1]["expect"]
    failures = task_eval.judge_turn(expect, result, before, after, [], 0)
    assert any("DB payments.1003.status" in failure for failure in failures)
    after["payments"]["1003"]["amount"] = 999
    failures = task_eval.judge_turn(expect, result, before, after, [], 0)
    assert any("unexpected business DB mutations" in failure for failure in failures)


def test_network_guard_fails_closed():
    with task_eval._offline_guard() as attempts:
        with pytest.raises(RuntimeError, match="Network access is disabled"):
            task_eval.socket.create_connection(("example.invalid", 443))
    assert attempts == ["network access attempted"]


def test_metrics_count_failed_model_attempts_and_never_invent_dollar_cost():
    report = task_eval.run_evaluation(task_eval.load_dataset(), ["T55", "T58", "T60"])
    assert report["metrics"]["model_acquisitions"] >= 3
    assert report["metrics"]["model_invocations"] >= 4
    assert report["metrics"]["cost"]["estimated_amount"] is None
    assert report["metrics"]["cost"]["currency"] is None
    assert report["metrics"]["cost"]["billable_provider_requests"] == 0
    duplicate = next(case for case in report["cases"] if case["id"] == "T58")
    assert duplicate["passed"]
    assert duplicate["turns"][0]["physical_tools"].count("get_order_status") == 1
    assert report["summary"]["total"] == 3
    assert all("usage_source" in turn["metrics"] for case in report["cases"] for turn in case["turns"])


def test_report_retains_wrong_expectations_as_failed_tasks():
    dataset = task_eval.load_dataset()
    case = _case("T01")
    case["turns"][0]["expect"]["tools"][0]["result"]["status"] = "delivered"
    dataset["cases"] = [case]
    report = task_eval.run_evaluation(dataset)
    assert report["summary"]["passed"] == 0
    assert report["summary"]["success_rate"] == 0
    assert report["cases"][0]["turns"][0]["failures"]
    assert report["summary"]["by_category"]["order"]["total"] == 1


def test_evaluation_gate_rejects_dataset_or_budget_regressions():
    baseline = {
        "schema_version": 1,
        "dataset_sha256": "expected",
        "expected": {"cases_passed": 2, "cases_total": 2, "turns_passed": 3, "turns_total": 3},
        "maximums": {"model_invocations": 4, "tool_calls": 5, "physical_tool_calls": 5},
    }
    report = {
        "dataset_sha256": "changed",
        "summary": {"passed": 1, "total": 2, "turns_passed": 2, "turns_total": 3},
        "metrics": {"model_invocations": 6, "tool_calls": 5, "physical_tool_calls": 5},
    }

    failures = task_eval.evaluate_gate(report, baseline)

    assert any("fingerprint" in failure for failure in failures)
    assert any("passed" in failure for failure in failures)
    assert any("model_invocations" in failure for failure in failures)
