import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval.semantic_contracts import SemanticCase, load_dataset_payload
from eval.semantic_oracle import _tool_records, apply_mode_consistency, score_semantic_result


ROOT = Path(__file__).resolve().parents[1]


def _case(case_id):
    dataset = load_dataset_payload(json.loads((ROOT / "eval/semantic_cases.json").read_text()))
    return next(case for case in dataset.cases if case.id == case_id)


def _result(objectives, operations, *, status="SUCCEEDED", reason="COMPLETED", commands=()):
    run_id = "semantic-oracle-test"
    events = []

    def event(event_type, **payload):
        events.append({"run_id": run_id, "sequence": len(events) + 1, "type": event_type, **payload})

    event("model_started", purpose="dialogue_understanding", agent="intake")
    event("model_call", purpose="dialogue_understanding", agent="intake", status="SUCCEEDED",
          provider="fixture", model="semantic-fixture")
    steps = []
    for index, row in enumerate(operations, start=1):
        execution_id = f"execution-{index}"
        fingerprint = f"fingerprint-{index}"
        policy = {"effect": "READ", "risk_level": "READ_ONLY"}
        event("tool_call", tool=row["tool"], args=row["args"], execution_id=execution_id,
              action_fingerprint=fingerprint, policy=policy)
        event("tool_result", tool=row["tool"], args=row["args"], result=row["result"],
              status=row.get("status", "SUCCEEDED"), execution_id=execution_id,
              action_fingerprint=fingerprint, policy=policy)
        steps.append({
            "step_id": f"step-{index}", "tool": row["tool"], "args": row["args"],
            "result": row["result"], "status": row.get("status", "SUCCEEDED"),
            "capability_id": row.get("capability_id"), "objective_id": row.get("objective_id"),
            "objective_ids": [row["objective_id"]] if row.get("objective_id") else [],
        })
    outcomes = [{"objective_id": item["id"], "outcome": "SATISFIED", "step_ids": []}
                for item in objectives]
    verification = {"goal_outcomes": outcomes, "all_goals_satisfied": status == "SUCCEEDED",
                    "side_effects_executed": False}
    event("verification", **verification)
    event("terminal", status=status, reason=reason)
    return {
        "status": status,
        "terminal_reason": reason,
        "understanding": {"objectives": objectives, "commands": [{"kind": item} for item in commands]},
        "plan": {"steps": steps, "verification": verification},
        "events": events,
        "budget": {"limits": {"max_model_calls": 4}, "consumed": {"model_calls": 1}},
    }


def test_dataset_requires_both_identity_and_capabilities_commands():
    case = _case("identity-command")
    result = _result([], [], commands=("IDENTITY",))

    score = score_semantic_result(case, result, mode="auto").payload

    assert not score["passed"]
    assert "COMMAND_MISSING" in score["failure_codes"]


def test_dataset_rejects_indistinguishable_same_kind_goals_and_unknown_assertions():
    payload = json.loads((ROOT / "eval/semantic_cases.json").read_text())
    case = next(item for item in payload["cases"] if item["id"] == "two-timezones-two-objectives")
    case["oracle"]["goals"][1]["objective"] = case["oracle"]["goals"][0]["objective"]
    with pytest.raises(ValidationError, match="distinguishable"):
        load_dataset_payload(payload)

    payload = json.loads((ROOT / "eval/semantic_cases.json").read_text())
    payload["cases"][0]["oracle"]["goals"][0]["objective"][0]["operator"] = "python_eval"
    with pytest.raises(ValidationError, match="literal"):
        load_dataset_payload(payload)


def test_two_same_kind_goals_match_one_to_one_by_normalized_entity():
    case = _case("two-timezones-two-objectives")
    objectives = [
        {"id": "london", "kind": "SYSTEM_TIME", "timezone": "Europe/London"},
        {"id": "new-york", "kind": "SYSTEM_TIME", "timezone": "America/New_York"},
    ]
    operations = [
        {"tool": "read_clock", "args": {"timezone": "Europe/London"},
         "result": {"timezone": "Europe/London", "data_source": "RUNTIME_CLOCK"},
         "capability_id": "system.clock.read", "objective_id": "london"},
        {"tool": "read_clock", "args": {"timezone": "America/New_York"},
         "result": {"timezone": "America/New_York", "data_source": "RUNTIME_CLOCK"},
         "capability_id": "system.clock.read", "objective_id": "new-york"},
    ]

    score = score_semantic_result(case, _result(objectives, operations), mode="auto").payload

    assert score["passed"], score
    assert {item["actual_objective_id"] for item in score["goal_results"]} == {"london", "new-york"}


def test_same_kind_goal_with_wrong_second_timezone_cannot_pass_by_count():
    case = _case("two-timezones-two-objectives")
    objectives = [
        {"id": "new-york", "kind": "SYSTEM_TIME", "timezone": "America/New_York"},
        {"id": "los-angeles", "kind": "SYSTEM_TIME", "timezone": "America/Los_Angeles"},
    ]
    operations = [
        {"tool": "read_clock", "args": {"timezone": item["timezone"]},
         "result": {"timezone": item["timezone"], "data_source": "RUNTIME_CLOCK"},
         "capability_id": "system.clock.read", "objective_id": item["id"]}
        for item in objectives
    ]

    score = score_semantic_result(case, _result(objectives, operations), mode="auto").payload

    assert not score["passed"]
    assert "ENTITY_MISMATCH" in score["failure_codes"]
    assert "UNEXPECTED_GOAL" in score["failure_codes"]


def test_tool_name_without_matching_arguments_or_result_is_not_evidence():
    case = _case("trusted-timezone-normalization")
    result = _result(
        [{"id": "time", "kind": "SYSTEM_TIME", "timezone": "America/New_York"}],
        [{"tool": "read_clock", "args": {"timezone": "America/Los_Angeles"},
          "result": {"timezone": "America/Los_Angeles", "data_source": "RUNTIME_CLOCK", "observed_at": "now"},
          "capability_id": "system.clock.read", "objective_id": "time"}],
    )

    score = score_semantic_result(case, result, mode="auto").payload

    assert not score["passed"]
    assert "NORMALIZED_PARAMETER_MISMATCH" in score["failure_codes"]


def test_forbidden_operation_is_read_from_physical_events_not_presentation_trace():
    case = _case("negated-payment-query")
    result = _result(
        [{"id": "payment", "kind": "PAYMENT_STATUS", "order_id": 1001}],
        [{"tool": "get_payment_status", "args": {"order_id": 1001},
          "result": {"order_id": 1001, "status": "pending"},
          "capability_id": "payment.status.read", "objective_id": "payment"}],
    )
    sequence = len(result["events"])
    result["events"][-1:-1] = [
        {"run_id": "semantic-oracle-test", "sequence": sequence, "type": "tool_call",
         "tool": "process_payment", "args": {"order_id": 1001, "method": "card"},
         "execution_id": "forbidden", "action_fingerprint": "forbidden", "policy": {"effect": "WRITE"}},
        {"run_id": "semantic-oracle-test", "sequence": sequence + 1, "type": "tool_result",
         "tool": "process_payment", "args": {"order_id": 1001, "method": "card"},
         "result": {"success": True}, "status": "SUCCEEDED", "execution_id": "forbidden",
         "action_fingerprint": "forbidden", "policy": {"effect": "WRITE"}},
    ]
    for index, event in enumerate(result["events"], start=1):
        event["sequence"] = index

    score = score_semantic_result(case, result, mode="auto").payload

    assert "FORBIDDEN_OPERATION_EXECUTED" in score["failure_codes"]
    assert "FORBIDDEN_SIDE_EFFECT" in score["failure_codes"]


def test_auto_adaptive_consistency_compares_goal_projection():
    samples = [
        {"case_id": "clock", "repeat": 1, "mode": "auto", "passed": True,
         "failure_codes": [], "failures": [], "semantic_projection": {
             "goals": [{"objective_kind": "SYSTEM_TIME", "objective": {"timezone": "Asia/Tokyo"},
                        "outcome": "SATISFIED"}], "commands": [], "status": "SUCCEEDED", "terminal_reason": "COMPLETED"}},
        {"case_id": "clock", "repeat": 1, "mode": "adaptive", "passed": True,
         "failure_codes": [], "failures": [], "semantic_projection": {
             "goals": [{"objective_kind": "SYSTEM_TIME", "objective": {"timezone": "Europe/London"},
                        "outcome": "SATISFIED"}], "commands": [], "status": "SUCCEEDED", "terminal_reason": "COMPLETED"}},
    ]

    consistency = apply_mode_consistency(samples)

    assert consistency == [False]
    assert not any(sample["passed"] for sample in samples)
    assert all("AUTO_ADAPTIVE_SEMANTIC_MISMATCH" in sample["failure_codes"] for sample in samples)


def test_allowed_fast_path_uses_actual_tool_evidence_without_fake_model_call():
    case = _case("history-order-pronoun")
    result = _result(
        [{"id": "shipment", "kind": "SHIPMENT_TRACKING", "order_id": 1002}],
        [{"tool": "track_shipment", "args": {"order_id": 1002},
          "result": {"order_id": 1002, "status": "shipped"},
          "capability_id": "shipment.track", "objective_id": "shipment"}],
    )
    result["events"] = [event for event in result["events"] if event["type"] not in {"model_started", "model_call"}]
    for index, event in enumerate(result["events"], start=1):
        event["sequence"] = index

    score = score_semantic_result(case, result, mode="auto").payload

    assert score["passed"], score
    assert not score["false_fast_path"]


def test_recovered_tool_result_requires_matching_ledger_and_scope_evidence():
    case = _case("trusted-timezone-normalization")
    result = _result(
        [{"id": "time", "kind": "SYSTEM_TIME", "timezone": "America/New_York"}],
        [{"tool": "read_clock", "args": {"timezone": "America/New_York"},
          "result": {"timezone": "America/New_York", "data_source": "RUNTIME_CLOCK",
                     "observed_at": "now"},
          "capability_id": "system.clock.read", "objective_id": "time"}],
    )
    call = next(event for event in result["events"] if event["type"] == "tool_call")
    tool_result = next(event for event in result["events"] if event["type"] == "tool_result")
    call.update(ledger_key="ledger-proof", observation_scope_sha256="scope-proof")
    tool_result.update(
        status="REUSED", reused=True, source_ledger="ledger-proof",
        observation_scope_sha256="scope-proof",
    )

    score = score_semantic_result(case, result, mode="auto").payload

    assert score["passed"], score
    tool_result["source_ledger"] = "forged-ledger"
    forged = score_semantic_result(case, result, mode="auto").payload
    assert not forged["passed"]
    assert "AUTHORITATIVE_EVIDENCE_INVALID" in forged["failure_codes"]


def test_same_run_reuse_points_to_an_earlier_complete_execution_and_same_scope():
    events = [
        {"run_id": "run", "sequence": 1, "type": "tool_call", "tool": "read_clock",
         "args": {"timezone": "Asia/Shanghai"}, "execution_id": "original",
         "action_fingerprint": "fingerprint", "observation_scope_sha256": "scope"},
        {"run_id": "run", "sequence": 2, "type": "tool_result", "tool": "read_clock",
         "args": {"timezone": "Asia/Shanghai"}, "result": {"timezone": "Asia/Shanghai"},
         "status": "SUCCEEDED", "execution_id": "original", "action_fingerprint": "fingerprint",
         "observation_scope_sha256": "scope"},
        {"run_id": "run", "sequence": 3, "type": "tool_result", "tool": "read_clock",
         "args": {"timezone": "Asia/Shanghai"}, "result": {"timezone": "Asia/Shanghai"},
         "status": "REUSED", "reused": True, "execution_id": None,
         "source_execution_id": "original", "source_sequence": 2,
         "source_ledger": None, "action_fingerprint": "fingerprint",
         "observation_scope_sha256": "scope"},
    ]

    records, failures = _tool_records(events)

    assert failures == []
    assert len(records) == 2
    assert records[1]["execution_id"] == "original:reuse:3"
    events[-1]["observation_scope_sha256"] = "different-scope"
    _, failures = _tool_records(events)
    assert failures == ["AUTHORITATIVE_EVIDENCE_INVALID"]
