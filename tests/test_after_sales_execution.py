"""政策感知自主执行只消费已持久化计划，不放宽政策或授权边界。"""
from copy import deepcopy

import pytest

from src.after_sales.execution import decide_next_case_action, project_case_execution_state


def _case_result(action, *, condition="AUTHORITATIVE_ORDER", source="ORDER_RECORD", valid=True):
    requirement = {
        "requirement_id": "policy:v1:ORDER_FACT",
        "evidence_type": "ORDER_FACT",
        "description": "读取权威订单状态",
        "accepted_sources": [source],
        "validation_condition": condition,
        "status": "UNSATISFIED",
        "evidence_refs": [],
    }
    plan = {
        "schema_version": 1,
        "planner_version": "evidence-driven-case-plan-v1",
        "plan_id": "plan-1",
        "revision": 1,
        "case_revision": 4,
        "decision_id": "decision-1",
        "policy_version_id": "policy:v1",
        "status": "AWAITING_EVIDENCE",
        "valid": valid,
        "goals": [
            {
                "goal_id": "evidence:ORDER_FACT",
                "requirement_id": requirement["requirement_id"],
                "status": "UNSATISFIED",
                "depends_on": [],
                "evidence_refs": [],
            },
            {
                "goal_id": "resolution",
                "status": "AWAITING_EVIDENCE",
                "depends_on": ["evidence:ORDER_FACT"],
                "evidence_refs": [],
            },
        ],
        "next_actions": [action],
    }
    decision = {
        "decision_id": "decision-1",
        "policy_version_id": "policy:v1",
        "policy_checksum": "a" * 64,
        "outcome": "NEEDS_EVIDENCE",
        "reason_codes": ["MISSING_EVIDENCE:ORDER_FACT"],
        "allowed_actions": ["ADD_EVIDENCE"],
        "evidence_refs": ["orders:1001"],
        "facts": {"evidence_requirements": [requirement]},
        "valid": True,
    }
    return {
        "case_id": "case-1",
        "order_id": 1001,
        "case_status": "AWAITING_EVIDENCE",
        "revision": 4,
        "policy_decision": decision,
        "requires_confirmation": False,
        "case": {
            "case_id": "case-1",
            "order_id": 1001,
            "status": "AWAITING_EVIDENCE",
            "revision": 4,
            "latest_decision": decision,
            "plans": [plan],
        },
        "data_source": "AFTER_SALES_CASE_LEDGER",
    }


def test_state_projection_binds_case_policy_and_plan_identity():
    action = {
        "kind": "COLLECT_BUSINESS_EVIDENCE",
        "goal_id": "evidence:ORDER_FACT",
        "operation": "continue_after_sales_case",
        "requirement_id": "policy:v1:ORDER_FACT",
    }
    result = _case_result(action)

    state = project_case_execution_state(result)
    repeated = project_case_execution_state(result)
    changed = deepcopy(result)
    changed["revision"] = changed["case"]["revision"] = 5

    assert state.status == "READY"
    assert state.state_id == repeated.state_id
    assert state.case_revision == 4
    assert state.policy_decision_id == "decision-1"
    assert state.plan_id == "plan-1"
    assert state.obligations[0].validation_condition == "AUTHORITATIVE_ORDER"
    assert project_case_execution_state(changed).state_id != state.state_id


def test_planner_executes_only_policy_bound_authoritative_read():
    action = {
        "kind": "COLLECT_BUSINESS_EVIDENCE",
        "goal_id": "evidence:ORDER_FACT",
        "operation": "continue_after_sales_case",
        "requirement_id": "policy:v1:ORDER_FACT",
    }
    directive = decide_next_case_action(project_case_execution_state(_case_result(action)))

    assert directive.status == "EXECUTE"
    assert directive.reason_code == "AUTHORITATIVE_EVIDENCE_AVAILABLE"
    assert directive.operation == "continue_after_sales_case"
    assert directive.arguments == {
        "case_id": "case-1",
        "expected_revision": 4,
        "requirement_id": "policy:v1:ORDER_FACT",
    }
    assert directive.requires_approval is False


@pytest.mark.parametrize(
    ("action", "expected_status", "expected_reason"),
    [
        (
            {"kind": "REQUEST_CUSTOMER_EVIDENCE", "goal_id": "evidence:ORDER_FACT",
             "requirement_id": "policy:v1:ORDER_FACT", "question": "请补充说明"},
            "WAIT_USER_INPUT",
            "CUSTOMER_EVIDENCE_REQUIRED",
        ),
        (
            {"kind": "REQUEST_CUSTOMER_CONFIRMATION", "goal_id": "resolution",
             "operation": "submit_after_sales_case", "requires_approval": True},
            "WAIT_APPROVAL",
            "EXPLICIT_APPROVAL_REQUIRED",
        ),
        (
            {"kind": "REQUIRES_HUMAN_CAPABILITY", "goal_id": "resolution"},
            "BLOCKED",
            "UNSUPPORTED_CAPABILITY",
        ),
    ],
)
def test_planner_stops_at_user_approval_or_unsupported_boundaries(action, expected_status, expected_reason):
    directive = decide_next_case_action(project_case_execution_state(_case_result(action)))

    assert directive.status == expected_status
    assert directive.reason_code == expected_reason
    assert directive.operation is None


def test_planner_rejects_a_forged_automatic_action_for_customer_attestation():
    action = {
        "kind": "COLLECT_BUSINESS_EVIDENCE",
        "goal_id": "evidence:ORDER_FACT",
        "operation": "continue_after_sales_case",
        "requirement_id": "policy:v1:ORDER_FACT",
    }
    state = project_case_execution_state(
        _case_result(action, condition="CUSTOMER_ATTESTATION", source="CUSTOMER_STATEMENT")
    )

    directive = decide_next_case_action(state)

    assert directive.status == "ABSTAINED"
    assert directive.reason_code == "POLICY_OBLIGATION_NOT_AUTOMATABLE"


def test_stale_case_plan_never_drives_execution():
    action = {
        "kind": "COLLECT_BUSINESS_EVIDENCE",
        "goal_id": "evidence:ORDER_FACT",
        "operation": "continue_after_sales_case",
        "requirement_id": "policy:v1:ORDER_FACT",
    }
    state = project_case_execution_state(_case_result(action, valid=False))

    assert state.status == "STALE"
    directive = decide_next_case_action(state)
    assert directive.status == "ABSTAINED"
    assert directive.reason_code == "CASE_STATE_NOT_EXECUTABLE"
