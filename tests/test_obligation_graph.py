"""义务图只求解已持久化候选，不绕过政策、证据或授权边界。"""
from copy import deepcopy

from src.after_sales.execution import project_case_execution_state
from src.after_sales.obligation_graph import (
    build_obligation_graph,
    decide_next_obligation_action,
    solve_obligation_graph,
)
from src.after_sales.service import _checksum
from tests.test_after_sales_execution import _case_result


def _mixed_obligation_result():
    automatic = {
        "kind": "COLLECT_BUSINESS_EVIDENCE",
        "goal_id": "evidence:ORDER_FACT",
        "operation": "continue_after_sales_case",
        "requirement_id": "policy:v1:ORDER_FACT",
    }
    result = _case_result(automatic)
    customer_requirement = {
        "requirement_id": "policy:v1:CUSTOMER_FACT",
        "evidence_type": "CUSTOMER_FACT",
        "description": "请补充用户可提供的事实",
        "accepted_sources": ["CUSTOMER_STATEMENT"],
        "validation_condition": "CUSTOMER_ATTESTATION",
        "status": "UNSATISFIED",
        "evidence_refs": [],
    }
    customer_action = {
        "kind": "REQUEST_CUSTOMER_EVIDENCE",
        "goal_id": "evidence:CUSTOMER_FACT",
        "requirement_id": customer_requirement["requirement_id"],
        "question": "请补充用户可提供的事实",
    }
    result["policy_decision"]["facts"]["evidence_requirements"].append(customer_requirement)
    result["case"]["latest_decision"] = result["policy_decision"]
    plan = result["case"]["plans"][0]
    plan["goals"].insert(1, {
        "goal_id": "evidence:CUSTOMER_FACT",
        "requirement_id": customer_requirement["requirement_id"],
        "status": "UNSATISFIED",
        "depends_on": [],
        "evidence_refs": [],
    })
    plan["goals"][-1]["depends_on"].append("evidence:CUSTOMER_FACT")
    # The graph solver must not depend on persisted list order.
    plan["next_actions"] = [customer_action, automatic]
    return result


def test_graph_compiles_all_obligations_and_prefers_trusted_autonomous_progress():
    state = project_case_execution_state(_mixed_obligation_result())

    graph = build_obligation_graph(state)
    selection = solve_obligation_graph(graph)
    directive = decide_next_obligation_action(state, graph)

    assert graph.status == "READY"
    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    assert {row.mode for row in graph.candidates} == {"AUTONOMOUS", "ASK_USER"}
    assert selection.reason_code == "TRUSTED_AUTONOMOUS_ACTION_DOMINATES"
    assert len(selection.dominated_candidate_ids) == 1
    assert selection.selection_id == _checksum({
        "graph_id": graph.graph_id,
        "selected_candidate_id": selection.selected_candidate_id,
        "eligible_candidate_ids": selection.eligible_candidate_ids,
        "dominated_candidate_ids": selection.dominated_candidate_ids,
        "reason_code": selection.reason_code,
    })
    assert directive.status == "EXECUTE"
    assert directive.requirement_id == "policy:v1:ORDER_FACT"
    assert directive.arguments["expected_revision"] == 4
    assert directive.selection == selection


def test_information_gain_upper_bound_includes_resolution_for_last_unknown():
    question = {
        "kind": "REQUEST_CUSTOMER_EVIDENCE",
        "goal_id": "evidence:ORDER_FACT",
        "requirement_id": "policy:v1:ORDER_FACT",
        "question": "请补充说明",
    }
    state = project_case_execution_state(
        _case_result(question, condition="CUSTOMER_ATTESTATION", source="CUSTOMER_STATEMENT")
    )

    graph = build_obligation_graph(state)
    candidate = graph.candidates[0]
    directive = decide_next_obligation_action(state, graph)

    assert candidate.information_gain_upper_bound == 2
    assert candidate.unlocks == ("requirement:policy:v1:ORDER_FACT", "resolution")
    assert directive.status == "WAIT_USER_INPUT"
    assert directive.reason_code == "CUSTOMER_EVIDENCE_REQUIRED"
    assert directive.selection.reason_code == "HIGHEST_INFORMATION_GAIN_QUESTION"


def test_invalid_candidate_invalidates_graph_instead_of_falling_back_to_another_action():
    result = _mixed_obligation_result()
    forged = result["case"]["plans"][0]["next_actions"][1]
    forged["requirement_id"] = "policy:v1:CUSTOMER_FACT"
    forged["goal_id"] = "evidence:CUSTOMER_FACT"
    state = project_case_execution_state(result)

    graph = build_obligation_graph(state)
    directive = decide_next_obligation_action(state, graph)

    assert graph.status == "INVALID"
    assert "POLICY_OBLIGATION_NOT_AUTOMATABLE" in graph.validation_issues
    assert directive.status == "ABSTAINED"
    assert directive.reason_code == "POLICY_OBLIGATION_NOT_AUTOMATABLE"
    assert directive.candidate_id is None


def test_malformed_persisted_action_becomes_a_structured_invalid_graph():
    result = _case_result({
        "kind": "REQUEST_CUSTOMER_EVIDENCE",
        "goal_id": "evidence:ORDER_FACT",
        "requirement_id": "policy:v1:ORDER_FACT",
        "question": {"untrusted": True},
    }, condition="CUSTOMER_ATTESTATION", source="CUSTOMER_STATEMENT")
    state = project_case_execution_state(result)

    graph = build_obligation_graph(state)

    assert graph.status == "INVALID"
    assert graph.candidates[0].admissible is False
    assert graph.candidates[0].question is None
    assert graph.reason_code == "CUSTOMER_EVIDENCE_BINDING_INVALID"


def test_boolean_lookalike_cannot_bypass_no_approval_boundary():
    action = {
        "kind": "COLLECT_BUSINESS_EVIDENCE",
        "goal_id": "evidence:ORDER_FACT",
        "operation": "continue_after_sales_case",
        "requirement_id": "policy:v1:ORDER_FACT",
        "requires_approval": 1,
    }
    state = project_case_execution_state(_case_result(action))

    graph = build_obligation_graph(state)

    assert graph.status == "INVALID"
    assert graph.candidates[0].mode == "INVALID"
    assert graph.reason_code == "POLICY_OBLIGATION_NOT_AUTOMATABLE"


def test_missing_action_for_unmet_obligation_is_a_process_level_failure():
    result = _case_result({
        "kind": "REQUEST_CUSTOMER_EVIDENCE",
        "goal_id": "evidence:ORDER_FACT",
        "requirement_id": "policy:v1:ORDER_FACT",
        "question": "请补充说明",
    }, condition="CUSTOMER_ATTESTATION", source="CUSTOMER_STATEMENT")
    broken = deepcopy(result)
    broken["case"]["plans"][0]["next_actions"] = []
    state = project_case_execution_state(broken)

    graph = build_obligation_graph(state)
    directive = decide_next_obligation_action(state, graph)

    assert graph.status == "INVALID"
    assert graph.reason_code == "OBLIGATION_ACTION_MISSING:policy:v1:ORDER_FACT"
    assert directive.status == "ABSTAINED"
    assert directive.reason_code == graph.reason_code


def test_deferred_claim_is_preserved_without_reasking_user():
    claim_id = "b" * 64
    wait = {
        "kind": "WAIT_FOR_CUSTOMER_EVIDENCE",
        "goal_id": "evidence:ORDER_FACT",
        "requirement_id": "policy:v1:ORDER_FACT",
        "claim_id": claim_id,
        "disposition": "DEFERRED",
        "question": "案件已保留；准备好后请补充说明",
    }
    result = _case_result(wait, condition="CUSTOMER_ATTESTATION", source="CUSTOMER_STATEMENT")
    result["case"]["plans"][0]["goals"][0]["readiness"] = {
        "schema_version": 1,
        "projector_version": "evidence-readiness-v1",
        "claim_id": claim_id,
        "requirement_id": "policy:v1:ORDER_FACT",
        "evidence_type": "ORDER_FACT",
        "disposition": "DEFERRED",
        "source_quote": "稍后才能确认",
        "source_ref": "agent_runs:run-1:request",
        "policy_version_id": "policy:v1",
        "policy_checksum": "a" * 64,
        "case_revision": 3,
        "recorded_at": "2026-10-26T04:00:00",
    }
    state = project_case_execution_state(result)

    graph = build_obligation_graph(state)
    directive = decide_next_obligation_action(state, graph)

    assert graph.status == "READY"
    assert graph.candidates[0].mode == "WAIT"
    assert directive.status == "WAIT_USER_INPUT"
    assert directive.reason_code == "CUSTOMER_EVIDENCE_DEFERRED"
    assert directive.selection.reason_code == "DEFERRED_OBLIGATION_PRESERVED"


def test_eligible_resolution_requires_a_bound_approval_candidate():
    approval = {
        "kind": "REQUEST_CUSTOMER_CONFIRMATION",
        "goal_id": "resolution",
        "operation": "submit_after_sales_case",
        "requires_approval": True,
    }
    result = _case_result(approval)
    requirement = result["policy_decision"]["facts"]["evidence_requirements"][0]
    requirement["status"] = "SATISFIED"
    result["policy_decision"].update(outcome="ELIGIBLE", reason_codes=[], allowed_actions=["SUBMIT"])
    result["case"]["latest_decision"] = result["policy_decision"]
    result["case_status"] = result["case"]["status"] = "AWAITING_CONFIRMATION"
    result["case"]["plans"][0]["status"] = "AWAITING_CONFIRMATION"
    result["case"]["plans"][0]["goals"][0]["status"] = "SATISFIED"
    result["requires_confirmation"] = True
    state = project_case_execution_state(result)

    graph = build_obligation_graph(state)
    directive = decide_next_obligation_action(state, graph)

    assert graph.status == "READY"
    assert directive.status == "WAIT_APPROVAL"
    assert directive.requires_approval is True
    assert directive.operation is None

    missing = deepcopy(result)
    missing["case"]["plans"][0]["next_actions"] = []
    missing_state = project_case_execution_state(missing)
    missing_graph = build_obligation_graph(missing_state)
    assert missing_graph.status == "INVALID"
    assert missing_graph.reason_code == "OBLIGATION_APPROVAL_ACTION_MISSING"
    assert decide_next_obligation_action(missing_state, missing_graph).status == "ABSTAINED"


def test_manual_review_accepts_root_boundary_without_inventing_evidence_work():
    result = _case_result({
        "kind": "REQUIRES_HUMAN_CAPABILITY",
        "goal_id": "resolution",
    })
    result["policy_decision"].update(
        outcome="MANUAL_REVIEW",
        reason_codes=["REFUND_SCOPE_REQUIRES_LINE_ITEM_EXECUTION"],
        allowed_actions=["ESCALATE_TO_HUMAN"],
    )
    result["case"]["latest_decision"] = result["policy_decision"]
    result["case_status"] = result["case"]["status"] = "MANUAL_REVIEW"
    result["case"]["plans"][0]["status"] = "MANUAL_REVIEW"
    state = project_case_execution_state(result)

    graph = build_obligation_graph(state)
    directive = decide_next_obligation_action(state, graph)

    assert graph.status == "READY"
    assert directive.status == "BLOCKED"
    assert directive.reason_code == "UNSUPPORTED_CAPABILITY"
    assert directive.requirement_id is None


def test_legacy_capability_action_resolves_one_stable_evidence_goal():
    result = _case_result(
        {"kind": "REQUIRES_HUMAN_CAPABILITY", "goal_id": "evidence:ORDER_FACT"},
        condition="UNSUPPORTED",
    )
    state = project_case_execution_state(result)

    graph = build_obligation_graph(state)
    directive = decide_next_obligation_action(state, graph)

    assert graph.status == "READY"
    assert graph.candidates[0].requirement_id == "policy:v1:ORDER_FACT"
    assert directive.status == "BLOCKED"
    assert directive.requirement_id == "policy:v1:ORDER_FACT"


def test_policy_outcome_must_match_the_nonterminal_case_status():
    result = _case_result({
        "kind": "COLLECT_BUSINESS_EVIDENCE",
        "goal_id": "evidence:ORDER_FACT",
        "operation": "continue_after_sales_case",
        "requirement_id": "policy:v1:ORDER_FACT",
    })
    result["policy_decision"]["outcome"] = "INELIGIBLE"
    result["case"]["latest_decision"] = result["policy_decision"]
    state = project_case_execution_state(result)

    graph = build_obligation_graph(state)
    directive = decide_next_obligation_action(state, graph)

    assert graph.status == "INVALID"
    assert graph.reason_code == "POLICY_OUTCOME_CASE_STATUS_MISMATCH"
    assert directive.status == "ABSTAINED"
