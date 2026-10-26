"""主动澄清只排序已绑定问题，不预测用户或发明政策事实。"""
from copy import deepcopy

from src.after_sales.clarification import (
    build_clarification_portfolio,
    solve_active_clarification,
)
from src.after_sales.execution import project_case_execution_state
from src.after_sales.obligation_graph import (
    ACTIVE_CLARIFICATION_PLANNER_VERSION,
    build_obligation_graph,
    decide_next_obligation_action,
)
from tests.test_after_sales_execution import _case_result


def _semantic(
    *,
    requirement_id: str,
    status: str,
    missing: list[str],
    question: str,
    assessment_id: str,
) -> dict:
    return {
        "requirement_id": requirement_id,
        "status": status,
        "citations": [],
        "missing_information": missing,
        "explanation": "测试中的结构化语义缺口",
        "question": question,
        "assessment_id": assessment_id,
        "verifier_version": "evidence-semantic-v1",
        "error_code": None,
    }


def _two_question_result():
    first_question = "请说明原包装是否完整，以及附件是否齐全？"
    first_action = {
        "kind": "REQUEST_CUSTOMER_EVIDENCE",
        "goal_id": "evidence:ORDER_FACT",
        "requirement_id": "policy:v1:ORDER_FACT",
        "question": first_question,
        "semantic_status": "INCOMPLETE",
    }
    result = _case_result(
        first_action,
        condition="CUSTOMER_ATTESTATION",
        source="CUSTOMER_STATEMENT",
    )
    first_requirement = result["policy_decision"]["facts"]["evidence_requirements"][0]
    first_requirement["description"] = "需要确认原包装和附件状态"
    first_requirement["semantic_assessment"] = _semantic(
        requirement_id=first_requirement["requirement_id"],
        status="INCOMPLETE",
        missing=["原包装是否完整", "附件是否齐全"],
        question=first_question,
        assessment_id="assessment-packaging",
    )

    second_requirement = {
        "requirement_id": "policy:v1:SERIAL_LABEL",
        "evidence_type": "SERIAL_LABEL",
        "description": "需要确认序列号标签是否清晰",
        "accepted_sources": ["CUSTOMER_STATEMENT"],
        "validation_condition": "CUSTOMER_ATTESTATION",
        "status": "UNSATISFIED",
        "evidence_refs": [],
        "semantic_assessment": _semantic(
            requirement_id="policy:v1:SERIAL_LABEL",
            status="UNKNOWN",
            missing=["序列号标签是否清晰"],
            question="请确认序列号标签是否清晰？",
            assessment_id="assessment-serial",
        ),
    }
    second_action = {
        "kind": "REQUEST_CUSTOMER_EVIDENCE",
        "goal_id": "evidence:SERIAL_LABEL",
        "requirement_id": second_requirement["requirement_id"],
        "question": "请确认序列号标签是否清晰？",
        "semantic_status": "UNKNOWN",
    }
    result["policy_decision"]["facts"]["evidence_requirements"].append(second_requirement)
    result["case"]["latest_decision"] = result["policy_decision"]
    plan = result["case"]["plans"][0]
    plan["goals"].insert(1, {
        "goal_id": "evidence:SERIAL_LABEL",
        "requirement_id": second_requirement["requirement_id"],
        "status": "UNSATISFIED",
        "depends_on": [],
        "evidence_refs": [],
    })
    plan["goals"][-1]["depends_on"].append("evidence:SERIAL_LABEL")
    # Selection must not inherit the persisted list order.
    plan["next_actions"] = [second_action, first_action]
    return result


def test_active_clarification_selects_structural_information_value_without_probabilities():
    result = _two_question_result()
    state = project_case_execution_state(result)
    graph = build_obligation_graph(state)

    portfolio = build_clarification_portfolio(state, graph, result)
    decision = solve_active_clarification(graph, portfolio)
    directive = decide_next_obligation_action(
        state,
        graph,
        selection=decision.candidate_selection,
        planner_version=ACTIVE_CLARIFICATION_PLANNER_VERSION,
    )

    assert portfolio.status == "READY"
    assert len(portfolio.questions) == 2
    selected = next(row for row in portfolio.questions if row.question_id == decision.selected_question_id)
    assert selected.requirement_id == "policy:v1:ORDER_FACT"
    assert [need.text for need in selected.needs] == ["原包装是否完整", "附件是否齐全"]
    assert selected.reported_need_count == 2
    assert selected.covered_need_count == 2
    assert selected.information_gain_upper_bound == 2
    assert {row.outcome for row in selected.answer_branches} == {
        "SUFFICIENT_EVIDENCE",
        "PARTIAL_EVIDENCE",
        "EXPLICIT_DEFERRAL",
        "UNRESOLVED_OR_CONFLICT",
    }
    assert decision.status == "SELECTED"
    assert decision.reason_code == "ACTIVE_CLARIFICATION_MAX_STRUCTURAL_VALUE"
    assert decision.candidate_selection.solver_version == "active-clarification-solver-v1"
    assert directive.status == "WAIT_USER_INPUT"
    assert directive.question == "请说明原包装是否完整，以及附件是否齐全？"
    assert directive.planner_version == ACTIVE_CLARIFICATION_PLANNER_VERSION
    dumped = portfolio.model_dump(mode="json")
    assert "probability" not in str(dumped).lower()
    assert "entropy" not in str(dumped).lower()


def test_uncovered_reported_gaps_cannot_inflate_information_value():
    result = _two_question_result()
    requirement = result["policy_decision"]["facts"]["evidence_requirements"][0]
    requirement["semantic_assessment"]["missing_information"].append("问题没有询问的住址")
    result["case"]["latest_decision"] = result["policy_decision"]
    state = project_case_execution_state(result)
    graph = build_obligation_graph(state)

    portfolio = build_clarification_portfolio(state, graph, result)

    question = next(
        row for row in portfolio.questions
        if row.requirement_id == "policy:v1:ORDER_FACT"
    )
    assert portfolio.status == "READY"
    assert question.reported_need_count == 3
    assert question.covered_need_count == 2
    assert question.information_gain_upper_bound == 2
    assert [need.text for need in question.needs] == ["原包装是否完整", "附件是否齐全"]


def test_trusted_autonomous_action_bypasses_question_selection():
    result = _two_question_result()
    automatic = {
        "kind": "COLLECT_BUSINESS_EVIDENCE",
        "goal_id": "evidence:ORDER_FACT",
        "operation": "continue_after_sales_case",
        "requirement_id": "policy:v1:ORDER_FACT",
    }
    first = result["policy_decision"]["facts"]["evidence_requirements"][0]
    first.update(
        validation_condition="AUTHORITATIVE_ORDER",
        accepted_sources=["ORDER_RECORD"],
        semantic_assessment=None,
    )
    result["case"]["plans"][0]["next_actions"][1] = automatic
    state = project_case_execution_state(result)
    graph = build_obligation_graph(state)

    portfolio = build_clarification_portfolio(state, graph, result)
    decision = solve_active_clarification(graph, portfolio)

    assert portfolio.status == "READY"
    assert len(portfolio.questions) == 1
    assert decision.status == "BYPASSED"
    assert decision.selected_question_id is None
    assert decision.candidate_selection.reason_code == "TRUSTED_AUTONOMOUS_ACTION_DOMINATES"
    selected = next(
        row for row in graph.candidates
        if row.candidate_id == decision.candidate_selection.selected_candidate_id
    )
    assert selected.mode == "AUTONOMOUS"


def test_semantic_question_must_match_the_persisted_plan_action():
    result = _two_question_result()
    requirement = result["policy_decision"]["facts"]["evidence_requirements"][0]
    requirement["semantic_assessment"]["question"] = "这是没有写入计划的另一个问题"
    result["case"]["latest_decision"] = result["policy_decision"]
    state = project_case_execution_state(result)
    graph = build_obligation_graph(state)

    portfolio = build_clarification_portfolio(state, graph, result)
    decision = solve_active_clarification(graph, portfolio)

    assert portfolio.status == "INVALID"
    assert "CLARIFICATION_QUESTION_BINDING_INVALID:policy:v1:ORDER_FACT" in portfolio.validation_issues
    assert decision.status == "INVALID"
    assert decision.candidate_selection is None


def test_deferred_obligation_is_not_compiled_as_a_question():
    result = _two_question_result()
    claim_id = "b" * 64
    serial_goal = next(
        goal for goal in result["case"]["plans"][0]["goals"]
        if goal.get("requirement_id") == "policy:v1:SERIAL_LABEL"
    )
    serial_goal["readiness"] = {
        "schema_version": 1,
        "projector_version": "evidence-readiness-v1",
        "claim_id": claim_id,
        "requirement_id": "policy:v1:SERIAL_LABEL",
        "evidence_type": "SERIAL_LABEL",
        "disposition": "DEFERRED",
        "source_quote": "序列号稍后确认",
        "source_ref": "agent_runs:run-1:request",
        "policy_version_id": "policy:v1",
        "policy_checksum": "a" * 64,
        "case_revision": 3,
        "recorded_at": "2026-10-26T04:00:00",
    }
    result["case"]["plans"][0]["next_actions"][0] = {
        "kind": "WAIT_FOR_CUSTOMER_EVIDENCE",
        "goal_id": "evidence:SERIAL_LABEL",
        "requirement_id": "policy:v1:SERIAL_LABEL",
        "claim_id": claim_id,
        "disposition": "DEFERRED",
        "question": "案件已保留；准备好后请补充序列号标签情况",
    }
    state = project_case_execution_state(result)
    graph = build_obligation_graph(state)

    portfolio = build_clarification_portfolio(state, graph, result)
    decision = solve_active_clarification(graph, portfolio)

    assert portfolio.status == "READY"
    assert [row.requirement_id for row in portfolio.questions] == ["policy:v1:ORDER_FACT"]
    assert decision.status == "SELECTED"
    assert decision.selected_question_id == portfolio.questions[0].question_id


def test_malformed_semantic_gap_invalidates_portfolio_without_throwing():
    result = _two_question_result()
    requirement = result["policy_decision"]["facts"]["evidence_requirements"][0]
    requirement["semantic_assessment"]["missing_information"] = {"unexpected": True}
    result["case"]["latest_decision"] = result["policy_decision"]
    state = project_case_execution_state(result)
    graph = build_obligation_graph(state)

    portfolio = build_clarification_portfolio(state, graph, result)

    assert portfolio.status == "INVALID"
    assert "CLARIFICATION_NEEDS_INVALID:policy:v1:ORDER_FACT" in portfolio.validation_issues


def test_malformed_requirement_set_invalidates_portfolio_without_throwing():
    result = _two_question_result()
    state = project_case_execution_state(result)
    graph = build_obligation_graph(state)
    result["policy_decision"]["facts"]["evidence_requirements"] = {"unexpected": True}
    result["case"]["latest_decision"] = result["policy_decision"]

    portfolio = build_clarification_portfolio(state, graph, result)

    assert portfolio.status == "INVALID"
    assert "CLARIFICATION_REQUIREMENT_SET_INVALID" in portfolio.validation_issues


def test_selection_behavior_does_not_inherit_persisted_action_order():
    left = _two_question_result()
    right = deepcopy(left)
    right["case"]["plans"][0]["next_actions"].reverse()

    left_state = project_case_execution_state(left)
    right_state = project_case_execution_state(right)
    left_graph = build_obligation_graph(left_state)
    right_graph = build_obligation_graph(right_state)
    left_portfolio = build_clarification_portfolio(left_state, left_graph, left)
    right_portfolio = build_clarification_portfolio(right_state, right_graph, right)
    left_decision = solve_active_clarification(left_graph, left_portfolio)
    right_decision = solve_active_clarification(right_graph, right_portfolio)

    left_selected = next(
        row for row in left_portfolio.questions
        if row.question_id == left_decision.selected_question_id
    )
    right_selected = next(
        row for row in right_portfolio.questions
        if row.question_id == right_decision.selected_question_id
    )
    # The persisted plan representation participates in state identity, but
    # the semantic decision must remain invariant to list order.
    assert left_state.state_id != right_state.state_id
    assert left_selected.requirement_id == right_selected.requirement_id == "policy:v1:ORDER_FACT"
    assert left_selected.question == right_selected.question
    assert left_decision.reason_code == right_decision.reason_code


def test_tampered_active_selection_fails_closed():
    result = _two_question_result()
    state = project_case_execution_state(result)
    graph = build_obligation_graph(state)
    portfolio = build_clarification_portfolio(state, graph, result)
    selection = solve_active_clarification(graph, portfolio).candidate_selection
    tampered = selection.model_copy(update={"selection_id": "a" * 64})

    directive = decide_next_obligation_action(
        state,
        graph,
        selection=tampered,
        planner_version=ACTIVE_CLARIFICATION_PLANNER_VERSION,
    )

    assert directive.status == "ABSTAINED"
    assert directive.reason_code == "OBLIGATION_SELECTION_INVALID"
