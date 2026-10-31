from langchain_core.messages import HumanMessage
from copy import deepcopy
import pytest

from src.dialogue_understanding import TurnUnderstanding, to_task_plan
from src.request_audit import RequestAudit
from src.request_ir import (
    locate_source_span,
    project_adaptive_request_ir,
    project_dialogue_request_ir,
)


def test_source_span_is_host_located_and_prefers_current_turn():
    history = [HumanMessage(content="之前说预算500元")]
    current = "还是按预算500元推荐"

    span = locate_source_span("预算500元", current, history)

    assert span.source == "CURRENT_TURN"
    assert span.history_index is None
    assert current[span.start:span.end] == span.text == "预算500元"
    assert len(span.source_sha256) == 64


def test_adaptive_ir_keeps_prohibition_out_of_executable_references():
    request = "推荐两款500元以内的耳机，每款单独核验，不要执行支付"
    goal = {
        "summary": "耳机推荐",
        "objectives": [{
            "id": "headphones",
            "description": "推荐两款耳机",
            "source_quote": "推荐两款500元以内的耳机",
            "kind": "recommend",
            "quantity": 2,
            "hard_constraints": [],
        }],
        "hard_constraints": [{
            "description": "价格不超过500元",
            "source_quote": "500元以内",
            "field": "price",
            "op": "lte",
            "value": 500,
        }],
        "soft_preferences": [],
        "uncertainties": [],
    }
    audit = RequestAudit.model_validate({
        "verdict": "APPROVED",
        "issues": [],
        "requirements": [
            {
                "scope": "GOAL_REQUIREMENT",
                "source_quote": "推荐两款500元以内的耳机",
                "interpretation": "推荐两款耳机",
                "objective_ids": ["headphones"],
                "constraint_refs": [],
                "status": "COVERED",
            },
            {
                "scope": "GOAL_REQUIREMENT",
                "source_quote": "500元以内",
                "interpretation": "价格不超过500元",
                "objective_ids": [],
                "constraint_refs": ["global:0"],
                "status": "COVERED",
            },
            {
                "scope": "EXECUTION_BOUNDARY",
                "source_quote": "每款单独核验",
                "interpretation": "分别核验候选",
                "objective_ids": [],
                "constraint_refs": [],
                "status": "COVERED",
            },
            {
                "scope": "PROHIBITION",
                "source_quote": "不要执行支付",
                "interpretation": "禁止执行支付",
                "objective_ids": [],
                "constraint_refs": [],
                "status": "COVERED",
            },
        ],
    })

    request_ir = project_adaptive_request_ir(request, None, goal, audit)
    rows = {row.kind: row for row in request_ir.requirements}

    assert rows["BUSINESS_OUTCOME"].objective_ids == ["headphones"]
    assert rows["PRODUCT_PREDICATE"].constraint_refs == ["global:0"]
    assert rows["EXECUTION_BOUNDARY"].objective_ids == []
    assert rows["PROHIBITION"].constraint_refs == []
    assert rows["PROHIBITION"].source.text == "不要执行支付"


def test_dialogue_ir_classifies_preferences_exclusions_and_user_facts():
    request = "推荐耳机，偏好轻便，排除入耳式；订单1001原包装稍后确认"
    understanding = TurnUnderstanding.model_validate({
        "reason": "同时保留选购偏好和售后事实",
        "commands": [],
        "objectives": [
            {
                "id": "shopping",
                "kind": "PRODUCT_DISCOVERY",
                "source_quote": "推荐耳机",
                "query": "耳机",
                "soft_preferences": ["轻便"],
                "excluded_terms": ["入耳式"],
            },
            {
                "id": "case",
                "kind": "AFTER_SALES_CASE",
                "source_quote": "订单1001原包装稍后确认",
                "order_id": 1001,
                "reason_code": "CHANGED_MIND",
                "evidence_context_quotes": ["原包装稍后确认"],
            },
        ],
    })

    request_ir = project_dialogue_request_ir(request, None, understanding)
    kinds = [row.kind for row in request_ir.requirements]

    assert kinds.count("BUSINESS_OUTCOME") == 2
    assert "SOFT_PREFERENCE" in kinds
    assert "PRODUCT_PREDICATE" in kinds
    assert "USER_FACT" in kinds
    exclusion = next(row for row in request_ir.requirements if row.kind == "PRODUCT_PREDICATE")
    assert exclusion.source.text == "入耳式"

    intake, _plan = to_task_plan(understanding, request)
    assert intake.entities["request_ir"]["sha256"] == request_ir.sha256


def test_non_executable_audit_scope_cannot_carry_goal_references():
    with pytest.raises(ValueError):
        RequestAudit.model_validate({
            "verdict": "APPROVED",
            "issues": [],
            "requirements": [{
                "scope": "PROHIBITION",
                "source_quote": "不要支付",
                "interpretation": "禁止支付",
                "objective_ids": ["payment"],
                "constraint_refs": [],
                "status": "COVERED",
            }],
        })


def compiled_request():
    request = "推荐两款耳机，预算500；查看订单1001状态；不要支付"
    goal = dict(summary="消费任务", objectives=[
        dict(id="shopping", kind="recommend", description="两款耳机", source_quote="推荐两款耳机", quantity=2),
        dict(id="order", kind="get_order_status", description="订单状态", source_quote="查看订单1001状态", order_id=1001),
    ], hard_constraints=[dict(description="预算上限", source_quote="预算500", field="price", op="lte", value=500)])
    audit = dict(verdict="APPROVED", requirements=[
        dict(scope="GOAL_REQUIREMENT", source_quote="推荐两款耳机", interpretation="推荐两款", objective_ids=["shopping"], constraint_refs=[], status="COVERED"),
        dict(scope="GOAL_REQUIREMENT", source_quote="查看订单1001状态", interpretation="查订单", objective_ids=["order"], constraint_refs=[], status="COVERED"),
        dict(scope="GOAL_REQUIREMENT", source_quote="预算500", interpretation="预算", objective_ids=[], constraint_refs=["global:0"], status="COVERED"),
        dict(scope="PROHIBITION", source_quote="不要支付", interpretation="禁止支付", objective_ids=[], constraint_refs=[], status="COVERED"),
    ])
    return request, goal, audit


@pytest.mark.parametrize("mutation", ["quantity", "scope", "missing", "extra", "source", "history", "target"])
def test_compiler_rejects_semantically_rehashed_inconsistent_contract(mutation):
    from src.request_ir import UserRequestIR
    from src.request_compiler import validate_adaptive_compilation
    from src.errors import AgentError

    request, goal, audit = compiled_request()
    ir = project_adaptive_request_ir(request, None, goal, audit).model_dump()
    if mutation == "quantity":
        ir["requirements"][0]["quantity"] = 1
    elif mutation == "scope":
        ir["requirements"][2]["objective_ids"] = ["order"]
    elif mutation == "missing":
        ir["requirements"].pop(0)
    elif mutation == "extra":
        extra = deepcopy(ir["requirements"][0])
        extra["requirement_id"] = "outcome:extra"
        ir["requirements"].append(extra)
    elif mutation == "source":
        ir["requirements"][0]["source"] = locate_source_span("查看订单1001状态", request).model_dump()
    elif mutation == "history":
        ir["requirements"][0]["source"]["source"] = "HISTORY"
        ir["requirements"][0]["source"]["history_index"] = 0
    else:
        goal["objectives"][1]["order_id"] = 1004
    payload = UserRequestIR.model_validate(ir).payload()
    with pytest.raises(AgentError, match="请求编译一致性"):
        validate_adaptive_compilation(payload, goal, request, None)


def test_compiler_does_not_allow_review_of_unrelated_text_to_cover_goal():
    from src.errors import AgentError
    request, goal, audit = compiled_request()
    audit["requirements"][0]["source_quote"] = "查看订单1001状态"
    with pytest.raises(AgentError, match="审查原文"):
        project_adaptive_request_ir(request, None, goal, audit)


def test_compiler_rejects_prohibition_compiled_as_product_predicate():
    from src.errors import AgentError
    request, goal, audit = compiled_request()
    goal["hard_constraints"][0]["source_quote"] = "不要支付"
    audit["requirements"][2]["source_quote"] = "不要支付"
    with pytest.raises(AgentError, match="非执行要求"):
        project_adaptive_request_ir(request, None, goal, audit)


def test_compiler_preserves_quantity_and_recommendation_only_scope():
    from src.request_compiler import validate_adaptive_compilation
    request, goal, audit = compiled_request()
    ir = project_adaptive_request_ir(request, None, goal, audit)
    validate_adaptive_compilation(ir.payload(), goal, request, None)
    assert ir.requirements[0].quantity == 2
    assert ir.requirements[2].objective_ids == ["shopping"]
    assert ir.adaptive_goal_sha256
