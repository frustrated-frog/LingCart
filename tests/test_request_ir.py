from langchain_core.messages import HumanMessage
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
