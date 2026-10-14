"""Structured fallback intake for ambiguous compound requests only."""
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.agents.base import as_text
from src.errors import AgentError
from src.intake import CLARIFICATION_REQUIRED, DIRECT, TaskIntake, analyze_request
from src.llm import get_llm
from src.runtime import current_runtime
from src.task_plan import TaskPlan, build_plan

Intent = Literal["ORDER_STATUS", "ORDER_DETAILS", "SHIPMENT_TRACKING", "REFUND_ELIGIBILITY", "REFUND_REQUEST",
                 "PAYMENT_STATUS", "PAYMENT_PROCESS", "FAQ", "PRODUCT_SEARCH", "PRODUCT_COMPARE"]
ORDER_INTENTS = {"ORDER_STATUS", "ORDER_DETAILS", "SHIPMENT_TRACKING", "REFUND_ELIGIBILITY", "REFUND_REQUEST", "PAYMENT_STATUS", "PAYMENT_PROCESS"}


class GoalDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    intent: Intent
    order_id: int | None = Field(None, gt=0)
    query: str = ""
    category: Literal["耳机", "键盘", "扩展坞"] | None = None
    max_price: float | None = Field(None, ge=0)
    product_ids: list[str] = Field(default_factory=list, max_length=4)


class PlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    goals: list[GoalDraft] = Field(..., min_length=1, max_length=8)


def needs_model_plan(request):
    return bool(re.search(r"然后|同时|另外|并且|分别|一并|以及|and then|also|as well as", request, re.IGNORECASE))


def create_model_plan(request, history):
    context = current_runtime()
    prompt = (
        "Interpret this compound consumer-support request as JSON with exactly one goals array. "
        "Each goal uses the provided schema. Preserve all requested goals. Never invent order IDs or product IDs. "
        "Use null order_id for a missing order. Payment/refund requests are preparation for approval, never authorization. "
        "Return only valid JSON, no prose or tool calls.\nSchema: " + json.dumps(PlanDraft.model_json_schema(), ensure_ascii=False)
        + "\nRecent conversation: " + json.dumps([{"role": getattr(m, "type", "unknown"), "content": as_text(m.content)[:1000]} for m in (history or [])[-6:]], ensure_ascii=False)
        + "\nUser request: " + request
    )
    response = context.model_call(get_llm(), prompt, purpose="task_intake", allow_tools=False,
                                  schemas=[PlanDraft.model_json_schema()])
    try:
        draft = PlanDraft.model_validate(json.loads(as_text(response.content)), strict=True)
        if getattr(response, "tool_calls", []):
            raise ValueError("规划阶段不允许调用工具")
    except (ValueError, TypeError, ValidationError) as exc:
        raise AgentError("MODEL_FORMAT_ERROR", "任务识别模型没有返回有效计划", retryable=False) from exc
    allowed_orders = set(analyze_request(request).entities.get("order_ids", []))
    current_order = analyze_request(request).entities.get("order_id")
    if current_order:
        allowed_orders.add(current_order)
    if not allowed_orders:
        for message in reversed(history or []):
            previous = analyze_request(as_text(message.content))
            ids = previous.entities.get("order_ids") or ([previous.entities["order_id"]] if previous.entities.get("order_id") else [])
            if ids:
                if len(ids) == 1:
                    allowed_orders.update(ids)
                break
    allowed_products = {value.upper() for value in re.findall(r"\bP\d{4}S?\b", request, re.IGNORECASE)}
    if any(goal.intent in ORDER_INTENTS and goal.order_id is None for goal in draft.goals):
        intake = TaskIntake(request, tuple(goal.intent for goal in draft.goals), missing_slots=("order_id",),
                            execution_mode=CLARIFICATION_REQUIRED, reason="复合任务缺少订单号")
        return intake, None
    steps, seen, goals = [], {}, []
    for goal in draft.goals:
        if goal.intent in ORDER_INTENTS and goal.order_id not in allowed_orders:
            raise AgentError("UNGROUNDED_PLAN_ENTITY", "模型计划引用了用户与会话均未提供的订单号", retryable=False)
        if goal.intent == "PRODUCT_COMPARE" and (len(set(goal.product_ids)) < 2 or not {p.upper() for p in goal.product_ids} <= allowed_products):
            raise AgentError("UNGROUNDED_PLAN_ENTITY", "商品对比缺少用户明确提供的不同商品编号", retryable=False)
        entities = goal.model_dump(exclude={"intent"}, exclude_none=True)
        entities["product_ids"] = [p.upper() for p in goal.product_ids]
        part = build_plan(TaskIntake(request, (goal.intent,), entities, execution_mode=DIRECT))
        mapping = {}
        for step in part.steps:
            from src.runtime import ToolExecutor
            identity = ToolExecutor.fingerprint(step.tool, step.args)
            old_id = step.step_id
            if identity in seen:
                mapping[old_id] = seen[identity]
                continue
            step.step_id = f"step-{len(steps) + 1}"
            step.depends_on = [mapping[d] for d in step.depends_on]
            seen[identity] = step.step_id
            mapping[old_id] = step.step_id
            steps.append(step)
        goals.append(goal.intent)
    intake = TaskIntake(request, tuple(dict.fromkeys(goals)), {"source": "structured_model"}, execution_mode=DIRECT,
                        confidence=0, reason="模型生成结构化目标，工具与审批仍由运行时校验")
    return intake, TaskPlan(goals, steps)
