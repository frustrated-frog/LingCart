"""Open-set, context-aware dialogue understanding shared by runtime modes.

The model declares dialogue commands and business objectives, but never calls a
business tool from this layer. Code validates grounding, authorization and the
executable plan separately.
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.agents.base import as_text
from src.errors import AgentError
from src.intake import (
    CLARIFICATION_REQUIRED,
    DIRECT,
    FAQ,
    ORDER_DETAILS,
    ORDER_STATUS,
    PAYMENT_PROCESS,
    PAYMENT_STATUS,
    PRODUCT_COMPARE,
    PRODUCT_SEARCH,
    REFUND_ELIGIBILITY,
    REFUND_REQUEST,
    SHIPMENT_TRACKING,
    TaskIntake,
)
from src.runtime import ToolExecutor, current_runtime
from src.task_plan import TaskPlan, build_plan


BusinessKind = Literal[
    "ORDER_STATUS",
    "ORDER_DETAILS",
    "SHIPMENT_TRACKING",
    "REFUND_ELIGIBILITY",
    "REFUND_REQUEST",
    "PAYMENT_STATUS",
    "PAYMENT_PROCESS",
    "FAQ",
    "PRODUCT_SEARCH",
    "PRODUCT_COMPARE",
]
CommandKind = Literal[
    "GREETING",
    "IDENTITY",
    "CAPABILITIES",
    "ACKNOWLEDGEMENT",
    "CLARIFY",
    "HUMAN_HANDOFF",
    "UNSUPPORTED",
    "CONFIRM_PENDING",
    "CANCEL_PENDING",
]

_ORDER_KINDS = {
    ORDER_STATUS,
    ORDER_DETAILS,
    SHIPMENT_TRACKING,
    REFUND_ELIGIBILITY,
    REFUND_REQUEST,
    PAYMENT_STATUS,
    PAYMENT_PROCESS,
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DialogueCommand(StrictModel):
    kind: CommandKind
    source_quote: str = Field(min_length=1, max_length=500)
    message: str | None = Field(None, min_length=1, max_length=800)

    @model_validator(mode="after")
    def command_fields(self):
        if self.kind == "CLARIFY" and self.message is None:
            raise ValueError("CLARIFY requires a focused question")
        if self.kind != "CLARIFY" and self.message is not None:
            raise ValueError("Only CLARIFY may provide a message")
        return self


class BusinessObjective(StrictModel):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,39}$")
    kind: BusinessKind
    source_quote: str = Field(min_length=1, max_length=500)
    order_id: int | None = Field(None, ge=1)
    question: str | None = Field(None, min_length=1, max_length=500)
    query: str = Field("", max_length=300)
    category: Literal["耳机", "键盘", "扩展坞"] | None = None
    max_price: float | None = Field(None, ge=0)
    product_ids: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def objective_fields(self):
        if self.kind not in _ORDER_KINDS and self.order_id is not None:
            raise ValueError("Only order objectives may contain order_id")
        if self.kind == FAQ:
            if self.question is None:
                raise ValueError("FAQ requires the user's exact question")
        elif self.question is not None:
            raise ValueError("Only FAQ may contain question")
        normalized_products = [value.upper() for value in self.product_ids]
        if len(normalized_products) != len(set(normalized_products)):
            raise ValueError("Product IDs must be unique")
        if self.kind == PRODUCT_COMPARE:
            if not 2 <= len(normalized_products) <= 3:
                raise ValueError("Product comparison requires two or three IDs")
        elif self.product_ids:
            raise ValueError("Only product comparison may contain product_ids")
        self.product_ids = normalized_products
        return self


class TurnUnderstanding(StrictModel):
    reason: str = Field("", max_length=800)
    commands: list[DialogueCommand] = Field(default_factory=list, max_length=8)
    objectives: list[BusinessObjective] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def nonempty_and_consistent(self):
        if not self.commands and not self.objectives:
            raise ValueError("Declare at least one command or objective")
        if len({objective.id for objective in self.objectives}) != len(self.objectives):
            raise ValueError("Objective IDs must be unique")
        command_kinds = {command.kind for command in self.commands}
        terminal = command_kinds & {"CLARIFY", "HUMAN_HANDOFF", "CONFIRM_PENDING", "CANCEL_PENDING"}
        if len(terminal) > 1:
            raise ValueError("Conflicting terminal commands")
        if terminal and self.objectives:
            raise ValueError("Terminal commands cannot execute business objectives in the same turn")
        return self


_PROMPT = """You are the dialogue-understanding layer for LingCart, a consumer shopping and after-sales assistant.
Interpret the latest HUMAN message in context. Return exactly one JSON object matching the supplied schema.
Do not answer the user, call tools, invent identifiers, or force the message into a business category.

Declare zero or more dialogue commands and zero or more independent business objectives. One user turn may
contain several objectives or a social command plus business objectives. source_quote must be an exact substring
of the latest HUMAN message; history may supply referenced entity values, never a stale current intent. Preserve
negation, corrections, scope and every requested outcome.

Commands:
- GREETING: social greeting only.
- IDENTITY: asks who/what LingCart is.
- CAPABILITIES: asks what LingCart can do or how to use it.
- ACKNOWLEDGEMENT: thanks or acknowledgement with no new request.
- CLARIFY: material ambiguity prevents safe interpretation; message is one focused Chinese question.
- HUMAN_HANDOFF: explicitly asks for a human.
- UNSUPPORTED: a clear request outside shopping, orders, shipment, refunds, payments and service policies.
- CONFIRM_PENDING/CANCEL_PENDING: answer to the pending action shown in context. Never infer confirmation from
  politeness, a new request, a policy question, or ambiguous wording.

Business objective kinds:
ORDER_STATUS, ORDER_DETAILS, SHIPMENT_TRACKING, REFUND_ELIGIBILITY, REFUND_REQUEST,
PAYMENT_STATUS, PAYMENT_PROCESS, FAQ, PRODUCT_SEARCH, PRODUCT_COMPARE.
Order objectives use order_id only when grounded in HUMAN text/history; leave it null when missing.
FAQ question must be the exact human question. PRODUCT_COMPARE needs 2-3 explicit product IDs.
For PRODUCT_SEARCH extract supported category, maximum budget and compact literal feature query when present.
Do not turn identity/capability/small-talk into FAQ. Use UNSUPPORTED rather than the nearest business label.
Use CLARIFY only for consequential ambiguity, not for optional unspecified preferences.
"""


def _human_sources(user_message: str, history: list | None) -> list[str]:
    return [
        user_message,
        *[
            as_text(message.content)
            for message in history or []
            if getattr(message, "type", None) == "human"
        ],
    ]


def _validate_grounding(understanding: TurnUnderstanding, user_message: str, history: list | None) -> None:
    sources = _human_sources(user_message, history)
    quoted = [*(command.source_quote for command in understanding.commands),
              *(objective.source_quote for objective in understanding.objectives)]
    if any(quote not in user_message for quote in quoted):
        raise AgentError("UNGROUNDED_INTENT", "对话理解必须引用当前用户原文", retryable=False)
    for objective in understanding.objectives:
        if objective.order_id is not None and not any(
            re.search(r"(?<!\d)" + str(objective.order_id) + r"(?!\d)", source) for source in sources
        ):
            raise AgentError("UNGROUNDED_PLAN_ENTITY", "目标订单编号必须来自用户或对话历史", retryable=False)
        if objective.question is not None and not any(objective.question in source for source in sources):
            raise AgentError("UNGROUNDED_INTENT", "政策问题必须引用用户原文", retryable=False)
        for product_id in objective.product_ids:
            if not any(re.search(r"(?<![A-Za-z0-9])" + re.escape(product_id) + r"(?![A-Za-z0-9])",
                                 source, re.IGNORECASE) for source in sources):
                raise AgentError("UNGROUNDED_PLAN_ENTITY", "商品编号必须来自用户或对话历史", retryable=False)


def _prompt(user_message: str, history: list | None, pending_action: dict | None, error: str | None = None) -> str:
    recent = [
        {"role": getattr(message, "type", "unknown"), "content": as_text(message.content)[:1200]}
        for message in (history or [])[-8:]
    ]
    pending = None
    if pending_action:
        pending = {key: pending_action.get(key) for key in ("type", "order_id", "amount", "method")
                   if pending_action.get(key) is not None}
    payload = {
        "recent_conversation": recent,
        "pending_action": pending,
        "latest_human_message": user_message,
    }
    repair = "\nPrevious output was invalid. Correct it using this error: " + error if error else ""
    return (_PROMPT + "\nSchema:\n" + json.dumps(TurnUnderstanding.model_json_schema(), ensure_ascii=False)
            + "\nConversation input:\n" + json.dumps(payload, ensure_ascii=False) + repair)


def understand_turn(user_message: str, history: list | None, pending_action: dict | None, *, model) -> TurnUnderstanding:
    """Return a validated semantic interpretation without executing anything."""

    context = current_runtime()
    error_text = None
    for attempt in range(2):
        selected = model.json_mode() if hasattr(model, "json_mode") else model
        prompt = _prompt(user_message, history, pending_action, error_text)
        response = (
            context.model_call(selected, prompt, purpose="dialogue_understanding", agent="intake",
                               allow_tools=False, schemas=[TurnUnderstanding.model_json_schema()])
            if context else selected.invoke(prompt)
        )
        try:
            if getattr(response, "tool_calls", []):
                raise ValueError("Dialogue understanding cannot call tools")
            understanding = TurnUnderstanding.model_validate(json.loads(as_text(response.content)))
            _validate_grounding(understanding, user_message, history)
            if context:
                context.record(
                    "dialogue_understood",
                    commands=[command.kind for command in understanding.commands],
                    objectives=[objective.kind for objective in understanding.objectives],
                    attempt=attempt + 1,
                )
            return understanding
        except AgentError:
            raise
        except (ValueError, TypeError, ValidationError) as exc:
            details = (
                [{"path": ".".join(map(str, issue["loc"])), "message": issue["msg"]}
                 for issue in exc.errors(include_input=False, include_url=False)[:8]]
                if isinstance(exc, ValidationError) else str(exc)[:600]
            )
            error_text = json.dumps(details, ensure_ascii=False)
            if context:
                context.record("dialogue_protocol_retry", attempt=attempt + 1, error=error_text)
    raise AgentError("MODEL_FORMAT_ERROR", "对话理解模型连续返回无效协议：" + (error_text or "未知错误"), retryable=False)


def to_task_plan(understanding: TurnUnderstanding, request: str) -> tuple[TaskIntake, TaskPlan | None]:
    """Compile grounded business objectives into the existing safe task plan."""

    missing_order = any(objective.kind in _ORDER_KINDS and objective.order_id is None
                        for objective in understanding.objectives)
    intents = tuple(dict.fromkeys(objective.kind for objective in understanding.objectives))
    if missing_order:
        return TaskIntake(
            request=request,
            intents=intents,
            entities={"source": "dialogue_understanding"},
            missing_slots=("order_id",),
            execution_mode=CLARIFICATION_REQUIRED,
            confidence=0.0,
            reason="语义理解识别到订单任务，但缺少订单号",
        ), None

    steps, known, goals = [], {}, []
    for objective in understanding.objectives:
        entities = {
            key: value
            for key, value in objective.model_dump(exclude={"id", "kind", "source_quote"}).items()
            if value not in (None, "", [])
        }
        if objective.kind == FAQ:
            part_request = objective.question or request
        else:
            part_request = objective.source_quote
        part = build_plan(TaskIntake(part_request, (objective.kind,), entities, execution_mode=DIRECT))
        remap = {}
        for step in part.steps:
            old_id = step.step_id
            identity = ToolExecutor.fingerprint(step.tool, step.args)
            if identity in known:
                remap[old_id] = known[identity]
                continue
            step.step_id = f"step-{len(steps) + 1}"
            step.depends_on = [remap[dependency] for dependency in step.depends_on]
            known[identity] = step.step_id
            remap[old_id] = step.step_id
            steps.append(step)
        goals.append(objective.kind)
    intake = TaskIntake(
        request=request,
        intents=intents,
        entities={"source": "dialogue_understanding"},
        execution_mode=DIRECT,
        confidence=0.0,
        reason="结构化对话理解已通过来源校验，执行仍由运行时验证",
    )
    return intake, TaskPlan(goals, steps)


def command_kinds(understanding: TurnUnderstanding) -> set[str]:
    return {command.kind for command in understanding.commands}


_DIRECT_REPLIES = {
    "GREETING": "你好，我是 LingCart。",
    "IDENTITY": "我是 LingCart，一个面向消费决策和售后服务的 AI 助手。",
    "CAPABILITIES": "我可以帮你查订单与物流、查看支付状态、判断退款条件、查询服务规则，也可以搜索和比较本地演示商品。涉及支付或退款操作时，我会先请你明确确认。",
    "ACKNOWLEDGEMENT": "不客气。",
}


def render_direct_commands(commands: list[DialogueCommand]) -> str:
    kinds = {command.kind for command in commands}
    return "\n\n".join(_DIRECT_REPLIES[kind] for kind in _DIRECT_REPLIES if kind in kinds)


def validate_command_grounding(commands: list[DialogueCommand], request: str, history: list | None) -> None:
    del history  # Entity grounding may use history; current-turn commands may not.
    if any(command.source_quote not in request for command in commands):
        raise AgentError("UNGROUNDED_INTENT", "对话命令必须引用当前用户原文", retryable=False)
