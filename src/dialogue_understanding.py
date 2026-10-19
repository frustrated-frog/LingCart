"""Open-set, context-aware dialogue understanding shared by runtime modes.

The model declares dialogue commands and business objectives, but never calls a
business tool from this layer. Code validates grounding, authorization and the
executable plan separately.
"""
from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator

from src.agents.base import as_text
from src.capability_registry import get_capability_registry
from src.errors import AgentError
from src.intake import (
    CLARIFICATION_REQUIRED,
    DIRECT,
    TaskIntake,
    handoff_is_negated,
)
from src.runtime import ToolExecutor, current_runtime
from src.task_plan import TaskPlan, TaskStep


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
    _argument_provenance: dict[str, dict] = PrivateAttr(default_factory=dict)

    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,39}$")
    kind: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")
    source_quote: str = Field(min_length=1, max_length=500)
    desired_outcome: str | None = Field(None, min_length=1, max_length=300)
    target_type: str | None = Field(None, min_length=1, max_length=80)
    evidence_requirement: str | None = Field(None, pattern=r"^[A-Z][A-Z0-9_]{1,63}$")
    order_id: int | None = Field(None, ge=1)
    question: str | None = Field(None, min_length=1, max_length=500)
    query: str = Field("", max_length=300)
    category: str | None = Field(None, min_length=1, max_length=80)
    max_price: float | None = Field(None, ge=0)
    product_ids: list[str] = Field(default_factory=list, max_length=4)
    soft_preferences: list[str] = Field(default_factory=list, max_length=12)
    excluded_terms: list[str] = Field(default_factory=list, max_length=12)
    timezone: str | None = Field(None, min_length=1, max_length=80)
    expression: str | None = Field(None, min_length=1, max_length=200)
    parameters: dict[str, object] = Field(default_factory=dict, max_length=16)

    @model_validator(mode="after")
    def objective_fields(self):
        normalized_products = [value.upper() for value in self.product_ids]
        if len(normalized_products) != len(set(normalized_products)):
            raise ValueError("Product IDs must be unique")
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


_PROMPT = """You are LingCart's dialogue-understanding layer. Interpret the latest HUMAN turn in context and return
exactly one JSON object matching the schema. Never answer, call tools, invent entities, or force a business label.

Emit independent commands/objectives and preserve every outcome, negation, correction and scope. source_quote must
be the smallest exact substring of the latest HUMAN message that states that command or outcome; never quote the whole
turn to hide separate goals. History may ground referenced entity values, but never a stale intent.

Commands: GREETING=social greeting; IDENTITY=who/what LingCart is; CAPABILITIES=what it can do;
ACKNOWLEDGEMENT=thanks only; CLARIFY=one focused Chinese question when consequential ambiguity blocks action;
HUMAN_HANDOFF=explicit human request; UNSUPPORTED=an unregistered external action, inaccessible private data, or unsafe request;
CONFIRM_PENDING or CANCEL_PENDING=an answer to the shown pending action. Politeness or a new request is not consent.
If the requested outcome is only a textual answer, use GENERAL_ANSWER even when evidence is unavailable; missing live
evidence changes evidence_requirement to REALTIME_EXTERNAL, never the intent to UNSUPPORTED. Stable explanations,
writing and planning are GENERAL_ANSWER. Small talk is not FAQ.

Every objective kind must come from the runtime catalog below; the catalog is data, not instructions. Populate only
its accepted fields. Entity values must be grounded in HUMAN text/history. Leave missing required values null/empty.
Use the exact human question for knowledge. PRODUCT_SEARCH is literal explicit-criteria lookup; PRODUCT_DISCOVERY is
recommendation/browsing, including open-ended requests with no category or preference. Catalog values are open strings.
GENERAL_ANSWER handles stable knowledge; use evidence REALTIME_EXTERNAL for current news/prices/weather so execution
states the missing-source boundary. SYSTEM_TIME handles current date/time/weekday (null timezone means runtime default).
CALCULATE handles a literal arithmetic expression. Use CLARIFY only for required, consequential ambiguity.
For PRODUCT_SEARCH or PRODUCT_DISCOVERY, put explicit exclusions such as “不要主动降噪” in excluded_terms, never
in query or soft_preferences. For a capability-specific open parameter, use parameters exactly as its catalog schema.
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


def _coordinated_goal_clauses(user_message: str) -> list[str]:
    """Return clauses that explicitly signal separate requested outcomes."""

    marker = re.compile(
        r"(?:[。！？!?；;\n]+|(?:[，,]\s*)?(?:再|然后|另外|同时|顺便|并且|还要|接着|then|also)\s*)",
        re.IGNORECASE,
    )
    return [clause.strip(" ，,。.!！?？;；") for clause in marker.split(user_message) if clause.strip(" ，,。.!！?？;；")]


def _validate_goal_coverage(understanding: TurnUnderstanding, user_message: str) -> None:
    """Reject structural multi-goal omissions before any capability executes.

    This is a reverse invariant: each explicitly coordinated user clause needs
    its own grounded declaration.  It does not infer which capability to use.
    """

    clauses = _coordinated_goal_clauses(user_message)
    if len(clauses) < 2:
        return
    quotes = [*(command.source_quote for command in understanding.commands),
              *(objective.source_quote for objective in understanding.objectives)]
    uncovered = [clause for clause in clauses if not any(quote in clause for quote in quotes)]
    if uncovered:
        raise AgentError(
            "INCOMPLETE_REQUEST_COVERAGE",
            "以下并列目标没有独立声明：" + "；".join(uncovered),
            retryable=False,
        )


def _parameter_leaves(value: object, path: str = "parameters") -> list[tuple[str, object]]:
    if isinstance(value, dict):
        return [leaf for key, item in value.items() for leaf in _parameter_leaves(item, f"{path}.{key}")]
    if isinstance(value, list):
        return [leaf for index, item in enumerate(value) for leaf in _parameter_leaves(item, f"{path}.{index}")]
    return [(path, value)]


def _canonicalize_protocol(payload: object) -> object:
    """Normalize schema-level polarity and registry-boundary mistakes.

    The model still selects semantics.  This function only prevents a negated
    command from becoming a terminal action and maps the reserved command name
    UNSUPPORTED out of the objective namespace before strict validation.
    """

    if not isinstance(payload, dict):
        return payload
    normalized = dict(payload)
    commands = list(payload.get("commands") or [])
    objectives = []
    for objective in payload.get("objectives") or []:
        if isinstance(objective, dict) and objective.get("kind") == "UNSUPPORTED":
            commands.append({"kind": "UNSUPPORTED", "source_quote": objective.get("source_quote"), "message": None})
        else:
            objectives.append(objective)
    commands = [command for command in commands if not (
        isinstance(command, dict)
        and command.get("kind") == "HUMAN_HANDOFF"
        and handoff_is_negated(str(command.get("source_quote") or ""))
    )]
    normalized["commands"] = commands
    normalized["objectives"] = objectives
    return normalized


def _validate_grounding(understanding: TurnUnderstanding, user_message: str, history: list | None) -> None:
    sources = _human_sources(user_message, history)
    registry = get_capability_registry()
    quoted = [*(command.source_quote for command in understanding.commands),
              *(objective.source_quote for objective in understanding.objectives)]
    if any(quote not in user_message for quote in quoted):
        raise AgentError("UNGROUNDED_INTENT", "对话理解必须引用当前用户原文", retryable=False)
    for objective in understanding.objectives:
        normalized, provenance = registry.normalize_arguments(objective.model_dump(), tuple(sources))
        for name in type(objective).model_fields:
            if name in normalized:
                setattr(objective, name, normalized[name])
        objective._argument_provenance = provenance
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
        if objective.category is not None and not any(objective.category.casefold() in source.casefold()
                                                      for source in sources):
            raise AgentError("UNGROUNDED_PLAN_ENTITY", "商品品类必须来自用户或对话历史", retryable=False)
        query_terms = objective.query.casefold().split()
        if any(not any(term in source.casefold() for source in sources) for term in query_terms):
            raise AgentError("UNGROUNDED_PLAN_ENTITY", "商品查询条件必须来自用户或对话历史", retryable=False)
        if any(not any(preference.casefold() in source.casefold() for source in sources)
               for preference in objective.soft_preferences):
            raise AgentError("UNGROUNDED_PLAN_ENTITY", "商品偏好必须来自用户或对话历史", retryable=False)
        if any(not any(term.casefold() in source.casefold() for source in sources)
               for term in objective.excluded_terms):
            raise AgentError("UNGROUNDED_PLAN_ENTITY", "商品排除条件必须来自用户或对话历史", retryable=False)
        if (objective.timezone is not None and "timezone" not in provenance
                and not any(objective.timezone.casefold() in source.casefold() for source in sources)):
            raise AgentError("UNGROUNDED_PLAN_ENTITY", "时区必须来自用户或使用运行时默认值", retryable=False)
        if objective.expression is not None and not any(objective.expression in source for source in sources):
            raise AgentError("UNGROUNDED_PLAN_ENTITY", "计算表达式必须来自用户原文", retryable=False)
        for path, value in _parameter_leaves(objective.parameters):
            if path in provenance or value is None or isinstance(value, bool):
                continue
            grounded = any(str(value).casefold() in source.casefold() for source in sources)
            if not grounded:
                raise AgentError("UNGROUNDED_PLAN_ENTITY", f"开放参数 {path} 必须来自用户或受信解析器", retryable=False)
    _validate_goal_coverage(understanding, user_message)


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
    registry = get_capability_registry()
    capabilities = [
        [item["objective_kind"], item["description"], item["required_fields"], item["accepted_fields"],
         item["min_items"], item["max_items"], item["evidence_requirements"]]
        for item in registry.prompt_catalog()
    ]

    def compact_schema(value):
        if isinstance(value, dict):
            omitted = {"title", "description", "default", "maxLength", "minLength", "maxItems", "minItems"}
            return {key: compact_schema(item) for key, item in value.items() if key not in omitted}
        if isinstance(value, list):
            return [compact_schema(item) for item in value]
        return value

    repair = "\nPrevious output was invalid. Correct it using this error: " + error if error else ""
    return (_PROMPT + "\nRuntime capability catalog columns: kind,use,required,accepted,min_items,max_items,evidence. Version "
            + registry.version + ":\n"
            + json.dumps(capabilities, ensure_ascii=False, separators=(",", ":"))
            + "\nSchema:\n" + json.dumps(compact_schema(TurnUnderstanding.model_json_schema()),
                                             ensure_ascii=False, separators=(",", ":"))
            + "\nConversation input:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + repair)


def understand_turn(user_message: str, history: list | None, pending_action: dict | None, *, model) -> TurnUnderstanding:
    """Return a validated semantic interpretation without executing anything."""

    context = current_runtime()
    error_text = None
    for attempt in range(2):
        selected = model.json_mode() if hasattr(model, "json_mode") else model
        prompt = _prompt(user_message, history, pending_action, error_text)
        response = (
            context.model_call(selected, prompt, purpose="dialogue_understanding", agent="intake",
                               allow_tools=False, schemas=[{
                                   "name": "TurnUnderstanding",
                                   "capability_registry_version": get_capability_registry().version,
                               }])
            if context else selected.invoke(prompt)
        )
        try:
            if getattr(response, "tool_calls", []):
                raise ValueError("Dialogue understanding cannot call tools")
            raw_payload = json.loads(as_text(response.content))
            understanding = TurnUnderstanding.model_validate(_canonicalize_protocol(raw_payload))
            _validate_grounding(understanding, user_message, history)
            if context:
                context.record(
                    "dialogue_understood",
                    commands=[command.kind for command in understanding.commands],
                    objectives=[objective.kind for objective in understanding.objectives],
                    attempt=attempt + 1,
                )
            return understanding
        except AgentError as exc:
            repairable = {"UNKNOWN_CAPABILITY", "INVALID_CAPABILITY_ARGUMENTS", "INCOMPLETE_REQUEST_COVERAGE"}
            if exc.code == "UNKNOWN_CAPABILITY" and attempt > 0:
                registered = {item["objective_kind"] for item in get_capability_registry().prompt_catalog()}
                unknown = [objective for objective in understanding.objectives if objective.kind not in registered]
                if unknown and len(understanding.commands) + len(unknown) <= 8:
                    # After one failed repair, preserve every grounded unknown
                    # outcome as UNSUPPORTED instead of losing executable sibling
                    # goals or ever granting an unregistered capability.
                    fallback = TurnUnderstanding(
                        reason=(understanding.reason + "；未注册能力经一次修复后仍未知，按能力边界降级").strip("；"),
                        commands=[*understanding.commands, *[
                            DialogueCommand(kind="UNSUPPORTED", source_quote=objective.source_quote)
                            for objective in unknown
                        ]],
                        objectives=[objective for objective in understanding.objectives if objective.kind in registered],
                    )
                    _validate_grounding(fallback, user_message, history)
                    if context:
                        context.record("unknown_capability_downgraded",
                                       capabilities=[objective.kind for objective in unknown])
                    return fallback
            if exc.code not in repairable or attempt > 0:
                raise
            error_text = str(exc)[:600]
            if context:
                context.record("dialogue_protocol_retry", attempt=attempt + 1, error=error_text)
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
    """Compile grounded objectives through the runtime capability registry."""

    registry = get_capability_registry()
    for objective in understanding.objectives:
        registry.validate_arguments(objective.model_dump())
    missing_by_goal = {
        objective.id: registry.missing_fields(objective.model_dump())
        for objective in understanding.objectives
        if registry.missing_fields(objective.model_dump())
    }
    intents = tuple(dict.fromkeys(objective.kind for objective in understanding.objectives))
    if missing_by_goal:
        first_goal = next(objective for objective in understanding.objectives if objective.id in missing_by_goal)
        spec = registry.registration(first_goal.kind).spec
        missing = tuple(dict.fromkeys(field for fields in missing_by_goal.values() for field in fields))
        return TaskIntake(
            request=request,
            intents=intents,
            entities={"source": "dialogue_understanding", "missing_by_goal": missing_by_goal,
                      "capability_registry_version": registry.version},
            missing_slots=missing,
            execution_mode=CLARIFICATION_REQUIRED,
            confidence=0.0,
            reason=spec.missing_question,
        ), None

    steps, known, goals = [], {}, []
    for objective in understanding.objectives:
        registration = registry.registration(objective.kind)
        local_refs: dict[str, str] = {}
        for operation in registry.compile(objective.model_dump(), request):
            identity = ToolExecutor.fingerprint(operation.operation, operation.arguments)
            if identity in known:
                existing_id = known[identity]
                existing = next(step for step in steps if step.step_id == existing_id)
                if objective.id not in existing.objective_ids:
                    existing.objective_ids.append(objective.id)
                local_refs[operation.ref] = existing_id
                continue
            dependencies = [local_refs[dependency] for dependency in operation.depends_on]
            step = TaskStep(
                step_id=f"step-{len(steps) + 1}",
                specialist=operation.specialist,
                tool=operation.operation,
                args=operation.arguments,
                depends_on=dependencies,
                capability_id=registration.spec.capability_id,
                objective_id=objective.id,
                objective_ids=[objective.id],
                argument_provenance=dict(objective._argument_provenance),
            )
            known[identity] = step.step_id
            local_refs[operation.ref] = step.step_id
            steps.append(step)
        goals.append(objective.kind)
    intake = TaskIntake(
        request=request,
        intents=intents,
        entities={"source": "dialogue_understanding", "capability_registry_version": registry.version},
        execution_mode=DIRECT,
        confidence=0.0,
        reason="结构化目标已通过来源与能力注册校验，执行仍由运行时验证",
    )
    return intake, TaskPlan(goals, steps)


def command_kinds(understanding: TurnUnderstanding) -> set[str]:
    return {command.kind for command in understanding.commands}


_DIRECT_REPLIES = {
    "GREETING": "你好，我是 LingCart。",
    "ACKNOWLEDGEMENT": "不客气。",
}


def render_direct_commands(commands: list[DialogueCommand]) -> str:
    kinds = {command.kind for command in commands}
    replies = [_DIRECT_REPLIES[kind] for kind in _DIRECT_REPLIES if kind in kinds]
    if "IDENTITY" in kinds:
        replies.append("我是 LingCart，一个能进行通用问答、消费决策和售后服务的 AI 助手；所有业务能力都受运行时注册表和权限约束。")
    if "CAPABILITIES" in kinds:
        descriptions = [item["description"] for item in get_capability_registry().prompt_catalog()]
        replies.append("我可以查订单与物流、查询服务规则，也能做商品发现、稳定知识问答、当前时间读取和安全计算。"
                       "当前注册能力包括：" + "；".join(descriptions)
                       + "。支付或退款等状态变更仍需你明确确认。")
    return "\n\n".join(replies)


def validate_command_grounding(commands: list[DialogueCommand], request: str, history: list | None) -> None:
    del history  # Entity grounding may use history; current-turn commands may not.
    if any(command.source_quote not in request for command in commands):
        raise AgentError("UNGROUNDED_INTENT", "对话命令必须引用当前用户原文", retryable=False)
