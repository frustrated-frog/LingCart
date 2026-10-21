import logging
import re
import time
import uuid
from datetime import UTC, datetime

from src.agents.base import as_text, sum_usage
from src.direct_executor import execute_approved_action, execute_direct
from src.errors import AgentError, classify_exception
from src.intake import (
    CLARIFICATION_REQUIRED,
    DIRECT,
    ESCALATION_REQUESTED,
    LLM_REQUIRED,
    UNSUPPORTED,
    APPROVAL_CANCELLED,
    APPROVAL_CONFIRMED,
    TaskIntake,
    analyze_request,
    parse_approval_response,
)
from src.llm import get_llm
from src.logging_config import configure_logging
from src.runtime import BudgetLimits, RunRuntime, ToolExecutor, TraceRecorder, current_runtime, runtime_scope
from src.tools import escalate_to_human

configure_logging()
logger = logging.getLogger("support_agent.orchestrator")

def _event(run_id: str, sequence: int, event_type: str, **data) -> dict:
    context = current_runtime()
    if context and context.run_id == run_id:
        return context.record(event_type, **data)
    return {
        "run_id": run_id,
        "sequence": sequence,
        "timestamp": datetime.now(UTC).isoformat(),
        "type": event_type,
        **data,
    }


def _escalate(run_id, request, reason, terminal_reason, usage=None):
    recorder = TraceRecorder(run_id)
    executor = ToolExecutor({"escalate_to_human": escalate_to_human}, recorder, max_calls=1)
    args = {"reason": reason, "context": request}
    result, event = executor.execute("escalate_to_human", args)
    recorder.record("terminal", status="WAITING_HUMAN", reason=terminal_reason)
    return dict(run_id=run_id, reply=f"人工请求 #{result['ticket_id']} 已保存到本地演示队列。尚未接入真人客服或通知服务，请勿视为已经转交。",
                trace=[dict(run_id=run_id, agent="escalate", tool="escalate_to_human", args=args,
                            result=result, status=event["status"], duration_ms=event["duration_ms"])],
                events=recorder.events, agent="escalate", route="ESCALATE", usage=usage or {},
                status="WAITING_HUMAN", terminal_reason=terminal_reason, pending_action=None, error=None)


def _simple_result(
    intake: TaskIntake,
    run_id: str,
    *,
    reply: str,
    agent: str,
    route: str,
    status: str,
    terminal_reason: str,
    trace: list | None = None,
    error: dict | None = None,
    pending_action: dict | None = None,
) -> dict:
    recorder = TraceRecorder(run_id)
    recorder.record(
        "route_decision",
        source="deterministic",
        route=route,
        intents=list(intake.intents),
        confidence=intake.confidence,
    )
    recorder.record("terminal", status=status, reason=terminal_reason, error=error)
    return {
        "run_id": run_id,
        "reply": reply,
        "trace": trace or [],
        "events": recorder.events,
        "agent": agent,
        "route": route,
        "status": status,
        "terminal_reason": terminal_reason,
        "error": error,
        "pending_action": pending_action,
        "intake": intake.to_dict(),
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "cached": False,
    }


def _explicit_escalation(intake: TaskIntake, run_id: str, user_message: str) -> dict:
    TraceRecorder(run_id).record("route_decision", source="deterministic", route="ESCALATE", confidence=1.0)
    result = _escalate(run_id, user_message, "用户明确要求人工客服", "USER_REQUESTED_ESCALATION")
    result["intake"] = intake.to_dict()
    return result


def _execute_understanding(user_message, history, pending_action, intake, run_id, *, understanding=None):
    """Interpret an open-set turn, then hand only validated goals to executors."""

    from src.dialogue_understanding import command_kinds, render_direct_commands, to_task_plan, understand_turn

    if understanding is None:
        understanding_model = get_llm(purpose="dialogue_understanding")
        understanding = understand_turn(
            user_message,
            history,
            pending_action,
            model=understanding_model,
        )
        if any(objective.catalog_query is not None for objective in understanding.objectives):
            from src.catalog_request_audit import review_catalog_request

            audit = review_catalog_request(
                user_message,
                history,
                understanding.objectives,
                model=get_llm(purpose="catalog_request_audit"),
            )
            if audit.get("verdict") == "REVISE":
                understanding = understand_turn(
                    user_message,
                    history,
                    pending_action,
                    model=understanding_model,
                    review_feedback={"issues": audit.get("issues"), "requirements": audit.get("requirements")},
                )
                audit = review_catalog_request(
                    user_message,
                    history,
                    understanding.objectives,
                    model=get_llm(purpose="catalog_request_audit"),
                )
            if audit.get("verdict") == "CLARIFY":
                result = _simple_result(
                    intake,
                    run_id,
                    reply="；".join(audit.get("issues") or ["这个目录请求仍有需要确认的含义。"]),
                    agent="intake",
                    route="CLARIFY",
                    status="WAITING_CLARIFICATION",
                    terminal_reason="CATALOG_REQUEST_AMBIGUOUS",
                )
                result["understanding"] = understanding.model_dump()
                result["catalog_request_audit"] = audit
                return result
            if audit.get("verdict") != "APPROVED":
                raise AgentError(
                    "INCOMPLETE_REQUEST_COVERAGE",
                    "目录请求经过一次修正后仍未通过原始需求覆盖审查",
                    retryable=False,
                )
        else:
            audit = None
    else:
        audit = None
    kinds = command_kinds(understanding)
    payload = understanding.model_dump()

    # The semantic layer may recognize an implicit answer, but mutating actions
    # retain the stricter explicit authorization parser above this function.
    if kinds & {"CONFIRM_PENDING", "CANCEL_PENDING"} and pending_action:
        result = _simple_result(
            intake,
            run_id,
            reply="为避免误操作，请明确回复“确认”继续，或回复“取消”放弃。",
            agent="intake",
            route="APPROVAL",
            status="WAITING_APPROVAL",
            terminal_reason="APPROVAL_REQUIRED",
            pending_action=pending_action,
        )
    elif kinds & {"CONFIRM_PENDING", "CANCEL_PENDING"}:
        result = _simple_result(
            intake,
            run_id,
            reply="当前没有等待确认的支付或退款操作，请重新说明你要处理的事项。",
            agent="intake",
            route="CLARIFY",
            status="WAITING_CLARIFICATION",
            terminal_reason="NO_PENDING_ACTION",
        )
    elif "HUMAN_HANDOFF" in kinds:
        result = _explicit_escalation(intake, run_id, user_message)
    elif "CLARIFY" in kinds:
        question = next(command.message for command in understanding.commands if command.kind == "CLARIFY")
        result = _simple_result(
            intake,
            run_id,
            reply=question,
            agent="intake",
            route="CLARIFY",
            status="WAITING_CLARIFICATION",
            terminal_reason="MODEL_CLARIFICATION",
        )
    elif understanding.objectives:
        planned_intake, plan = to_task_plan(understanding, user_message)
        if planned_intake.execution_mode == CLARIFICATION_REQUIRED:
            missing_order = planned_intake.missing_slots == ("order_id",)
            result = _simple_result(
                planned_intake,
                run_id,
                reply=planned_intake.reason,
                agent="intake",
                route="CLARIFY",
                status="WAITING_CLARIFICATION",
                terminal_reason="MISSING_ORDER_ID" if missing_order else "MISSING_REQUIRED_INFORMATION",
            )
        else:
            from src.task_plan import execute_plan

            result = execute_plan(
                planned_intake,
                run_id,
                plan=plan,
                source="dialogue_model",
                incomplete_reason="PARTIAL_UNSUPPORTED_INTENT" if "UNSUPPORTED" in kinds else None,
            )
            prefix = render_direct_commands(understanding.commands)
            if prefix:
                result["reply"] = prefix + "\n\n" + result["reply"]
            if "UNSUPPORTED" in kinds:
                result["reply"] += "\n\n另外一部分请求超出了当前消费服务助手的能力范围。"
    elif "UNSUPPORTED" in kinds:
        result = _simple_result(
            intake,
            run_id,
            reply=("我理解这个请求，但它需要当前未注册的外部操作、不可访问的私有数据，或属于不能安全执行的事项。"
                   "如果你只需要解释、写作或规划，请直接说明目标；如果依赖实时信息，可以提供可验证来源。"),
            agent="intake",
            route="UNSUPPORTED",
            status="FAILED",
            terminal_reason="UNSUPPORTED_INTENT",
            error={"code": "UNSUPPORTED_INTENT", "message": "结构化对话理解判定请求超出能力范围", "retryable": False},
        )
    else:
        result = _simple_result(
            intake,
            run_id,
            reply=render_direct_commands(understanding.commands),
            agent="intake",
            route="CONVERSATIONAL",
            status="SUCCEEDED",
            terminal_reason="COMPLETED",
        )
    result["understanding"] = payload
    if audit:
        result["catalog_request_audit"] = audit
    return result


def resolve_followup(request, history):
    """Fill a missing order only from an unambiguous conversational reference."""
    intake = analyze_request(request)
    if re.fullmatch(r"(?:订单|order)?\s*#?\s*\d{4,}", request.strip(), re.IGNORECASE):
        context = current_runtime()
        clarification = ((context.snapshot.get("conversation") or {}).get("clarification") if context else None) or {}
        if "order_id" in clarification.get("missing_slots", []):
            order_id = re.search(r"\d{4,}", request).group()
            return f"{clarification['request']}（订单 {order_id}）", clarification.get("entities", {}).get("source") == "dialogue_understanding"
        for message in reversed(history or []):
            if getattr(message, "type", "") != "human":
                continue
            prior = analyze_request(as_text(message.content))
            if prior.execution_mode == CLARIFICATION_REQUIRED and "order_id" in prior.missing_slots:
                order_id = re.search(r"\d{4,}", request).group()
                return f"{prior.request}（订单 {order_id}）", prior.execution_mode == LLM_REQUIRED
            break
    reference = any(word in request.lower() for word in ("刚才", "那个", "这个", "它", "呢", "再", "继续", " it", "that order", "this order", "same order"))
    if reference and "order_id" in intake.missing_slots:
        for message in reversed(history or []):
            prior = analyze_request(as_text(message.content))
            if len(prior.entities.get("order_ids", [])) > 1:
                return request, False
            order_id = prior.entities.get("order_id")
            if order_id:
                return f"(Context: the customer was last discussing order {order_id}.)\n{request}", False
    return request, False


def _run_orchestrator(user_message: str, history: list | None = None, pending_action: dict | None = None) -> dict:
    """Run the V1 intake and execution path without replaying the whole graph."""

    run_id = current_runtime().run_id if current_runtime() else uuid.uuid4().hex
    started_at = time.perf_counter()
    user_message, semantic_followup = resolve_followup(user_message, history)
    intake = (TaskIntake(user_message, execution_mode=LLM_REQUIRED, confidence=0.0,
                         reason="继续完成先前的结构化语义澄清")
              if semantic_followup else analyze_request(user_message))

    approval_decision = parse_approval_response(user_message, pending_action) if pending_action else None
    active_execution_mode = (
        "APPROVAL"
        if pending_action
        and approval_decision in {APPROVAL_CONFIRMED, APPROVAL_CANCELLED}
        else intake.execution_mode
    )
    if pending_action and approval_decision == APPROVAL_CONFIRMED:
        result = execute_approved_action(pending_action, user_message, run_id)
    elif pending_action and approval_decision == APPROVAL_CANCELLED:
        result = _simple_result(
            intake,
            run_id,
            reply="已取消待执行操作，没有修改订单、支付或退款状态。",
            agent="intake",
            route="APPROVAL",
            status="CANCELLED",
            terminal_reason="USER_CANCELLED_ACTION",
        )
    elif intake.execution_mode == DIRECT:
        result = execute_direct(intake, run_id)
    elif intake.execution_mode == CLARIFICATION_REQUIRED:
        result = _simple_result(
            intake,
            run_id,
            reply="请提供要处理的订单号，例如“查询订单 1004 的状态和物流”。" if "order_id" in intake.missing_slots else intake.reason,
            agent="intake",
            route="CLARIFY",
            status="WAITING_CLARIFICATION",
            terminal_reason="MISSING_ORDER_ID" if "order_id" in intake.missing_slots else "AMBIGUOUS_ORDER_GOALS",
        )
    elif intake.execution_mode == UNSUPPORTED:
        result = _simple_result(
            intake,
            run_id,
            reply="这个请求超出了当前消费服务助手的能力范围。我目前可以处理订单、物流、退款、支付状态和服务规则。",
            agent="intake",
            route="UNSUPPORTED",
            status="FAILED",
            terminal_reason="UNSUPPORTED_INTENT",
            error={"code": "UNSUPPORTED_INTENT", "message": intake.reason, "retryable": False},
        )
    elif intake.execution_mode == ESCALATION_REQUESTED:
        result = _explicit_escalation(intake, run_id, user_message)
    elif intake.execution_mode == LLM_REQUIRED:
        try:
            result = _execute_understanding(user_message, history, pending_action, intake, run_id)
            result.setdefault("run_id", run_id)
            result.setdefault("status", "SUCCEEDED")
            result.setdefault("terminal_reason", "COMPLETED")
            result.setdefault("events", [])
            result.setdefault("trace", [])
            result.setdefault("error", None)
            result.setdefault("pending_action", None)
            # Semantic compilation may add missing slots or a more specific
            # execution contract; preserve it for durable follow-up recovery.
            result.setdefault("intake", intake.to_dict())
            if not result["events"] or result["events"][-1].get("type") != "terminal":
                result["events"].append(
                    _event(
                        run_id,
                        len(result["events"]) + 1,
                        "terminal",
                        status=result["status"],
                        reason=result["terminal_reason"],
                    )
                )
        except Exception as exc:  # noqa: BLE001 - typed failed run, never fake escalation
            failure = classify_exception(exc, source="model")
            error = {"code": failure.code, "message": str(failure), "retryable": failure.retryable}
            result = _simple_result(
                intake,
                run_id,
                reply="当前模型服务暂时不可用，任务没有执行，也没有转交人工。请稍后重试。"
                if failure.code == "MODEL_UNAVAILABLE"
                else "任务执行失败，系统已保留明确的错误原因，请稍后重试。",
                agent="runtime",
                route=failure.code,
                status="FAILED",
                terminal_reason=failure.code,
                error=error,
            )
    else:  # pragma: no cover - defensive guard for future intake modes
        raise ValueError(f"Unknown intake mode: {intake.execution_mode}")

    # Browsing another read-only task must not silently discard an existing
    # approval. Only explicit confirmation/cancellation consumes that action.
    if pending_action and approval_decision not in {APPROVAL_CONFIRMED, APPROVAL_CANCELLED} and not result.get("pending_action"):
        result["pending_action"] = pending_action
    result["latency_ms"] = round((time.perf_counter() - started_at) * 1000, 2)
    logger.info(
        "agent_decision",
        extra={
            "extra_fields": {
                "run_id": run_id,
                "user_message": user_message,
                "execution_mode": active_execution_mode,
                "intents": list(intake.intents),
                "route": result.get("route"),
                "agent": result.get("agent"),
                "status": result.get("status"),
                "terminal_reason": result.get("terminal_reason"),
                "tools_called": [t.get("tool") for t in result.get("trace", []) if t.get("tool")],
                "usage": result.get("usage"),
            }
        },
    )
    return result


def run_orchestrator(user_message: str, history: list | None = None, pending_action: dict | None = None, *, limits=None) -> dict:
    """Shared runtime boundary used by API, web frontend and offline evaluation."""
    context = current_runtime()
    if context is None:
        context = RunRuntime(uuid.uuid4().hex, limits or BudgetLimits())
        with runtime_scope(context):
            return run_orchestrator(user_message, history, pending_action)
    context.record("intake", request=user_message)
    try:
        result = _run_orchestrator(user_message, history, pending_action)
    except AgentError as error:
        trace = [dict(run_id=context.run_id, agent="runtime", tool=event["tool"], args=event.get("args", {}),
                      result=event.get("result"), status=event.get("status"), duration_ms=event.get("duration_ms"))
                 for event in context.events if event["type"] == "tool_result"]
        status = "PARTIAL" if trace else "FAILED"
        context.record("terminal", status=status, reason=error.code)
        result = dict(run_id=context.run_id, reply="运行已停止，已保留此前获得的结果。" if trace else str(error),
                      trace=trace, agent="runtime", route="RUNTIME", status=status, terminal_reason=error.code,
                      error={"code": error.code, "message": str(error), "retryable": error.retryable}, pending_action=pending_action,
                      intake={}, usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, cached=False)
    result["events"] = context.events
    model_events = [e for e in context.events if e["type"] == "model_call"]
    if model_events or any(e["type"] == "model_started" for e in context.events):
        result["usage"] = sum_usage(*(e.get("usage") for e in model_events)) if model_events else {"input_tokens": None, "output_tokens": None, "total_tokens": None}
        if any(e["type"] == "error" and e.get("source") == "model" for e in context.events):
            result["usage"] = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    result["budget"] = {"limits": context.limits.__dict__, "consumed": dict(context.counters)}
    return result
