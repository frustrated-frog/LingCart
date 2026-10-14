import logging
import re
import time
import uuid
from datetime import UTC, datetime
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.base import as_text, sum_usage
from src.agents.faq_agent import run_faq_agent
from src.agents.order_agent import run_order_agent
from src.agents.payment_agent import run_payment_agent
from src.agents.refund_agent import run_refund_agent
from src.direct_executor import execute_approved_action, execute_direct
from src.errors import classify_exception
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
from src.llm import extract_usage, get_llm
from src.logging_config import configure_logging
from src.runtime import TraceRecorder
from src.tools import escalate_to_human

configure_logging()
logger = logging.getLogger("support_agent.orchestrator")

ROUTES = ("ORDER", "REFUND", "PAYMENT", "FAQ", "ESCALATE")

ROUTER_PROMPT = """You are a customer-support router. Read the user's message and decide which
specialist should handle it. Respond with ONLY one word from this list:
- ORDER    : order status, tracking, order details
- REFUND   : refund requests, refund status, eligibility
- PAYMENT  : payment status, making a payment
- FAQ      : general policy questions (shipping times, return policy)
- ESCALATE : an explicit human-support request, abuse, or a truly unsupported request

Do not choose ESCALATE merely because information is missing; the application handles clarification.

User message: {user_message}
Answer with one word only."""


class SupportState(TypedDict, total=False):
    user_message: str
    history: list
    route: str
    reply: str
    trace: list
    events: list
    agent: str
    usage: dict
    cached: bool
    status: str
    terminal_reason: str
    run_id: str
    error: dict | None


def _event(run_id: str, sequence: int, event_type: str, **data) -> dict:
    return {
        "run_id": run_id,
        "sequence": sequence,
        "timestamp": datetime.now(UTC).isoformat(),
        "type": event_type,
        **data,
    }


def parse_route(text: str) -> str:
    normalized = text.strip().upper()
    for match in re.finditer(r"\b(ORDER|REFUND|PAYMENT|FAQ|ESCALATE)\b", normalized):
        return match.group(1)
    return "ESCALATE"


def route_node(state: SupportState) -> dict:
    run_id = state.get("run_id") or uuid.uuid4().hex
    started_at = time.perf_counter()
    llm = get_llm()
    result = llm.invoke(ROUTER_PROMPT.format(user_message=state["user_message"]))
    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    route = parse_route(as_text(result.content))
    events = list(state.get("events") or [])
    events.append(
        _event(
            run_id,
            len(events) + 1,
            "model_call",
            purpose="route",
            status="SUCCEEDED",
            duration_ms=duration_ms,
            usage=extract_usage(result),
        )
    )
    events.append(
        _event(run_id, len(events) + 1, "route_decision", source="model", route=route, confidence=None)
    )
    return {"route": route, "usage": extract_usage(result), "events": events, "run_id": run_id}


def _make_agent_node(agent_name: str, run_fn):
    def node(state: SupportState) -> dict:
        result = run_fn(state["user_message"], state.get("history"), run_id=state.get("run_id"))
        prior_events = list(state.get("events") or [])
        for event in result.get("events") or []:
            prior_events.append({**event, "sequence": len(prior_events) + 1})
        return {
            "reply": result["reply"],
            "trace": result["trace"],
            "events": prior_events,
            "agent": agent_name,
            "usage": sum_usage(state.get("usage", {}), result.get("usage", {})),
            "cached": result.get("cached", False),
            "status": result.get("status", "SUCCEEDED"),
            "terminal_reason": result.get("terminal_reason", "COMPLETED"),
            "error": result.get("error"),
        }

    return node


def escalate_node(state: SupportState) -> dict:
    run_id = state.get("run_id") or uuid.uuid4().hex
    reason = "模型路由判断该请求需要人工客服"
    result = escalate_to_human(reason=reason, context=state["user_message"])
    trace = [
        {
            "run_id": run_id,
            "agent": "escalate",
            "tool": "escalate_to_human",
            "args": {"reason": reason, "context": state["user_message"]},
            "result": result,
            "status": "SUCCEEDED",
            "duration_ms": 0.0,
        }
    ]
    events = list(state.get("events") or [])
    events.append(_event(run_id, len(events) + 1, "tool_call", tool="escalate_to_human", args=trace[0]["args"]))
    events.append(
        _event(
            run_id,
            len(events) + 1,
            "tool_result",
            tool="escalate_to_human",
            result=result,
            status="SUCCEEDED",
            duration_ms=0.0,
        )
    )
    return {
        "reply": f"当前请求已转交人工客服（工单 #{result['ticket_id']}），处理进展会在后续通知你。",
        "trace": trace,
        "events": events,
        "agent": "escalate",
        "usage": state.get("usage", {}),
        "status": "ESCALATED",
        "terminal_reason": "MODEL_ROUTED_ESCALATION",
    }


def build_graph():
    graph = StateGraph(SupportState)
    graph.add_node("route", route_node)
    graph.add_node("order", _make_agent_node("order", run_order_agent))
    graph.add_node("refund", _make_agent_node("refund", run_refund_agent))
    graph.add_node("payment", _make_agent_node("payment", run_payment_agent))
    graph.add_node("faq", _make_agent_node("faq", run_faq_agent))
    graph.add_node("escalate", escalate_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {"ORDER": "order", "REFUND": "refund", "PAYMENT": "payment", "FAQ": "faq", "ESCALATE": "escalate"},
    )
    for node in ("order", "refund", "payment", "faq", "escalate"):
        graph.add_edge(node, END)

    return graph.compile()


_GRAPH = build_graph()


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
    reason = "用户明确要求人工客服"
    result = escalate_to_human(reason=reason, context=user_message)
    trace = [
        {
            "run_id": run_id,
            "agent": "escalate",
            "tool": "escalate_to_human",
            "args": {"reason": reason, "context": user_message},
            "result": result,
            "status": "SUCCEEDED",
            "duration_ms": 0.0,
        }
    ]
    response = _simple_result(
        intake,
        run_id,
        reply=f"已按你的要求转交人工客服（工单 #{result['ticket_id']}），处理进展会在后续通知你。",
        agent="escalate",
        route="ESCALATE",
        status="ESCALATED",
        terminal_reason="USER_REQUESTED_ESCALATION",
        trace=trace,
    )
    response["events"].insert(
        1,
        _event(
            run_id,
            2,
            "tool_result",
            tool="escalate_to_human",
            args=trace[0]["args"],
            result=result,
            status="SUCCEEDED",
            duration_ms=0.0,
        ),
    )
    for sequence, event in enumerate(response["events"], start=1):
        event["sequence"] = sequence
    return response


def run_orchestrator(user_message: str, history: list | None = None, pending_action: dict | None = None) -> dict:
    """Run the V1 intake and execution path without replaying the whole graph."""

    run_id = uuid.uuid4().hex
    started_at = time.perf_counter()
    intake = analyze_request(user_message)

    approval_decision = parse_approval_response(user_message) if pending_action else None
    active_execution_mode = (
        "APPROVAL"
        if pending_action
        and (approval_decision in {APPROVAL_CONFIRMED, APPROVAL_CANCELLED} or intake.execution_mode == LLM_REQUIRED)
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
    elif pending_action and intake.execution_mode == LLM_REQUIRED:
        result = _simple_result(
            intake,
            run_id,
            reply="当前仍有一项待确认操作。请明确回复“确认”继续，或回复“取消”放弃。",
            agent="intake",
            route="APPROVAL",
            status="WAITING_APPROVAL",
            terminal_reason="APPROVAL_REQUIRED",
            pending_action=pending_action,
        )
    elif intake.execution_mode == DIRECT:
        result = execute_direct(intake, run_id)
    elif intake.execution_mode == CLARIFICATION_REQUIRED:
        result = _simple_result(
            intake,
            run_id,
            reply="请提供要处理的订单号，例如“查询订单 1004 的状态和物流”。",
            agent="intake",
            route="CLARIFY",
            status="WAITING_CLARIFICATION",
            terminal_reason="MISSING_ORDER_ID",
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
            result = _GRAPH.invoke(
                {
                    "user_message": user_message,
                    "history": history or [],
                    "trace": [],
                    "events": [],
                    "run_id": run_id,
                }
            )
            result.setdefault("run_id", run_id)
            result.setdefault("status", "SUCCEEDED")
            result.setdefault("terminal_reason", "COMPLETED")
            result.setdefault("events", [])
            result.setdefault("trace", [])
            result.setdefault("error", None)
            result.setdefault("pending_action", None)
            result["intake"] = intake.to_dict()
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
