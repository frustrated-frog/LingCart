"""Deterministic fast path for high-confidence service tasks.

It deliberately executes only read-only tools. Requests that would change
payment/refund state stop at an explicit approval boundary.
"""

from __future__ import annotations

from src import tools
from src.intake import (
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
from src.runtime import ToolExecutor, TraceRecorder


_READ_ONLY_TOOLS = {
    "get_order_status": tools.get_order_status,
    "get_order_details": tools.get_order_details,
    "track_shipment": tools.track_shipment,
    "check_refund_eligibility": tools.check_refund_eligibility,
    "get_payment_status": tools.get_payment_status,
    "answer_faq": tools.answer_faq,
    "search_products": tools.search_products,
    "compare_products": tools.compare_products,
}

_APPROVED_TOOLS = {
    "initiate_refund": tools.initiate_refund,
    "process_payment": tools.process_payment,
}

_STATUS_ZH = {
    "placed": "已下单，等待发货",
    "shipped": "运输中",
    "delivered": "已送达",
    "cancelled": "已取消",
    "pending": "待支付",
    "paid": "已支付",
    "refunded": "已退款",
}

_TRACKING_ZH = {
    "Order placed, not yet shipped.": "订单已创建，暂未发货",
    "In transit.": "包裹正在运输中",
    "Delivered.": "包裹已送达",
    "Order was cancelled; no shipment in progress.": "订单已取消，没有进行中的物流",
}


def _trace_entry(run_id: str, agent: str, tool: str, args: dict, result, event: dict) -> dict:
    return {
        "run_id": run_id,
        "agent": agent,
        "tool": tool,
        "args": args,
        "result": result,
        "status": event["status"],
        "duration_ms": event["duration_ms"],
    }


def _primary_agent(intents: tuple[str, ...]) -> str:
    if REFUND_REQUEST in intents or REFUND_ELIGIBILITY in intents:
        return "refund"
    if PAYMENT_PROCESS in intents or PAYMENT_STATUS in intents:
        return "payment"
    if FAQ in intents:
        return "faq"
    return "order"


def execute_direct(intake: TaskIntake, run_id: str) -> dict:
    from src.task_plan import execute_plan
    return execute_plan(intake, run_id)


def execute_approved_action(pending_action: dict, user_message: str, run_id: str) -> dict:
    """Execute one explicitly confirmed side effect inside the current session."""

    action_type = pending_action.get("type")
    order_id = pending_action.get("order_id")
    intake = TaskIntake(
        request=user_message,
        intents=(action_type or "UNKNOWN_ACTION",),
        entities={"order_id": order_id} if order_id is not None else {},
        execution_mode=DIRECT,
        confidence=1.0,
        reason="用户确认了待执行操作",
    )
    recorder = TraceRecorder(run_id)
    recorder.record(
        "approval_decision",
        decision="CONFIRMED",
        action=action_type,
        order_id=order_id,
    )
    executor = ToolExecutor(_APPROVED_TOOLS, recorder, max_calls=1, approved_action=pending_action)
    trace: list[dict] = []

    if action_type == "PROCESS_PAYMENT":
        method = _payment_method(user_message) or pending_action.get("method")
        if not method:
            return _finish(
                recorder,
                intake,
                run_id,
                "payment",
                trace,
                reply="请明确支付方式，例如“确认使用 card 支付”或“确认使用 paypal 支付”。",
                status="WAITING_CLARIFICATION",
                terminal_reason="MISSING_PAYMENT_METHOD",
                pending_action=pending_action,
            )
        args = {"order_id": order_id, "method": method}
        result, event = executor.execute("process_payment", args)
        trace.append(_trace_entry(run_id, "payment", "process_payment", args, result, event))
        if result.get("error"):
            return _finish(recorder, intake, run_id, "payment", trace, reply=result["error"], status="FAILED",
                           terminal_reason="BUSINESS_DATA_NOT_FOUND", error={"code": "BUSINESS_DATA_NOT_FOUND", "message": result["error"], "retryable": False})
        if result.get("success"):
            return _finish(
                recorder,
                intake,
                run_id,
                "payment",
                trace,
                reply=f"已确认并完成订单 {order_id} 的支付，支付方式为 {method}。",
                status="SUCCEEDED",
                terminal_reason="COMPLETED",
            )
        return _finish(
            recorder,
            intake,
            run_id,
            "payment",
            trace,
            reply=f"没有重复执行支付：{result.get('reason', '当前状态不允许支付')} ",
            status="SUCCEEDED",
            terminal_reason="BUSINESS_RULE_REJECTED",
        )

    if action_type == "INITIATE_REFUND":
        reason = pending_action.get("reason") or "用户确认发起退款"
        args = {"order_id": order_id, "reason": reason}
        result, event = executor.execute("initiate_refund", args)
        trace.append(_trace_entry(run_id, "refund", "initiate_refund", args, result, event))
        if result.get("error"):
            return _finish(recorder, intake, run_id, "refund", trace, reply=result["error"], status="FAILED",
                           terminal_reason="BUSINESS_DATA_NOT_FOUND", error={"code": "BUSINESS_DATA_NOT_FOUND", "message": result["error"], "retryable": False})
        if result.get("success"):
            return _finish(
                recorder,
                intake,
                run_id,
                "refund",
                trace,
                reply=f"已按你的确认发起订单 {order_id} 的退款申请，退款单号为 {result['refund_id']}。",
                status="SUCCEEDED",
                terminal_reason="COMPLETED",
            )
        return _finish(
            recorder,
            intake,
            run_id,
            "refund",
            trace,
            reply=f"退款没有重复提交：{result.get('reason', '当前状态不允许退款')} ",
            status="SUCCEEDED",
            terminal_reason="BUSINESS_RULE_REJECTED",
        )

    return _finish(
        recorder,
        intake,
        run_id,
        "runtime",
        trace,
        reply="待确认操作已经失效，请重新发起请求。",
        status="FAILED",
        terminal_reason="INVALID_PENDING_ACTION",
        error={"code": "INVALID_PENDING_ACTION", "message": str(action_type), "retryable": False},
    )


def _payment_method(user_message: str) -> str | None:
    normalized = user_message.lower()
    if "paypal" in normalized:
        return "paypal"
    if "card" in normalized or "银行卡" in normalized or "信用卡" in normalized:
        return "card"
    return None


def _finish(
    recorder: TraceRecorder,
    intake: TaskIntake,
    run_id: str,
    agent: str,
    trace: list[dict],
    *,
    reply: str,
    status: str,
    terminal_reason: str,
    pending_action: dict | None = None,
    error: dict | None = None,
) -> dict:
    recorder.record("terminal", status=status, reason=terminal_reason)
    return {
        "run_id": run_id,
        "reply": reply,
        "trace": trace,
        "events": recorder.events,
        "agent": agent,
        "route": "DIRECT",
        "status": status,
        "terminal_reason": terminal_reason,
        "pending_action": pending_action,
        "error": error,
        "intake": intake.to_dict(),
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "cached": False,
    }
