"""Public response-stream protocol helpers.

The protocol carries answer text and coarse execution state only. It must never
expose prompts, hidden reasoning, credentials, or raw tool observations.
"""
from __future__ import annotations

import hashlib

from src.errors import AgentError


AGENT_STATUS_LABELS = {
    "QUEUED": "任务已进入队列",
    "UNDERSTANDING": "正在理解你的需求",
    "PLANNING": "正在制定执行计划",
    "USING_TOOL": "正在调用工具获取证据",
    "VERIFYING": "正在核验结果",
    "RESPONDING": "正在组织回答",
    "WAITING_USER": "等待你确认或补充信息",
    "COMPLETED": "任务已完成",
    "FAILED": "任务未完成",
    "CANCELLED": "任务已取消",
}

TOOL_STATUS_LABELS = {
    "search_products": "正在检索商品目录",
    "discover_products": "正在发现候选商品",
    "compare_products": "正在比较候选商品",
    "get_catalog_facets": "正在读取筛选条件",
    "get_order_status": "正在查询订单状态",
    "get_order_details": "正在读取订单详情",
    "track_shipment": "正在查询物流进度",
    "get_payment_status": "正在核验支付状态",
    "check_refund_eligibility": "正在核验退款条件",
    "process_payment": "正在执行已确认的支付",
    "initiate_refund": "正在执行已确认的退款",
    "answer_faq": "正在检索服务说明",
    "calculate": "正在计算结果",
    "read_clock": "正在读取当前时间",
    "escalate_to_human": "正在准备人工协助",
}


def model_public_status(purpose: str) -> tuple[str, str]:
    value = (purpose or "").lower()
    if "final" in value or "response" in value:
        return "RESPONDING", AGENT_STATUS_LABELS["RESPONDING"]
    if "audit" in value or "verify" in value or "reflect" in value:
        return "VERIFYING", AGENT_STATUS_LABELS["VERIFYING"]
    if "planner" in value or "goal" in value or "decision" in value:
        return "PLANNING", AGENT_STATUS_LABELS["PLANNING"]
    return "UNDERSTANDING", AGENT_STATUS_LABELS["UNDERSTANDING"]


def terminal_public_status(status: str, pending_action=None) -> tuple[str, str]:
    if status == "WAITING_HUMAN":
        return "WAITING_USER", "正在等待人工协助"
    if pending_action or status.startswith("WAITING"):
        code = "WAITING_USER"
    elif status == "SUCCEEDED":
        code = "COMPLETED"
    elif status == "CANCELLED":
        code = "CANCELLED"
    else:
        code = "FAILED"
    return code, AGENT_STATUS_LABELS[code]


def tool_status_label(name: str) -> str:
    return TOOL_STATUS_LABELS.get(name, "正在调用工具获取证据")


def response_stream_id(run_id: str, identity: str) -> str:
    digest = hashlib.sha256(f"{run_id}\n{identity}".encode("utf-8")).hexdigest()[:24]
    return f"response-{digest}"


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def split_response(content: str, max_chars: int = 160) -> list[str]:
    if not content:
        return []
    return [content[index:index + max_chars] for index in range(0, len(content), max_chars)]


def response_text(events: list[dict], message_id: str) -> str:
    """Rebuild one response while rejecting conflicting or missing chunks."""
    chunks: dict[int, str] = {}
    for event in events:
        if event.get("type") != "response_delta" or event.get("message_id") != message_id:
            continue
        index = event.get("chunk_index")
        delta = event.get("delta")
        if type(index) is not int or index < 0 or not isinstance(delta, str):
            raise AgentError("RESPONSE_STREAM_CORRUPT", "回答流包含无效分片", retryable=False)
        if index in chunks and chunks[index] != delta:
            raise AgentError("RESPONSE_STREAM_CONFLICT", "回答流分片发生冲突", retryable=False)
        chunks[index] = delta
    if chunks and sorted(chunks) != list(range(max(chunks) + 1)):
        raise AgentError("RESPONSE_STREAM_GAP", "回答流分片不连续", retryable=True)
    return "".join(chunks[index] for index in sorted(chunks))
