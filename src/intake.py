"""Deterministic task intake for common customer-service requests.

The intake layer intentionally handles only high-confidence patterns. Ambiguous
requests still fall through to the model router, while missing business slots
are clarified before any model or tool call is made.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


DIRECT = "DIRECT"
LLM_REQUIRED = "LLM_REQUIRED"
CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
UNSUPPORTED = "UNSUPPORTED"
ESCALATION_REQUESTED = "ESCALATION_REQUESTED"

ORDER_STATUS = "ORDER_STATUS"
ORDER_DETAILS = "ORDER_DETAILS"
SHIPMENT_TRACKING = "SHIPMENT_TRACKING"
REFUND_ELIGIBILITY = "REFUND_ELIGIBILITY"
REFUND_REQUEST = "REFUND_REQUEST"
PAYMENT_STATUS = "PAYMENT_STATUS"
PAYMENT_PROCESS = "PAYMENT_PROCESS"
FAQ = "FAQ"

APPROVAL_CONFIRMED = "CONFIRMED"
APPROVAL_CANCELLED = "CANCELLED"
APPROVAL_UNKNOWN = "UNKNOWN"

_ORDER_ID = re.compile(
    r"(?:订单|order)\s*(?:号|号码|id|number|no\.?|#)?\s*[:：#-]?\s*(\d{4,})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TaskIntake:
    request: str
    intents: tuple[str, ...] = ()
    entities: dict = field(default_factory=dict)
    missing_slots: tuple[str, ...] = ()
    execution_mode: str = LLM_REQUIRED
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["intents"] = list(self.intents)
        payload["missing_slots"] = list(self.missing_slots)
        return payload


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _extract_order_id(text: str) -> int | None:
    matches = _ORDER_ID.findall(text)
    return int(matches[-1]) if matches else None


def parse_approval_response(user_message: str) -> str:
    """Recognize an explicit answer to a previously proposed side effect."""

    normalized = " ".join(user_message.strip().lower().split())
    cancelled = ("取消", "算了", "不要", "不支付", "不退款", "cancel", "never mind", "no")
    confirmed = ("确认", "同意", "继续执行", "可以执行", "yes", "confirm", "proceed")
    if _has_any(normalized, cancelled):
        return APPROVAL_CANCELLED
    if _has_any(normalized, confirmed):
        return APPROVAL_CONFIRMED
    return APPROVAL_UNKNOWN


def analyze_request(user_message: str) -> TaskIntake:
    """Return a structured, conservative interpretation of a user request."""

    request = user_message.strip()
    normalized = " ".join(request.lower().split())
    order_id = _extract_order_id(normalized)
    entities = {"order_id": order_id} if order_id is not None else {}

    if _has_any(
        normalized,
        ("转人工", "人工客服", "真人客服", "找人工", "human agent", "talk to a human", "live agent"),
    ):
        return TaskIntake(
            request=request,
            intents=("ESCALATE",),
            entities=entities,
            execution_mode=ESCALATION_REQUESTED,
            confidence=1.0,
            reason="用户明确要求人工服务",
        )

    policy_words = ("规则", "政策", "时效", "多久", "policy", "how long", "shipping time", "return window")
    faq_topics = ("退货", "退款", "发货", "配送", "取消", "联系", "return", "refund", "shipping", "cancel", "contact")
    if _has_any(normalized, policy_words) and _has_any(normalized, faq_topics) and order_id is None:
        return TaskIntake(
            request=request,
            intents=(FAQ,),
            execution_mode=DIRECT,
            confidence=0.98,
            reason="识别为通用服务规则咨询",
        )

    intents: list[str] = []
    refund_words = ("退款", "退货", "退掉", "refund", "return this", "return order")
    payment_words = ("支付", "付款", "payment", "pay order", "pay for order")
    shipment_words = ("物流", "快递", "配送进度", "到哪", "运输", "追踪", "跟踪", "tracking", "shipment", "where is")
    status_words = ("状态", "status")
    detail_words = ("详情", "明细", "什么商品", "details")

    if _has_any(normalized, refund_words):
        eligibility_words = ("能不能", "能否", "可以退款", "是否可以", "资格", "eligible", "eligibility", "can i")
        request_words = ("想退款", "申请退款", "我要退款", "帮我退款", "退掉", "start a refund", "want a refund", "refund order")
        if _has_any(normalized, eligibility_words):
            intents.append(REFUND_ELIGIBILITY)
        elif _has_any(normalized, request_words):
            intents.append(REFUND_REQUEST)
        else:
            intents.append(REFUND_ELIGIBILITY)

    if _has_any(normalized, payment_words):
        status_only = ("支付状态", "付款状态", "支付了吗", "支付成功", "payment status", "paid yet", "gone through")
        process_words = ("帮我支付", "立即支付", "确认支付", "付款订单", "支付订单", "pay order", "pay for order")
        if _has_any(normalized, status_only):
            intents.append(PAYMENT_STATUS)
        elif _has_any(normalized, process_words):
            intents.append(PAYMENT_PROCESS)
        else:
            intents.append(PAYMENT_STATUS)

    if _has_any(normalized, shipment_words):
        intents.append(SHIPMENT_TRACKING)
    if _has_any(normalized, detail_words):
        intents.append(ORDER_DETAILS)
    asks_order_progress = "进度" in normalized and not _has_any(normalized, shipment_words)
    if (_has_any(normalized, status_words) or asks_order_progress) and not _has_any(
        normalized, ("支付状态", "付款状态", "refund status")
    ):
        intents.append(ORDER_STATUS)

    order_mentioned = "订单" in normalized or "order" in normalized
    query_words = ("查询", "查一下", "看看", "look up", "check")
    if order_mentioned and _has_any(normalized, query_words) and not intents:
        intents.append(ORDER_DETAILS)

    # Preserve order while removing duplicate matches from overlapping phrases.
    intents = list(dict.fromkeys(intents))
    order_intents = {
        ORDER_STATUS,
        ORDER_DETAILS,
        SHIPMENT_TRACKING,
        REFUND_ELIGIBILITY,
        REFUND_REQUEST,
        PAYMENT_STATUS,
        PAYMENT_PROCESS,
    }
    if any(intent in order_intents for intent in intents) and order_id is None:
        return TaskIntake(
            request=request,
            intents=tuple(intents),
            entities=entities,
            missing_slots=("order_id",),
            execution_mode=CLARIFICATION_REQUIRED,
            confidence=0.99,
            reason="业务任务缺少订单号",
        )

    if intents:
        return TaskIntake(
            request=request,
            intents=tuple(intents),
            entities=entities,
            execution_mode=DIRECT,
            confidence=0.96,
            reason="命中高置信度业务规则",
        )

    unsupported_words = (
        "天气",
        "股票",
        "基金",
        "机票",
        "酒店",
        "写代码",
        "weather",
        "stock price",
        "flight ticket",
    )
    if _has_any(normalized, unsupported_words):
        return TaskIntake(
            request=request,
            execution_mode=UNSUPPORTED,
            confidence=0.99,
            reason="请求超出消费服务业务范围",
        )

    return TaskIntake(
        request=request,
        entities=entities,
        execution_mode=LLM_REQUIRED,
        confidence=0.0,
        reason="确定性规则无法可靠判断，交由模型路由",
    )
