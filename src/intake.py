"""Deterministic task intake for common customer-service requests.

The intake layer intentionally handles only high-confidence patterns. Ambiguous
requests still fall through to structured dialogue understanding, while missing business slots
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
PRODUCT_SEARCH = "PRODUCT_SEARCH"
PRODUCT_DISCOVERY = "PRODUCT_DISCOVERY"
PRODUCT_COMPARE = "PRODUCT_COMPARE"
CATALOG_QUERY = "CATALOG_QUERY"

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
    route_hints: tuple[str, ...] = ()
    semantic_hazards: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["intents"] = list(self.intents)
        payload["missing_slots"] = list(self.missing_slots)
        payload["route_hints"] = list(self.route_hints)
        payload["semantic_hazards"] = list(self.semantic_hazards)
        return payload


def _has_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _extract_order_id(text: str) -> int | None:
    matches = _ORDER_ID.findall(text)
    return int(matches[-1]) if matches else None


def _semantic_hazards(text: str) -> tuple[str, ...]:
    """Detect scope operators that make lexical fast-path interpretation unsafe.

    This guard deliberately does not try to resolve polarity or intent.  It
    only withholds zero-token execution so structured understanding can model
    the operator and its scope.
    """
    patterns = {
        "NEGATION_SCOPE": r"(?:不要|不需要|不用|无需|(?<!分)别|并非|不是|(?<!能)不能|不想|排除|去掉|不包含|not\b|don't\b|do not\b|without\b|exclude\b|never\b)",
        "CORRECTION_SCOPE": r"(?:不是.+(?:而是|是)|改成|改为|算了|更正|actually\b|instead\b|rather than\b)",
        "EXCEPTION_SCOPE": r"(?:除了.+(?:都|之外)|除非|except\b|unless\b)",
    }
    folded = text.casefold()
    return tuple(name for name, pattern in patterns.items() if re.search(pattern, folded, re.IGNORECASE))


def handoff_is_negated(text: str) -> bool:
    """Return whether a handoff mention is explicitly under negative scope."""

    return bool(re.search(
        r"(?:不要|不需要|不用|无需|(?<!分)别|不想|不能)\s*(?:再\s*)?(?:帮我)?\s*(?:转人工|找人工|人工客服|真人客服)"
        r"|(?:don't|do not|not|never)\s+(?:transfer|connect|talk|escalate).{0,20}(?:human|agent)",
        text.casefold(),
        re.IGNORECASE,
    ))


def _catalog_terms(text: str, facets: dict) -> tuple[str | None, list[str]]:
    """Resolve terms from an explicit metadata snapshot supplied by a capability."""

    folded = text.casefold()
    categories = [str(value) for value in facets.get("categories", []) if str(value).casefold() in folded]
    category = max(categories, key=len) if categories else None
    if category is None:
        category = next((name for name, aliases in facets.get("category_aliases", {}).items()
                         if any(str(alias).casefold() in folded for alias in aliases)), None)
    matches: list[str] = []
    for raw in facets.get("searchable_values", []):
        value = str(raw).casefold()
        fragments = re.split(r"[\s·,/，、;；()（）]+", value)
        for fragment in fragments:
            if len(fragment) < 2:
                continue
            candidate = fragment if fragment in folded else next(
                (fragment[start:end]
                 for size in range(min(8, len(fragment)), 1, -1)
                 for start in range(0, len(fragment) - size + 1)
                 for end in (start + size,)
                 if fragment[start:end] in folded),
                None,
            )
            has_language = any(character.isalpha() or "\u4e00" <= character <= "\u9fff" for character in candidate or "")
            category_folded = (category or "").casefold()
            if (candidate and has_language and candidate not in matches
                    and candidate != category_folded and candidate not in category_folded):
                matches.append(candidate)
                break
    matches.sort(key=lambda value: (folded.find(value), -len(value), value))
    for alias, canonical in facets.get("search_aliases", {}).items():
        if str(alias).casefold() in folded and canonical not in matches:
            matches.append(str(canonical))
    return category, matches


def parse_approval_response(user_message: str, pending_action: dict | None = None) -> str:
    """Recognize an explicit answer to a previously proposed side effect."""

    normalized = " ".join(user_message.strip().lower().split())
    compact = re.sub(r"[\s，,。.!！?？]", "", normalized)
    english = re.sub(r"[,。.！!？?]", " ", normalized)
    english = " ".join(english.split())
    cancelled = {"取消", "取消操作", "取消支付", "取消退款", "算了", "不要", "不支付", "不退款", "取消先不要执行", "取消先不要支付", "取消不退款了"}
    if compact in cancelled or re.fullmatch(r"cancel(?: (?:it|this|the payment|the refund|never mind))?|never mind|no(?: thanks)?", english):
        return APPROVAL_CANCELLED
    confirmed = {"确认", "同意", "确认继续", "确认继续执行", "继续执行", "可以执行", "确认支付", "确认退款", "同意支付", "同意退款"}
    if compact in confirmed or re.fullmatch(r"确认(?:使用)?(?:card|paypal|银行卡|信用卡)支付", compact) or re.fullmatch(r"(?:yes(?: (?:please|proceed))?|confirm|proceed|confirm (?:payment|refund))", english):
        if pending_action:
            kind = pending_action.get("type")
            if ("退款" in normalized or "refund" in normalized) and kind != "INITIATE_REFUND":
                return APPROVAL_UNKNOWN
            if ("支付" in normalized or "payment" in normalized) and kind != "PROCESS_PAYMENT":
                return APPROVAL_UNKNOWN
        return APPROVAL_CONFIRMED
    return APPROVAL_UNKNOWN


def analyze_request(user_message: str, *, _check_clause_coverage: bool = True) -> TaskIntake:
    """Return a structured, conservative interpretation of a user request."""

    request = user_message.strip()
    normalized = " ".join(request.lower().split())
    order_ids = list(dict.fromkeys(int(value) for value in _ORDER_ID.findall(normalized)))
    # Keep explicitly listed sibling order numbers, without treating shopping
    # budgets as identifiers. More complex associations are clarified below.
    sibling = re.search(r"(?:订单|order).*?\b(\d{4,})\b\s*(?:和|及|、|and|,)\s*(?:订单\s*|order\s*)?(\d{4,})\b", normalized)
    if sibling:
        order_ids = list(dict.fromkeys([*order_ids, *(int(v) for v in sibling.groups())]))
    order_id = order_ids[-1] if order_ids else None
    entities = {"order_id": order_id} if order_id is not None else {}
    if len(order_ids) > 1:
        entities["order_ids"] = order_ids
    intents: list[str] = []
    hazards = _semantic_hazards(normalized)

    handoff_mentioned = _has_any(
        normalized,
        ("转人工", "人工客服", "真人客服", "找人工", "human agent", "talk to a human", "live agent"),
    )
    handoff_negated = handoff_is_negated(normalized)
    explicit_handoff = handoff_mentioned and not handoff_negated
    if explicit_handoff:
        return TaskIntake(
            request=request,
            intents=("ESCALATE",),
            entities=entities,
            execution_mode=ESCALATION_REQUESTED,
            confidence=1.0,
            reason="用户明确要求人工服务",
        )

    # A remembered order is useful for service follow-ups, but must not turn
    # an unrelated shopping request into an order task.
    shopping_text = re.sub(r"^\(context:.*?\)\s*", "", normalized)
    product_ids = re.findall(r"(?<![a-z0-9])p\d{4}s?(?![a-z0-9])", shopping_text, re.IGNORECASE)
    recommendation_action = _has_any(shopping_text, ("推荐", "选", "recommend"))
    listing_action = _has_any(shopping_text, ("搜索", "推荐", "search", "find", "recommend"))
    shopping_action = recommendation_action or _has_any(
        shopping_text, ("找", "对比", "比较", "搜索", "预算", "search", "compare", "find")
    )
    category, catalog_terms = None, []
    metadata_lookup_required = bool(
        product_ids
        or recommendation_action
        or _has_any(shopping_text, ("商品", "产品", "对比", "比较", "搜索", "找", "product", "catalog", "search", "compare", "find"))
    )
    if shopping_action and metadata_lookup_required:
        try:
            from src.capability_registry import get_capability_registry
            from src.catalog_metadata import CATALOG_FACETS_DEPENDENCY

            snapshot = get_capability_registry().metadata_snapshot(
                PRODUCT_SEARCH,
                CATALOG_FACETS_DEPENDENCY,
            )
            category, catalog_terms = _catalog_terms(shopping_text, snapshot.facets)
        except Exception:  # pragma: no cover - semantic fallback remains available
            category, catalog_terms = None, []
    if shopping_action and (category or product_ids or "商品" in shopping_text):
        budget_match = re.search(r"(?:预算\s*(?:为|是|不超过)?\s*[¥￥]?\s*)(-?\d+(?:\.\d+)?)", shopping_text)
        if not budget_match:
            budget_match = re.search(r"(\d+(?:\.\d+)?)\s*元?\s*(?:以内|以下|不超过)", shopping_text)
        if not budget_match:
            budget_match = re.search(r"(?:under|budget|max(?:imum)?)\s*(?:cny|rmb|[¥￥])?\s*(-?\d+(?:\.\d+)?)", shopping_text)
        shopping_entities = {"category": category, "query": ""}
        if budget_match:
            shopping_entities["max_price"] = float(budget_match.group(1))
        if product_ids:
            shopping_entities["product_ids"] = list(dict.fromkeys(value.upper() for value in product_ids))
        shopping_entities["query"] = " ".join(catalog_terms)
        entities.update(shopping_entities)
        if product_ids and category and listing_action:
            intents.append(PRODUCT_SEARCH)
        elif not product_ids:
            if recommendation_action and not catalog_terms and "max_price" not in shopping_entities:
                intents.append(PRODUCT_DISCOVERY)
            elif category or "商品" in shopping_text:
                intents.append(PRODUCT_SEARCH)
        if product_ids:
            intents.append(PRODUCT_COMPARE)

    policy_words = ("规则", "政策", "时效", "多久", "policy", "how long", "shipping time", "return window")
    faq_topics = ("退货", "退款", "发货", "配送", "取消", "联系", "return", "refund", "shipping", "cancel", "contact")
    policy_question = _has_any(normalized, policy_words) and _has_any(normalized, faq_topics)
    if policy_question:
        intents.append(FAQ)

    refund_words = ("退款", "退货", "退掉", "refund", "return this", "return order")
    payment_words = ("支付", "付款", "payment", "pay order", "pay for order")
    shipment_words = ("物流", "快递", "配送进度", "到哪", "运输", "追踪", "跟踪", "tracking", "shipment", "where is")
    status_words = ("状态", "status")
    detail_words = ("详情", "明细", "什么商品", "details")

    eligibility_words = ("能不能", "能否", "可以退款", "是否可以", "资格", "eligible", "eligibility", "can i")
    request_words = ("想退款", "申请退款", "我要退款", "帮我退款", "退掉", "start a refund", "want a refund", "refund order")
    explicit_refund_goal = _has_any(normalized, eligibility_words) or _has_any(normalized, request_words)
    # A pure policy question remains FAQ even when another clause mentions an
    # order. Explicit eligibility/action language still creates its own goal.
    if _has_any(normalized, refund_words) and (explicit_refund_goal or not policy_question):
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

    if _has_any(normalized, shipment_words) and not policy_question:
        intents.append(SHIPMENT_TRACKING)
    if _has_any(normalized, detail_words):
        intents.append(ORDER_DETAILS)
    asks_order_progress = "进度" in normalized and not _has_any(normalized, shipment_words)
    if (_has_any(normalized, status_words) or asks_order_progress) and not _has_any(
        normalized, ("支付状态", "付款状态", "payment status", "refund status")
    ):
        intents.append(ORDER_STATUS)

    order_mentioned = "订单" in normalized or "order" in normalized
    query_words = ("查询", "查一下", "看看", "look up", "check")
    if order_mentioned and _has_any(normalized, query_words) and not intents:
        intents.append(ORDER_DETAILS)

    # Preserve order while removing duplicate matches from overlapping phrases.
    intents = list(dict.fromkeys(intents))
    if intents and _check_clause_coverage:
        # The fast path is allowed only when it accounts for every coordinated
        # clause. Unknown siblings go to the open-set semantic layer instead of
        # being silently dropped because another clause matched a known rule.
        clauses = [part.strip() for part in re.split(
            r"(?:[，,；;]\s*)?(?:再|然后|另外|同时|顺便|并且|还要|then|also)\s*",
            normalized,
            flags=re.IGNORECASE,
        ) if part.strip()]
        if len(clauses) > 1:
            for clause in clauses:
                child = analyze_request(clause, _check_clause_coverage=False)
                bare_numbers = {int(value) for value in re.findall(
                    r"(?<![A-Za-z0-9])\d{4,}(?![A-Za-z0-9])", clause
                )}
                explicit_child_orders = set(child.entities.get("order_ids", []))
                if child.entities.get("order_id") is not None:
                    explicit_child_orders.add(child.entities["order_id"])
                if (not child.intents and child.execution_mode == LLM_REQUIRED) or bare_numbers - explicit_child_orders:
                    return TaskIntake(
                        request=request,
                        entities=entities,
                        execution_mode=LLM_REQUIRED,
                        confidence=0.0,
                        reason="复合请求包含确定性规则无法完整覆盖的子句",
                        route_hints=tuple(intents),
                    )
    if hazards:
        return TaskIntake(
            request=request,
            entities=entities,
            execution_mode=LLM_REQUIRED,
            confidence=0.0,
            reason="请求包含否定、修正或例外作用域，禁止确定性快路径直接解释",
            route_hints=tuple(dict.fromkeys(intents)),
            semantic_hazards=hazards,
        )
    order_intents = {
        ORDER_STATUS,
        ORDER_DETAILS,
        SHIPMENT_TRACKING,
        REFUND_ELIGIBILITY,
        REFUND_REQUEST,
        PAYMENT_STATUS,
        PAYMENT_PROCESS,
    }
    if len(order_ids) > 1 and sum(intent in order_intents for intent in intents) > 1 and not _has_any(normalized, ("分别", "都", "各", "each", "both")):
        return TaskIntake(request=request, intents=tuple(intents), entities=entities,
                          missing_slots=("order_intent_mapping",), execution_mode=CLARIFICATION_REQUIRED,
                          confidence=.99, reason="多个订单的目标关系不明确，请分别说明每个订单要处理什么")
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
        if not any(intent in order_intents for intent in intents):
            entities.pop("order_id", None)
            entities.pop("order_ids", None)
        return TaskIntake(
            request=request,
            intents=tuple(intents),
            entities=entities,
            execution_mode=DIRECT,
            confidence=0.96,
            reason="命中高置信度业务规则",
        )

    return TaskIntake(
        request=request,
        entities=entities,
        execution_mode=LLM_REQUIRED,
        confidence=0.0,
        reason="确定性规则无法可靠判断，交由结构化对话理解",
    )
