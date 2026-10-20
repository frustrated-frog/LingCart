"""Tool functions the agents call. Each hits the SQLite backend (read or write).

Docstrings are written for the LLM to read when deciding which tool to call.
"""
import ast
from datetime import datetime
import logging
import json
import math
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Annotated

from pydantic import Field
from sqlalchemy import select, text
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.backend.db import Order, Payment, Product, Refund, ShipmentEvent, get_engine, get_session_factory, init_db
from src.backend.seed import seed as _seed_data
from src.catalog_metadata import CatalogMetadataProvider

logger = logging.getLogger("support_agent.tools")

_engine = get_engine()
init_db(_engine)
_SessionLocal = get_session_factory(_engine)
_transaction_session = ContextVar("business_transaction_session", default=None)


def _ensure_seeded() -> None:
    """Add any missing demo fixtures without resetting existing state."""
    with _SessionLocal() as s:
        _seed_data(s)
        logger.info("Ensured missing demo fixtures on startup.")


_ensure_seeded()


def configure_db(engine) -> None:
    """Point tools at a different engine (used by tests)."""
    global _engine, _SessionLocal
    _engine = engine
    init_db(_engine)
    _SessionLocal = get_session_factory(_engine)
    # Catalog metadata is tied to the configured business-data source.  A DB
    # switch must never reuse a snapshot produced by the previous engine.
    provider = globals().get("_catalog_metadata_provider")
    if provider is not None:
        provider.invalidate()


def get_database_status() -> dict:
    """Return a safe connectivity check without exposing database details."""

    started = time.monotonic()
    try:
        with _SessionLocal() as session:
            session.execute(text("SELECT 1"))
    except Exception:
        logger.exception("Database readiness check failed")
        return {"ready": False, "message": "数据库连接失败"}
    return {
        "ready": True,
        "message": "数据库连接正常",
        "latency_ms": round((time.monotonic() - started) * 1000, 2),
    }


def read_clock(timezone: str = "Asia/Shanghai") -> dict:
    """Read the current wall clock from an explicit IANA timezone."""
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return {"error": f"未知 IANA 时区：{timezone}"}
    now = datetime.now(zone)
    return {
        "iso_datetime": now.isoformat(),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.isoweekday(),
        "timezone": timezone,
        "observed_at": now.isoformat(),
        "data_source": "RUNTIME_CLOCK",
    }


_ARITHMETIC_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.FloorDiv: lambda left, right: left // right,
    ast.Mod: lambda left, right: left % right,
    ast.Pow: lambda left, right: left ** right,
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


def calculate(expression: str) -> dict:
    """Evaluate bounded arithmetic without eval, names, calls or attribute access."""
    if not expression.strip() or len(expression) > 200:
        return {"error": "算式不能为空且长度不能超过 200 个字符"}
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return {"error": "算式格式无效"}
    nodes = list(ast.walk(tree))
    if len(nodes) > 64:
        return {"error": "算式过于复杂"}

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC_OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("指数超出限制")
            value = _ARITHMETIC_OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _ARITHMETIC_OPERATORS:
            value = _ARITHMETIC_OPERATORS[type(node.op)](evaluate(node.operand))
        else:
            raise ValueError("算式包含不允许的语法")
        if type(value) not in {int, float} or abs(value) > 1e15 or not math.isfinite(float(value)):
            raise ValueError("计算结果超出限制")
        return value

    try:
        result = evaluate(tree)
    except (ArithmeticError, ValueError) as exc:
        return {"error": str(exc)}
    return {"expression": expression, "result": result, "data_source": "DETERMINISTIC_CALCULATOR"}


@contextmanager
def _session():
    active = _transaction_session.get()
    if active is not None:
        yield active
    else:
        with _SessionLocal() as session:
            yield session


@contextmanager
def use_transaction(session):
    token = _transaction_session.set(session)
    try:
        yield
    finally:
        _transaction_session.reset(token)


def _commit(session):
    if _transaction_session.get() is None:
        session.commit()
    else:
        session.flush()


def get_order_status(order_id: int) -> dict:
    """Look up the current status (placed/shipped/delivered/cancelled) of an order."""
    with _session() as s:
        order = s.get(Order, order_id)
        if not order:
            return {"error": f"No order found with id {order_id}"}
        return {"order_id": order.order_id, "status": order.status}


def get_order_details(order_id: int) -> dict:
    """Get full details of an order: customer, item, status, date, amount."""
    with _session() as s:
        order = s.get(Order, order_id)
        if not order:
            return {"error": f"No order found with id {order_id}"}
        return {
            "order_id": order.order_id,
            "customer_name": order.customer_name,
            "item": order.item,
            "status": order.status,
            "order_date": order.order_date.isoformat(),
            "amount": order.amount,
        }


def track_shipment(order_id: int) -> dict:
    """Get shipment tracking info for an order based on its current status."""
    with _session() as s:
        order = s.get(Order, order_id)
        if not order:
            return {"error": f"No order found with id {order_id}"}
        messages = {
            "placed": "Order placed, not yet shipped.",
            "shipped": "In transit.",
            "delivered": "Delivered.",
            "cancelled": "Order was cancelled; no shipment in progress.",
        }
        return {
            "order_id": order.order_id,
            "status": order.status,
            "tracking_message": messages.get(order.status, "Unknown status."),
            "events": [dict(timestamp=event.occurred_at.isoformat(), description=event.description)
                       for event in s.scalars(select(ShipmentEvent).where(ShipmentEvent.order_id == order_id)
                                              .order_by(ShipmentEvent.occurred_at, ShipmentEvent.event_id))],
        }


def _product_dict(product: Product) -> dict:
    return dict(product_id=product.product_id, name=product.name, category=product.category,
                price=product.price, currency="CNY", summary=product.summary, specs=product.specs,
                stock=product.stock, data_source="DEMO_CATALOG")


def search_products(query: str = "", category: str | None = None, max_price: float | None = None,
                    available_only: bool = True, limit: int = 12,
                    excluded_terms: list[str] | None = None) -> dict:
    """Search the local, fictional demo catalog. Prices are in CNY."""
    if max_price is not None and max_price < 0:
        return {"error": "预算不能为负数"}
    if not 1 <= limit <= 60:
        return {"error": "返回数量必须在 1 到 60 之间"}
    with _session() as session:
        statement = select(Product).order_by(Product.price, Product.product_id)
        if category:
            statement = statement.where(Product.category == category)
        if max_price is not None:
            statement = statement.where(Product.price <= max_price)
        if available_only:
            statement = statement.where(Product.stock > 0)
        products = session.scalars(statement).all()
        terms = query.strip().casefold().split()
        exclusions = [term.strip().casefold() for term in (excluded_terms or []) if term.strip()]
        documents = {
            product.product_id: (f"{product.product_id} {product.name} {product.summary} "
                                 + json.dumps(product.specs, ensure_ascii=False)).casefold()
            for product in products
        }
        matched = [product for product in products
                   if all(term in documents[product.product_id] for term in terms)
                   and not any(term in documents[product.product_id] for term in exclusions)]
        filters = {"query": query, "category": category, "max_price": max_price,
                   "available_only": available_only}
        if excluded_terms:
            filters["excluded_terms"] = excluded_terms
        return {"products": [_product_dict(product) for product in matched[:limit]], "total": len(matched),
                "filters": filters,
                "data_source": "DEMO_CATALOG"}


def get_catalog_facets(available_only: bool = True) -> dict:
    """Describe live catalog categories and observable fields without a fixed ontology."""
    with _session() as session:
        statement = select(Product).order_by(Product.category, Product.product_id)
        if available_only:
            statement = statement.where(Product.stock > 0)
        products = session.scalars(statement).all()
        categories = sorted({product.category for product in products})
        prices = [float(product.price) for product in products]
        spec_keys: dict[str, set[str]] = {}
        searchable_values: set[str] = set()
        for product in products:
            spec_keys.setdefault(product.category, set()).update(str(key) for key in (product.specs or {}))
            searchable_values.update(
                str(value) for value in (product.specs or {}).values()
                if isinstance(value, (str, int, float)) and not isinstance(value, bool)
            )
            searchable_values.update(part.strip() for part in product.summary.split("·") if part.strip())
        from src.backend.catalog import CATEGORY_ALIASES, SEARCH_ALIASES
        return {
            "categories": categories,
            "category_aliases": {category: list(CATEGORY_ALIASES.get(category, ())) for category in categories},
            "search_aliases": dict(SEARCH_ALIASES),
            "price_range": {"min": min(prices), "max": max(prices)} if prices else None,
            "spec_keys": {category: sorted(keys) for category, keys in sorted(spec_keys.items())},
            "searchable_values": sorted(searchable_values),
            "available_products": len(products),
            "data_source": "DEMO_CATALOG",
        }


_catalog_metadata_provider = CatalogMetadataProvider(
    lambda: get_catalog_facets(),
    source="DEMO_CATALOG",
)


def get_catalog_metadata_provider():
    """Return the process-local catalog metadata boundary.

    The loader resolves ``get_catalog_facets`` at call time so diagnostics and
    the offline evaluator can still observe the physical metadata read.
    """

    return _catalog_metadata_provider


def discover_products(query: str = "", category: str | None = None, max_price: float | None = None,
                      soft_preferences: list[str] | None = None, excluded_terms: list[str] | None = None,
                      limit: Annotated[int, Field(ge=1, le=12)] = 4,
                      context_seed: str = "") -> dict:
    """Return an evidence-bound, diverse discovery slate from the live catalog."""
    if max_price is not None and max_price < 0:
        return {"error": "预算不能为负数"}
    if not 1 <= limit <= 12:
        return {"error": "探索推荐数量必须在 1 到 12 之间"}
    with _session() as session:
        statement = select(Product).where(Product.stock > 0).order_by(Product.product_id)
        if category:
            statement = statement.where(Product.category == category)
        if max_price is not None:
            statement = statement.where(Product.price <= max_price)
        products = session.scalars(statement).all()
        exclusions = [term.strip().casefold() for term in (excluded_terms or []) if term.strip()]
        if exclusions:
            products = [product for product in products if not any(
                term in (f"{product.product_id} {product.name} {product.category} {product.summary} "
                         + json.dumps(product.specs, ensure_ascii=False)).casefold()
                for term in exclusions
            )]
        terms = query.strip().casefold().split()
        unresolved_query = None
        if terms:
            documents = {
                product.product_id: (
                f"{product.product_id} {product.name} {product.category} {product.summary} "
                + json.dumps(product.specs, ensure_ascii=False)).casefold()
                for product in products
            }
            matched = [product for product in products
                       if all(term in documents[product.product_id] for term in terms)]
            # If none of the model's literal terms exist anywhere in the live
            # schema, they are not a valid hard catalog binding. Preserve this
            # uncertainty and offer a diagnostic slate instead of applying a
            # made-up filter or returning a false unsupported result.
            if not matched and not any(term in document for term in terms for document in documents.values()):
                unresolved_query = query
            else:
                products = matched
        from src.recommendation import select_discovery_slate
        result = select_discovery_slate(
            [_product_dict(product) for product in products],
            query=query,
            category=category,
            max_price=max_price,
            soft_preferences=soft_preferences,
            limit=limit,
            context_seed=context_seed,
        )
        if unresolved_query and result.get("products") and not result.get("error"):
            result["unbound_catalog_language"] = [unresolved_query]
            result["interaction_policy"] = "RECOMMEND_AND_ASK"
            result["personalization"] = "EXPLORATORY"
        return result


def compare_products(product_ids: list[str]) -> dict:
    """Return real catalog records for two or three distinct products."""
    ids = list(dict.fromkeys(str(value).upper() for value in product_ids))
    if not 2 <= len(ids) <= 3:
        return {"error": "请选择 2 到 3 款不同商品进行对比"}
    with _session() as session:
        products = [session.get(Product, product_id) for product_id in ids]
        missing = [product_id for product_id, product in zip(ids, products) if product is None]
        if missing:
            return {"error": f"商品不存在：{', '.join(missing)}"}
        return {"products": [_product_dict(product) for product in products], "data_source": "DEMO_CATALOG"}


def check_refund_eligibility(order_id: int) -> dict:
    """Check whether an order is eligible for a refund. Rule: order must be
    'delivered' and must not already have a requested/approved/completed refund."""
    with _session() as s:
        order = s.get(Order, order_id)
        if not order:
            return {"error": f"No order found with id {order_id}"}
        refund = s.query(Refund).filter_by(order_id=order_id).first()
        refund_status = refund.status if refund else "none"

        if order.status != "delivered":
            return {
                "order_id": order_id,
                "eligible": False,
                "reason": f"Order status is '{order.status}'; only delivered orders are refund-eligible.",
            }
        if refund_status != "none":
            return {
                "order_id": order_id,
                "eligible": False,
                "reason": f"A refund is already '{refund_status}' for this order.",
            }
        return {"order_id": order_id, "eligible": True, "reason": "Order is delivered and has no existing refund."}


def initiate_refund(order_id: int, reason: str) -> dict:
    """Start a refund for an order. MODIFIES STATE. Must only be called after
    check_refund_eligibility confirms eligibility; refuses otherwise."""
    eligibility = check_refund_eligibility(order_id)
    if eligibility.get("error"):
        return eligibility
    if not eligibility["eligible"]:
        return {"success": False, "order_id": order_id, "reason": eligibility["reason"]}

    with _session() as s:
        refund = s.query(Refund).filter_by(order_id=order_id).first()
        if refund is None:
            refund = Refund(order_id=order_id, status="requested", reason=reason)
            s.add(refund)
        else:
            refund.status = "requested"
            refund.reason = reason
        _commit(s)
        return {"success": True, "order_id": order_id, "refund_id": refund.refund_id, "status": "requested"}


def get_payment_status(order_id: int) -> dict:
    """Look up the payment status (pending/paid/refunded) for an order."""
    with _session() as s:
        payment = s.query(Payment).filter_by(order_id=order_id).first()
        if not payment:
            return {"error": f"No payment found for order {order_id}"}
        return {
            "order_id": order_id,
            "status": payment.status,
            "method": payment.method,
            "amount": payment.amount,
        }


def process_payment(order_id: int, method: str) -> dict:
    """Process a pending payment for an order using the given method. MODIFIES STATE."""
    if method not in {"card", "paypal"}:
        return {"success": False, "order_id": order_id, "reason": "Unsupported payment method; choose card or paypal."}
    with _session() as s:
        payment = s.query(Payment).filter_by(order_id=order_id).first()
        if not payment:
            return {"error": f"No payment found for order {order_id}"}
        if payment.status == "paid":
            return {"success": False, "order_id": order_id, "reason": "Payment is already paid."}
        if payment.status == "refunded":
            return {"success": False, "order_id": order_id, "reason": "Payment was refunded; cannot reprocess."}

        payment.status = "paid"
        payment.method = method
        _commit(s)
        return {"success": True, "order_id": order_id, "status": "paid", "method": method}


_FAQ = {
    "shipping": "Standard shipping takes 3-5 business days. Expedited shipping (1-2 days) is available at checkout.",
    "return": "Items can be returned within 30 days of delivery for a full refund, provided they are unused and in original packaging.",
    "cancel": "Orders can be cancelled free of charge before they ship. Once shipped, they must be returned instead.",
    "refund": "Refunds are issued to the original payment method within 5-7 business days of approval.",
    "contact": "You can reach human support at support@example.com, Mon-Fri 9am-6pm.",
}

_FAQ_ZH = {
    "配送": "标准配送通常需要 3–5 个工作日；结算时可选择 1–2 个工作日送达的加急配送。",
    "发货": "标准配送通常需要 3–5 个工作日；结算时可选择 1–2 个工作日送达的加急配送。",
    "退货": "商品送达后 30 天内可以申请全额退款，但商品需保持未使用并保留原包装。",
    "取消": "订单发货前可以免费取消；发货后需要按照退货流程处理。",
    "退款": "退款审核通过后，通常会在 5–7 个工作日内原路退回。",
    "联系": "人工客服邮箱为 support@example.com，服务时间为周一至周五 9:00–18:00。",
}


def answer_faq(question: str) -> dict:
    """检索演示政策，返回有版本的依据；NOT_FOUND 不代表问题已经解决。"""
    q = question.lower()
    policies = []
    ids = {'配送': 'shipping', '发货': 'shipping', '退货': 'return', '取消': 'cancel', '退款': 'refund', '联系': 'contact'}
    for keyword, answer in _FAQ_ZH.items():
        if keyword in question and ids[keyword] not in [p['policy_id'] for p in policies]:
            policies.append(dict(policy_id=ids[keyword], version='demo-v1', text=answer, source='DEMO_POLICY'))
    if not policies:
        for keyword, answer in _FAQ.items():
            if keyword in q:
                policies.append(dict(policy_id=keyword, version='demo-v1', text=answer, source='DEMO_POLICY'))
    return dict(status='ANSWERED' if policies else 'NOT_FOUND', question=question, policies=policies,
                answer='\n'.join(p['text'] for p in policies) if policies else '本地演示政策没有相关依据，当前问题尚未解决。',
                data_source='DEMO_POLICY')


def escalate_to_human(reason: str, context: str) -> dict:
    """Escalate the conversation to a human agent when the request can't be resolved automatically."""
    from src.backend.db import SupportTicket
    from src.runtime import current_runtime
    runtime = current_runtime()
    request_key = runtime.run_id if runtime else uuid.uuid4().hex
    with _session() as session:
        ticket = session.scalar(select(SupportTicket).where(SupportTicket.request_key == request_key))
        if ticket is None:
            from datetime import UTC, datetime
            ticket = SupportTicket(ticket_id=uuid.uuid4().hex, request_key=request_key, reason=reason,
                context=context, status='QUEUED', created_at=datetime.now(UTC).replace(tzinfo=None))
            session.add(ticket)
            _commit(session)
        return dict(escalated=False, ticket_id=ticket.ticket_id, status=ticket.status, mode='DEMO',
                    delivered=False, message='人工请求已保存到本地演示队列；尚未接入真人客服或通知服务。')
