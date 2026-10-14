"""Tool functions the agents call. Each hits the SQLite backend (read or write).

Docstrings are written for the LLM to read when deciding which tool to call.
"""
import logging
import json
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

from sqlalchemy import select

from src.backend.db import Order, Payment, Product, Refund, ShipmentEvent, get_engine, get_session_factory, init_db
from src.backend.catalog import seed_catalog, seed_shipments
from src.backend.seed import seed as _seed_data

logger = logging.getLogger("support_agent.tools")

_engine = get_engine()
init_db(_engine)
_SessionLocal = get_session_factory(_engine)
_transaction_session = ContextVar("business_transaction_session", default=None)


def _ensure_seeded() -> None:
    """Deployed environments (e.g. Streamlit Cloud) only ever run app.py, with
    no separate seed step - so seed on first import if the DB is empty."""
    with _SessionLocal() as s:
        if s.query(Order).count() == 0:
            _seed_data(s)
            logger.info("Auto-seeded empty database on startup.")
        else:
            seed_catalog(s)
            orders = s.scalars(select(Order)).all()
            seed_shipments(s, [(o.order_id, o.customer_name, o.item, o.status, o.order_date, o.amount) for o in orders])
            s.commit()


_ensure_seeded()


def configure_db(engine) -> None:
    """Point tools at a different engine (used by tests)."""
    global _engine, _SessionLocal
    _engine = engine
    init_db(_engine)
    _SessionLocal = get_session_factory(_engine)


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
                    available_only: bool = True, limit: int = 12) -> dict:
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
        matched = [product for product in products if all(term in (
            f"{product.product_id} {product.name} {product.summary} "
            + json.dumps(product.specs, ensure_ascii=False)).casefold() for term in terms)]
        return {"products": [_product_dict(product) for product in matched[:limit]], "total": len(matched),
                "filters": {"query": query, "category": category, "max_price": max_price,
                            "available_only": available_only}, "data_source": "DEMO_CATALOG"}


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


def answer_faq(question: str) -> str:
    """Answer a general policy question (shipping, returns, cancellations, refunds, contact info)."""
    q = question.lower()
    for keyword, answer in _FAQ_ZH.items():
        if keyword in question:
            return answer
    for keyword, answer in _FAQ.items():
        if keyword in q:
            return answer
    return (
        "I don't have a specific policy answer for that. "
        "You can reach human support at support@example.com for more detail."
    )


def escalate_to_human(reason: str, context: str) -> dict:
    """Escalate the conversation to a human agent when the request can't be resolved automatically."""
    ticket_id = uuid.uuid4().hex[:8]
    logger.warning("ESCALATION ticket=%s reason=%s context=%s", ticket_id, reason, context)
    return {
        "escalated": True,
        "ticket_id": ticket_id,
        "message": f"This has been escalated to a human agent (ticket #{ticket_id}). They'll follow up shortly.",
    }
