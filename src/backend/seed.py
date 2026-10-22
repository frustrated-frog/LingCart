from datetime import date, datetime
from decimal import Decimal
import hashlib
import json

from sqlalchemy import select

from src.backend.db import (
    Order,
    OrderItem,
    Payment,
    PolicyVersion,
    Refund,
    get_engine,
    get_session_factory,
    init_db,
)
from src.backend.catalog import seed_catalog, seed_shipments

ORDERS = [
    # order_id, customer_name, item, status, order_date, amount
    (1001, "Alice Chen", "Wireless Headphones", "delivered", date(2026, 6, 1), 79.99),
    (1002, "Ben Torres", "Mechanical Keyboard", "shipped", date(2026, 6, 20), 129.00),
    (1003, "Carla Diaz", "USB-C Hub", "placed", date(2026, 6, 28), 34.50),
    (1004, "Dev Patel", "Standing Desk", "delivered", date(2026, 5, 15), 349.00),
    (1005, "Elena Novak", "Desk Lamp", "cancelled", date(2026, 6, 10), 24.99),
    (1006, "Farid Haidari", "Bluetooth Speaker", "delivered", date(2026, 6, 5), 59.99),
    (1007, "Grace Kim", "Laptop Stand", "shipped", date(2026, 6, 25), 42.00),
    (1008, "Hassan Ali", "Webcam 1080p", "placed", date(2026, 6, 29), 55.00),
    (1009, "Ines Moreau", "Ergonomic Mouse", "delivered", date(2026, 6, 12), 29.99),
    (1010, "Jonas Weber", "Noise-Cancelling Earbuds", "shipped", date(2026, 6, 27), 149.00),
]

PAYMENTS = [
    # payment_id, order_id, status, method, amount
    (1, 1001, "paid", "card", 79.99),
    (2, 1002, "paid", "card", 129.00),
    (3, 1003, "pending", "card", 34.50),
    (4, 1004, "paid", "paypal", 349.00),
    (5, 1005, "refunded", "card", 24.99),
    (6, 1006, "paid", "card", 59.99),
    (7, 1007, "paid", "paypal", 42.00),
    (8, 1008, "pending", "card", 55.00),
    (9, 1009, "paid", "card", 29.99),
    (10, 1010, "paid", "card", 149.00),
]

REFUNDS = [
    # refund_id, order_id, status, reason
    (1, 1001, "none", None),
    (2, 1002, "none", None),
    (3, 1003, "none", None),
    (4, 1004, "requested", "Item arrived damaged"),
    (5, 1005, "completed", "Order cancelled before shipping"),
    (6, 1006, "none", None),
    (7, 1007, "none", None),
    (8, 1008, "none", None),
    (9, 1009, "none", None),
    (10, 1010, "none", None),
]

DEFAULT_AFTER_SALES_POLICY_ID = "standard-return-refund"
DEFAULT_AFTER_SALES_POLICY_VERSION = "2026-01"
DEFAULT_AFTER_SALES_POLICY_VERSION_ID = (
    f"{DEFAULT_AFTER_SALES_POLICY_ID}:{DEFAULT_AFTER_SALES_POLICY_VERSION}"
)
DEFAULT_AFTER_SALES_POLICY_RULES = {
    "supported_case_types": ["RETURN", "REFUND"],
    "eligible_order_statuses": ["delivered"],
    "eligible_payment_statuses": ["paid"],
    "blocking_refund_statuses": ["requested", "approved", "completed"],
    "return_window_days": 30,
    "manual_review_amount_minor": 30000,
    "required_evidence_by_reason": {
        "DAMAGED": ["DAMAGE_DESCRIPTION"],
        "NOT_AS_DESCRIBED": ["ISSUE_DESCRIPTION"],
        "CHANGED_MIND": ["ITEM_CONDITION", "ORIGINAL_PACKAGING"],
    },
}


def _json_checksum(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _minor_units(amount: float) -> int:
    return int((Decimal(str(amount)) * 100).quantize(Decimal("1")))


def seed(session) -> None:
    """Add missing demo fixtures without overwriting live demo state."""

    for order_id, customer_name, item, status, order_date, amount in ORDERS:
        if session.get(Order, order_id) is None:
            session.add(
                Order(
                    order_id=order_id,
                    customer_name=customer_name,
                    item=item,
                    status=status,
                    order_date=order_date,
                    amount=amount,
                )
            )
    for payment_id, order_id, status, method, amount in PAYMENTS:
        payment_exists = session.get(Payment, payment_id) is not None or session.scalar(
            select(Payment.payment_id).where(Payment.order_id == order_id)
        ) is not None
        if not payment_exists:
            session.add(
                Payment(payment_id=payment_id, order_id=order_id, status=status, method=method, amount=amount)
            )
    for refund_id, order_id, status, reason in REFUNDS:
        refund_exists = session.get(Refund, refund_id) is not None or session.scalar(
            select(Refund.refund_id).where(Refund.order_id == order_id)
        ) is not None
        if not refund_exists:
            session.add(Refund(refund_id=refund_id, order_id=order_id, status=status, reason=reason))
    for order_id, _, item, _, _, amount in ORDERS:
        item_id = f"OI-{order_id}-1"
        if session.get(OrderItem, item_id) is None:
            session.add(
                OrderItem(
                    item_id=item_id,
                    order_id=order_id,
                    line_number=1,
                    name=item,
                    quantity=1,
                    unit_price_minor=_minor_units(amount),
                    currency="CNY",
                    attributes={"source": "LEGACY_ORDER_SNAPSHOT"},
                )
            )
    if session.get(PolicyVersion, DEFAULT_AFTER_SALES_POLICY_VERSION_ID) is None:
        session.add(
            PolicyVersion(
                policy_version_id=DEFAULT_AFTER_SALES_POLICY_VERSION_ID,
                policy_id=DEFAULT_AFTER_SALES_POLICY_ID,
                version=DEFAULT_AFTER_SALES_POLICY_VERSION,
                policy_type="AFTER_SALES",
                status="ACTIVE",
                effective_from=datetime(2026, 1, 1),
                rules=DEFAULT_AFTER_SALES_POLICY_RULES,
                source_ref="DEMO_POLICY:standard-return-refund:2026-01",
                checksum=_json_checksum(DEFAULT_AFTER_SALES_POLICY_RULES),
                created_at=datetime(2026, 1, 1),
            )
        )
    seed_catalog(session)
    session.flush()
    orders = session.scalars(select(Order)).all()
    seed_shipments(
        session,
        [(order.order_id, order.customer_name, order.item, order.status, order.order_date, order.amount) for order in orders],
    )
    session.commit()


if __name__ == "__main__":
    engine = get_engine()
    init_db(engine)
    session = get_session_factory(engine)()
    seed(session)
    print(f"Ensured {len(ORDERS)} orders, {len(PAYMENTS)} payments, {len(REFUNDS)} refunds exist.")
