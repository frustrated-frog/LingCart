"""Read models and run archive shared by all Streamlit workspace pages."""

import json
from datetime import UTC, datetime

from sqlalchemy import select

from src import tools
from src.backend.db import Order, Payment, Refund, WorkspaceRun

_ORDER_NAMES = {"Wireless Headphones": "无线耳机", "Mechanical Keyboard": "机械键盘", "USB-C Hub": "USB-C 扩展坞",
                "Standing Desk": "升降桌", "Desk Lamp": "台灯", "Bluetooth Speaker": "蓝牙音箱",
                "Laptop Stand": "笔记本支架", "Webcam 1080p": "摄像头", "Ergonomic Mouse": "鼠标", "Noise-Cancelling Earbuds": "降噪耳塞"}


def list_orders(query: str = "", status: str | None = None) -> list[dict]:
    with tools._session() as session:
        statement = select(Order, Payment, Refund).outerjoin(Payment, Payment.order_id == Order.order_id).outerjoin(
            Refund, Refund.order_id == Order.order_id).order_by(Order.order_date.desc(), Order.order_id.desc())
        if status:
            statement = statement.where(Order.status == status)
        rows = session.execute(statement).all()
        return [dict(order_id=order.order_id, item=order.item, customer_name=order.customer_name,
                     status=order.status, order_date=order.order_date.isoformat(), amount=order.amount,
                     payment_status=payment.status if payment else None,
                     refund_status=refund.status if refund else None,
                     refund_reason=refund.reason if refund else None)
                for order, payment, refund in rows
                if not query or query.casefold() in f"{order.order_id} {order.item} {_ORDER_NAMES.get(order.item, '')} {order.customer_name}".casefold()]


def save_run(conversation_id: str, request: str, result: dict) -> None:
    payload = json.loads(json.dumps(result, ensure_ascii=False, default=str))
    with tools._session() as session:
        if session.get(WorkspaceRun, result["run_id"]) is None:
            session.add(WorkspaceRun(run_id=result["run_id"], conversation_id=conversation_id,
                                     request=request, payload=payload,
                                     created_at=datetime.now(UTC).replace(tzinfo=None)))
            session.commit()


def list_runs(limit: int = 100) -> list[dict]:
    with tools._session() as session:
        runs = session.scalars(select(WorkspaceRun).order_by(WorkspaceRun.created_at.desc()).limit(limit)).all()
        return [dict(run_id=run.run_id, conversation_id=run.conversation_id, request=run.request,
                     created_at=run.created_at.replace(tzinfo=UTC).isoformat(), result=run.payload) for run in runs]
