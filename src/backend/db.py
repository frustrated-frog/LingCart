import os
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import create_engine, Date, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "support_agent.db"


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_name: Mapped[str] = mapped_column(String(100))
    item: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20))  # placed/shipped/delivered/cancelled
    order_date: Mapped[date] = mapped_column(Date)
    amount: Mapped[float] = mapped_column(Float)


class Payment(Base):
    __tablename__ = "payments"

    payment_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.order_id"))
    status: Mapped[str] = mapped_column(String(20))  # pending/paid/refunded
    method: Mapped[str] = mapped_column(String(30))
    amount: Mapped[float] = mapped_column(Float)


class Refund(Base):
    __tablename__ = "refunds"

    refund_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.order_id"))
    status: Mapped[str] = mapped_column(String(20), default="none")  # none/requested/approved/completed
    reason: Mapped[str | None] = mapped_column(String(300), nullable=True)


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(40))
    price: Mapped[float] = mapped_column(Float)
    summary: Mapped[str] = mapped_column(Text)
    specs: Mapped[dict] = mapped_column(JSON)
    stock: Mapped[int] = mapped_column(Integer)


class ShipmentEvent(Base):
    __tablename__ = "shipment_events"

    event_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.order_id"), index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime)
    description: Mapped[str] = mapped_column(String(200))


class WorkspaceRun(Base):
    """Completed run snapshots for the local demo workspace, not checkpoints."""

    __tablename__ = "workspace_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(64), index=True)
    request: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    payload: Mapped[dict] = mapped_column(JSON)


def get_engine(db_url: str | None = None):
    default_url = f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"
    url = db_url or os.environ.get("SUPPORT_AGENT_DB_URL", default_url)
    kwargs = {"echo": False}
    if url.startswith("sqlite"):
        # FastAPI/TestClient may execute sync endpoints on a worker thread.
        # A shared in-memory database also needs one static connection or each
        # thread would see a different empty database.
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in {"sqlite:///:memory:", "sqlite://"}:
            kwargs["poolclass"] = StaticPool
    return create_engine(url, **kwargs)


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def get_session_factory(engine):
    return sessionmaker(bind=engine)
