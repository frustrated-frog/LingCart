import os
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import create_engine, Boolean, Date, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, text
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


class Conversation(Base):
    __tablename__ = "conversations"
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    state: Mapped[dict] = mapped_column(JSON, default=dict)
    active_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AgentRun(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (UniqueConstraint("session_id", "request_id"),)
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("conversations.session_id"), index=True)
    request_id: Mapped[str] = mapped_column(String(64))
    request: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="RUNNING")
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    lease_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    limits: Mapped[dict] = mapped_column(JSON)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    __table_args__ = (UniqueConstraint("run_id", "role"),)
    message_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("conversations.session_id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"))
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)


class AgentStep(Base):
    __tablename__ = "agent_steps"
    __table_args__ = (UniqueConstraint("run_id", "sequence"),)
    step_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(JSON)


class Checkpoint(Base):
    __tablename__ = "checkpoints"
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)


class ToolLedger(Base):
    __tablename__ = "tool_ledger"
    call_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), index=True)
    tool_name: Mapped[str] = mapped_column(String(80))
    args: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32))
    result: Mapped[dict | str | None] = mapped_column(JSON, nullable=True)


class PendingAction(Base):
    __tablename__ = "pending_actions"
    action_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("conversations.session_id"), index=True)
    source_run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"))
    status: Mapped[str] = mapped_column(String(24), default="OPEN")
    payload: Mapped[dict] = mapped_column(JSON)
    approved_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


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
    # Add constraints to existing SQLite files as well as fresh databases.
    # Duplicate legacy rows produce a migration error instead of being deleted.
    with engine.begin() as connection:
        connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_refund_order ON refunds(order_id)"))
        connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_payment_order ON payments(order_id)"))


def get_session_factory(engine):
    return sessionmaker(bind=engine)


class RunCommand(Base):
    """持久控制队列；command_id 同时承担客户端重试幂等键。"""
    __tablename__ = 'run_commands'
    command_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey('agent_runs.run_id'), index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey('conversations.session_id'), index=True)
    kind: Mapped[str] = mapped_column(String(24))
    payload: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(24), default='QUEUED')
    created_at: Mapped[datetime] = mapped_column(DateTime)
    outcome: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class RunControl(Base):
    __tablename__ = 'run_controls'
    run_id: Mapped[str] = mapped_column(ForeignKey('agent_runs.run_id'), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    closed: Mapped[bool] = mapped_column(Boolean, default=False)
