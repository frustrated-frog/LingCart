import os
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import (
    create_engine,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    text,
)
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


class OrderItem(Base):
    """Immutable order-line identity used by long-lived business cases."""

    __tablename__ = "order_items"
    __table_args__ = (
        UniqueConstraint("order_id", "line_number", name="uq_order_item_line"),
        CheckConstraint("quantity > 0", name="ck_order_item_quantity_positive"),
        CheckConstraint("unit_price_minor >= 0", name="ck_order_item_price_non_negative"),
    )

    item_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.order_id"), index=True)
    line_number: Mapped[int] = mapped_column(Integer)
    product_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    name: Mapped[str] = mapped_column(String(200))
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    unit_price_minor: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(8), default="CNY")
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)


class PolicyVersion(Base):
    """Versioned policy source. Existing decisions keep pointing at this row."""

    __tablename__ = "policy_versions"
    __table_args__ = (
        UniqueConstraint("policy_id", "version", name="uq_policy_version"),
        CheckConstraint("status IN ('DRAFT', 'ACTIVE', 'RETIRED')", name="ck_policy_version_status"),
    )

    policy_version_id: Mapped[str] = mapped_column(String(96), primary_key=True)
    policy_id: Mapped[str] = mapped_column(String(64), index=True)
    version: Mapped[str] = mapped_column(String(32))
    policy_type: Mapped[str] = mapped_column(String(40), index=True)
    status: Mapped[str] = mapped_column(String(16))
    effective_from: Mapped[datetime] = mapped_column(DateTime, index=True)
    effective_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    rules: Mapped[dict] = mapped_column(JSON)
    source_ref: Mapped[str] = mapped_column(String(200))
    checksum: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime)


class AfterSalesCase(Base):
    """A long-lived business case; deliberately separate from Run and Conversation."""

    __tablename__ = "after_sales_cases"
    __table_args__ = (
        CheckConstraint("revision > 0", name="ck_after_sales_case_revision_positive"),
    )

    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_key: Mapped[str] = mapped_column(String(128), unique=True)
    case_type: Mapped[str] = mapped_column(String(32), index=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.order_id"), index=True)
    order_item_id: Mapped[str | None] = mapped_column(ForeignKey("order_items.item_id"), nullable=True)
    requested_resolution: Mapped[str] = mapped_column(String(40))
    reason_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    reason_text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(40), index=True)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    origin_session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    context: Mapped[dict] = mapped_column(JSON, default=dict)
    opened_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AfterSalesCaseRun(Base):
    """Many-to-many audit link between a business case and execution attempts."""

    __tablename__ = "after_sales_case_runs"
    __table_args__ = (
        UniqueConstraint("case_id", "run_id", name="uq_after_sales_case_run"),
    )

    link_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("after_sales_cases.case_id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.run_id"), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(24))
    linked_at: Mapped[datetime] = mapped_column(DateTime)


class AfterSalesEvidence(Base):
    """Immutable evidence envelope; payloads remain untrusted until policy evaluation."""

    __tablename__ = "after_sales_evidence"
    __table_args__ = (
        UniqueConstraint("case_id", "source_ref", "checksum", name="uq_after_sales_evidence_content"),
        CheckConstraint("status IN ('ACTIVE', 'STALE', 'REVOKED')", name="ck_after_sales_evidence_status"),
    )

    evidence_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("after_sales_cases.case_id"), index=True)
    evidence_type: Mapped[str] = mapped_column(String(48), index=True)
    source_type: Mapped[str] = mapped_column(String(32))
    source_ref: Mapped[str] = mapped_column(String(200))
    payload: Mapped[dict] = mapped_column(JSON)
    checksum: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), default="ACTIVE")
    observed_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class AfterSalesEvidenceValidation(Base):
    """不可变来源核验记录；不代表陈述内容真实或满足政策。"""

    __tablename__ = "after_sales_evidence_validations"

    evidence_id: Mapped[str] = mapped_column(ForeignKey("after_sales_evidence.evidence_id"), primary_key=True)
    checksum: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24))
    reason_code: Mapped[str] = mapped_column(String(64))
    validator_version: Mapped[str] = mapped_column(String(40))
    case_revision: Mapped[int] = mapped_column(Integer)
    validated_at: Mapped[datetime] = mapped_column(DateTime)


class PolicyDecision(Base):
    """Append-only policy verdict with the exact facts and policy identity used."""

    __tablename__ = "policy_decisions"
    __table_args__ = (
        UniqueConstraint("case_id", "input_hash", name="uq_policy_decision_input"),
        UniqueConstraint("case_id", "case_revision", name="uq_policy_decision_revision"),
        CheckConstraint("case_revision > 0", name="ck_policy_decision_revision_positive"),
    )

    decision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("after_sales_cases.case_id"), index=True)
    policy_version_id: Mapped[str] = mapped_column(ForeignKey("policy_versions.policy_version_id"), index=True)
    case_revision: Mapped[int] = mapped_column(Integer)
    input_hash: Mapped[str] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(32), index=True)
    reason_codes: Mapped[list] = mapped_column(JSON)
    facts: Mapped[dict] = mapped_column(JSON)
    evidence_refs: Mapped[list] = mapped_column(JSON)
    allowed_actions: Mapped[list] = mapped_column(JSON)
    evaluator_version: Mapped[str] = mapped_column(String(32))
    decided_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    supersedes_decision_id: Mapped[str | None] = mapped_column(
        ForeignKey("policy_decisions.decision_id"), nullable=True
    )


class AfterSalesCaseEvent(Base):
    """Append-only case timeline; sequence and revision advance together."""

    __tablename__ = "after_sales_case_events"
    __table_args__ = (
        UniqueConstraint("case_id", "sequence", name="uq_after_sales_case_event_sequence"),
        UniqueConstraint("case_id", "idempotency_key", name="uq_after_sales_case_event_idempotency"),
        CheckConstraint("sequence > 0", name="ck_after_sales_case_event_sequence_positive"),
        CheckConstraint("case_revision > 0", name="ck_after_sales_case_event_revision_positive"),
    )

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("after_sales_cases.case_id"), index=True)
    sequence: Mapped[int] = mapped_column(Integer)
    case_revision: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(48), index=True)
    actor_type: Mapped[str] = mapped_column(String(24))
    actor_id: Mapped[str | None] = mapped_column(String(96), nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(128))
    payload: Mapped[dict] = mapped_column(JSON)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class AfterSalesCaseRefund(Base):
    """Immutable link proving which approved case produced a refund request."""

    __tablename__ = "after_sales_case_refunds"
    __table_args__ = (
        UniqueConstraint("case_id", name="uq_after_sales_case_refund_case"),
        UniqueConstraint("refund_id", name="uq_after_sales_case_refund_refund"),
    )

    link_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("after_sales_cases.case_id"), index=True)
    refund_id: Mapped[int] = mapped_column(ForeignKey("refunds.refund_id"), index=True)
    policy_decision_id: Mapped[str] = mapped_column(ForeignKey("policy_decisions.decision_id"), index=True)
    submitted_run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.run_id"), nullable=True, index=True)
    linked_at: Mapped[datetime] = mapped_column(DateTime)


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


class SupportTicket(Base):
    __tablename__ = 'support_tickets'
    ticket_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    request_key: Mapped[str] = mapped_column(String(64), unique=True)
    reason: Mapped[str] = mapped_column(Text)
    context: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime)
