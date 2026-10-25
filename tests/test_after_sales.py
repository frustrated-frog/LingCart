from datetime import datetime

import pytest
from sqlalchemy import select

from src.after_sales import (
    ActorType,
    AfterSalesCaseService,
    AfterSalesDomainError,
    AfterSalesPolicyEngine,
    CaseCreateCommand,
    CaseStatus,
    CaseType,
    EvidenceCommand,
    PolicyOutcome,
    RequestedResolution,
)
from src.backend.db import (
    AfterSalesCaseEvent,
    AgentRun,
    Conversation,
    Order,
    OrderItem,
    Payment,
    PolicyDecision,
    PolicyVersion,
    Refund,
    get_engine,
    get_session_factory,
    init_db,
)
from src.backend.seed import (
    DEFAULT_AFTER_SALES_POLICY_RULES,
    DEFAULT_AFTER_SALES_POLICY_VERSION_ID,
    seed,
)


@pytest.fixture()
def session():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    with get_session_factory(engine)() as db_session:
        seed(db_session)
        db_session.add(Conversation(session_id="conversation-1", state={}))
        db_session.flush()
        db_session.add(AgentRun(run_id="evidence-run", session_id="conversation-1", request_id="evidence",
                                request="包装完好但商品外壳开裂；实际功能与商品说明不一致", status="SUCCEEDED",
                                created_at=datetime(2026, 5, 25), limits={}))
        db_session.commit()
        yield db_session


def _command(request_key="case-request-1", **overrides):
    values = {
        "request_key": request_key,
        "case_type": CaseType.REFUND,
        "order_id": 1001,
        "requested_resolution": RequestedResolution.REFUND,
        "reason_code": "DAMAGED",
        "reason_text": "商品外壳到货时已经破损",
        "origin_session_id": "conversation-1",
        "context": {"channel": "web"},
    }
    values.update(overrides)
    return CaseCreateCommand(**values)


def _evidence(description="包装完好但商品外壳开裂"):
    return EvidenceCommand(
        evidence_type="DAMAGE_DESCRIPTION",
        source_type="CUSTOMER_STATEMENT",
        source_ref="agent_runs:evidence-run:request",
        payload={"statement": description},
        observed_at=datetime(2026, 6, 7, 9, 0),
    )


def test_seed_adds_order_items_and_versioned_policy_without_overwriting(session):
    items = session.scalars(select(OrderItem).order_by(OrderItem.order_id)).all()
    policy = session.get(PolicyVersion, DEFAULT_AFTER_SALES_POLICY_VERSION_ID)

    assert len(items) == 10
    assert items[0].item_id == "OI-1001-1"
    assert items[0].unit_price_minor == 7999
    assert policy.rules == DEFAULT_AFTER_SALES_POLICY_RULES

    policy.source_ref = "LOCAL_POLICY_OVERRIDE"
    session.commit()
    seed(session)
    session.refresh(policy)
    assert policy.source_ref == "LOCAL_POLICY_OVERRIDE"


def test_open_case_is_idempotent_and_keeps_case_separate_from_conversation(session):
    service = AfterSalesCaseService(session)
    first = service.open_case(_command())
    replay = service.open_case(_command())

    assert replay.case_id == first.case_id
    assert first.order_item_id == "OI-1001-1"
    assert first.status == CaseStatus.OPEN
    assert first.revision == 1
    assert first.origin_session_id == "conversation-1"
    assert [event["event_type"] for event in first.events] == ["CASE_OPENED"]

    with pytest.raises(AfterSalesDomainError) as caught:
        service.open_case(_command(reason_text="同一个幂等键下的另一项诉求"))
    assert caught.value.code == "CASE_REQUEST_CONFLICT"


def test_multi_item_order_requires_explicit_line_item(session):
    session.add(
        OrderItem(
            item_id="OI-1001-2",
            order_id=1001,
            line_number=2,
            name="附加配件",
            quantity=1,
            unit_price_minor=1200,
            currency="CNY",
            attributes={},
        )
    )
    session.flush()
    service = AfterSalesCaseService(session)

    with pytest.raises(AfterSalesDomainError) as caught:
        service.open_case(_command())
    assert caught.value.code == "ORDER_ITEM_REQUIRED"

    selected = service.open_case(_command(order_item_id="OI-1001-2"))
    assert selected.order_item_id == "OI-1001-2"


def test_evidence_write_is_versioned_and_idempotent(session):
    service = AfterSalesCaseService(session)
    case = service.open_case(_command())
    changed = service.add_evidence(
        case.case_id,
        _evidence(),
        idempotency_key="evidence-1",
        expected_revision=1,
    )
    replay = service.add_evidence(
        case.case_id,
        _evidence(),
        idempotency_key="evidence-1",
        expected_revision=1,
    )

    assert changed.revision == 2
    assert replay.revision == 2
    assert len(replay.evidence) == 1
    assert [event["sequence"] for event in replay.events] == [1, 2]

    with pytest.raises(AfterSalesDomainError) as conflict:
        service.add_evidence(
            case.case_id,
            _evidence("同一幂等键却更换内容"),
            idempotency_key="evidence-1",
            expected_revision=2,
        )
    assert conflict.value.code == "CASE_IDEMPOTENCY_CONFLICT"

    with pytest.raises(AfterSalesDomainError) as stale:
        service.add_evidence(
            case.case_id,
            EvidenceCommand(
                evidence_type="DAMAGE_PHOTO",
                source_type="CUSTOMER_UPLOAD",
                source_ref="upload:photo-1",
                payload={"object_key": "private/photo-1"},
                observed_at=datetime(2026, 6, 7, 9, 5),
            ),
            idempotency_key="evidence-2",
            expected_revision=1,
        )
    assert stale.value.code == "CASE_REVISION_CONFLICT"
    assert stale.value.retryable is True


def test_policy_decision_requires_evidence_then_becomes_eligible(session):
    service = AfterSalesCaseService(session)
    case = service.open_case(_command())
    engine = AfterSalesPolicyEngine(session)

    first = engine.evaluate(case.case_id, as_of=datetime(2026, 6, 20, 12, 0))
    assert first.outcome == PolicyOutcome.NEEDS_EVIDENCE
    assert first.reason_codes == ("MISSING_EVIDENCE:DAMAGE_DESCRIPTION",)
    assert first.facts["refund_scope"]["alignment"] == "EXACT"
    waiting = service.record_policy_decision(
        case.case_id,
        first,
        idempotency_key="policy-1",
        expected_revision=1,
    )
    assert waiting.status == CaseStatus.AWAITING_EVIDENCE
    assert waiting.revision == 2

    with_evidence = service.add_evidence(
        case.case_id,
        _evidence(),
        idempotency_key="evidence-1",
        expected_revision=2,
    )
    from tests.evidence_fixtures import record_assessment_fixture
    assert engine.evaluate(case.case_id, as_of=datetime(2026, 6, 20, 12)).outcome == PolicyOutcome.NEEDS_EVIDENCE
    record_assessment_fixture(session, with_evidence)
    second = engine.evaluate(case.case_id, as_of=datetime(2026, 6, 20, 12, 0))
    assert second.outcome == PolicyOutcome.ELIGIBLE
    assert second.input_hash != first.input_hash
    ready = service.record_policy_decision(
        case.case_id,
        second,
        idempotency_key="policy-2",
        expected_revision=with_evidence.revision,
    )

    assert ready.status == CaseStatus.AWAITING_CONFIRMATION
    assert ready.latest_decision["outcome"] == "ELIGIBLE"
    assert ready.latest_decision["policy_version_id"] == DEFAULT_AFTER_SALES_POLICY_VERSION_ID
    assert ready.latest_decision["evidence_refs"][-1].startswith("after_sales_evidence:")
    assert [event["sequence"] for event in ready.events] == [1, 2, 3, 4]
    decisions = session.scalars(
        select(PolicyDecision).where(PolicyDecision.case_id == case.case_id).order_by(PolicyDecision.decided_at)
    ).all()
    assert len(decisions) == 2
    assert decisions[1].supersedes_decision_id == decisions[0].decision_id

    replay = service.record_policy_decision(
        case.case_id,
        second,
        idempotency_key="policy-2",
        expected_revision=3,
    )
    assert replay.revision == 4


def test_policy_missing_or_unknown_reason_fails_closed(session):
    service = AfterSalesCaseService(session)
    missing_case = service.open_case(_command(request_key="missing-reason", reason_code=None))
    missing = AfterSalesPolicyEngine(session).evaluate(
        missing_case.case_id,
        as_of=datetime(2026, 6, 20, 12, 0),
    )

    assert missing.outcome == PolicyOutcome.NEEDS_EVIDENCE
    assert missing.reason_codes == ("AFTER_SALES_REASON_REQUIRED",)
    assert "CLARIFY_REASON" in missing.allowed_actions

    unknown_case = service.open_case(_command(request_key="unknown-reason", reason_code="OTHER"))
    unknown = AfterSalesPolicyEngine(session).evaluate(
        unknown_case.case_id,
        as_of=datetime(2026, 6, 20, 12, 0),
    )

    assert unknown.outcome == PolicyOutcome.MANUAL_REVIEW
    assert unknown.reason_codes == ("AFTER_SALES_REASON_NOT_SUPPORTED",)


def test_line_item_case_cannot_authorize_order_level_refund_for_multi_item_order(session):
    session.add(
        OrderItem(
            item_id="OI-1001-2",
            order_id=1001,
            line_number=2,
            name="附加配件",
            quantity=1,
            unit_price_minor=1200,
            currency="CNY",
            attributes={},
        )
    )
    session.get(Order, 1001).amount = 91.99
    session.scalar(select(Payment).where(Payment.order_id == 1001)).amount = 91.99
    session.flush()
    service = AfterSalesCaseService(session)
    case = service.open_case(
        _command(
            request_key="multi-item-scope",
            order_item_id="OI-1001-1",
            reason_code="NOT_AS_DESCRIBED",
        )
    )
    case = service.add_evidence(
        case.case_id,
        EvidenceCommand(
            evidence_type="ISSUE_DESCRIPTION",
            source_type="CUSTOMER_STATEMENT",
            source_ref="conversation-1:multi-item-description",
            payload={"description": "耳机功能与说明不一致"},
            observed_at=datetime(2026, 6, 20, 9, 0),
        ),
        idempotency_key="multi-item-evidence",
        expected_revision=case.revision,
    )

    decision = AfterSalesPolicyEngine(session).evaluate(
        case.case_id,
        as_of=datetime(2026, 6, 20, 12, 0),
    )

    assert decision.outcome == PolicyOutcome.MANUAL_REVIEW
    assert decision.reason_codes == ("REFUND_SCOPE_REQUIRES_LINE_ITEM_EXECUTION",)
    assert decision.facts["refund_scope"] == {
        "requested_scope": {
            "type": "ORDER_ITEM",
            "order_item_id": "OI-1001-1",
            "amount_minor": 7999,
        },
        "execution_scope": {"type": "ORDER", "order_id": 1001, "amount_minor": 9199},
        "alignment": "REQUIRES_SCOPED_REFUND",
        "mismatch_reasons": [
            "ORDER_HAS_MULTIPLE_LINE_ITEMS",
            "ORDER_AND_LINE_ITEM_AMOUNTS_DIFFER",
        ],
        "order_item_count": 2,
        "order_amount_minor": 9199,
        "payment_amount_minor": 9199,
    }
    assert "ESCALATE_TO_HUMAN" in decision.allowed_actions


def test_policy_expiration_and_existing_refund_are_machine_readable(session):
    service = AfterSalesCaseService(session)
    expired_case = service.open_case(_command(request_key="expired", reason_code=None))
    expired = AfterSalesPolicyEngine(session).evaluate(
        expired_case.case_id,
        as_of=datetime(2026, 8, 1),
    )
    assert expired.outcome == PolicyOutcome.INELIGIBLE
    assert expired.reason_codes == ("RETURN_WINDOW_EXPIRED",)

    requested_case = service.open_case(
        _command(request_key="already-requested", order_id=1004, reason_code=None)
    )
    requested = AfterSalesPolicyEngine(session).evaluate(
        requested_case.case_id,
        as_of=datetime(2026, 5, 25),
    )
    assert requested.outcome == PolicyOutcome.INELIGIBLE
    assert requested.reason_codes == ("REFUND_ALREADY_EXISTS",)


def test_high_value_case_is_sent_to_manual_review_after_hard_checks(session):
    refund = session.scalar(select(Refund).where(Refund.order_id == 1004))
    refund.status = "none"
    refund.reason = None
    session.flush()
    service = AfterSalesCaseService(session)
    case = service.open_case(
        _command(request_key="high-value", order_id=1004, reason_code="NOT_AS_DESCRIBED")
    )
    case = service.add_evidence(
        case.case_id,
        EvidenceCommand(
            evidence_type="ISSUE_DESCRIPTION",
            source_type="CUSTOMER_STATEMENT",
            source_ref="agent_runs:evidence-run:request",
            payload={"statement": "实际功能与商品说明不一致"},
            observed_at=datetime(2026, 5, 25, 9, 0),
        ),
        idempotency_key="high-value-evidence",
        expected_revision=case.revision,
    )

    from tests.evidence_fixtures import record_assessment_fixture
    record_assessment_fixture(session, case, now=datetime(2026, 5, 25, 12))
    decision = AfterSalesPolicyEngine(session).evaluate(case.case_id, as_of=datetime(2026, 5, 25, 12, 0))

    assert decision.outcome == PolicyOutcome.MANUAL_REVIEW
    assert decision.reason_codes == ("AMOUNT_REQUIRES_MANUAL_REVIEW",)


def test_policy_integrity_and_overlap_fail_closed(session):
    service = AfterSalesCaseService(session)
    case = service.open_case(_command())
    policy = session.get(PolicyVersion, DEFAULT_AFTER_SALES_POLICY_VERSION_ID)
    policy.rules = {**policy.rules, "return_window_days": 999}
    session.flush()

    with pytest.raises(AfterSalesDomainError) as corrupt:
        AfterSalesPolicyEngine(session).evaluate(case.case_id, as_of=datetime(2026, 6, 20))
    assert corrupt.value.code == "POLICY_INTEGRITY_ERROR"

    session.rollback()
    case = service.open_case(_command(request_key="overlap"))
    base = session.get(PolicyVersion, DEFAULT_AFTER_SALES_POLICY_VERSION_ID)
    session.add(
        PolicyVersion(
            policy_version_id="alternate-return-refund:2026-01",
            policy_id="alternate-return-refund",
            version="2026-01",
            policy_type="AFTER_SALES",
            status="ACTIVE",
            effective_from=datetime(2026, 1, 1),
            rules=base.rules,
            source_ref="TEST_POLICY",
            checksum=base.checksum,
            created_at=datetime(2026, 1, 1),
        )
    )
    session.flush()

    with pytest.raises(AfterSalesDomainError) as ambiguous:
        AfterSalesPolicyEngine(session).evaluate(case.case_id, as_of=datetime(2026, 6, 20))
    assert ambiguous.value.code == "POLICY_VERSION_AMBIGUOUS"


def test_terminal_case_rejects_late_writes(session):
    service = AfterSalesCaseService(session)
    case = service.open_case(_command())
    cancelled = service.transition(
        case.case_id,
        CaseStatus.CANCELLED,
        event_type="CASE_CANCELLED",
        idempotency_key="cancel-1",
        expected_revision=1,
        actor_type=ActorType.CUSTOMER,
    )
    assert cancelled.closed_at is not None

    with pytest.raises(AfterSalesDomainError) as terminal:
        service.add_evidence(
            case.case_id,
            _evidence(),
            idempotency_key="late-evidence",
            expected_revision=2,
        )
    assert terminal.value.code == "CASE_TERMINAL"

    events = session.scalars(
        select(AfterSalesCaseEvent)
        .where(AfterSalesCaseEvent.case_id == case.case_id)
        .order_by(AfterSalesCaseEvent.sequence)
    ).all()
    assert [(event.sequence, event.case_revision) for event in events] == [(1, 1), (2, 2)]
