import pytest
from pydantic import ValidationError

from src import tools
from src.backend.db import get_engine, get_session_factory
from src.backend.seed import seed
from src.runtime import tool_spec


@pytest.fixture(autouse=True)
def seeded_db():
    engine = get_engine("sqlite:///:memory:")
    tools.configure_db(engine)
    session = get_session_factory(engine)()
    seed(session)
    yield


def test_get_order_status_found():
    assert tools.get_order_status(1001) == {"order_id": 1001, "status": "delivered"}


def test_runtime_clock_has_explicit_source_and_timezone():
    result = tools.read_clock("Asia/Shanghai")
    assert result["data_source"] == "RUNTIME_CLOCK"
    assert result["timezone"] == "Asia/Shanghai"
    assert 1 <= result["weekday"] <= 7


def test_calculator_accepts_arithmetic_and_rejects_code():
    assert tools.calculate("(18 + 6) / 3")["result"] == 8
    assert "error" in tools.calculate("__import__('os').getcwd()")
    assert "error" in tools.calculate("2 ** 100")
    assert "error" in tools.calculate("9" * 180)


def test_discovery_limit_is_enforced_by_tool_schema_before_backend_execution():
    schema = tool_spec("discover_products", tools.discover_products).schema

    assert schema.model_validate({"limit": 12}).limit == 12
    with pytest.raises(ValidationError):
        schema.model_validate({"limit": 30})


def test_get_order_status_not_found():
    result = tools.get_order_status(9999)
    assert "error" in result


def test_get_order_details():
    result = tools.get_order_details(1002)
    assert result["customer_name"] == "Ben Torres"
    assert result["status"] == "shipped"


def test_track_shipment_delivered():
    result = tools.track_shipment(1001)
    assert result["tracking_message"] == "Delivered."


def test_refund_eligibility_delivered_no_existing_refund():
    result = tools.check_refund_eligibility(1001)
    assert result["eligible"] is True


def test_refund_eligibility_not_delivered():
    result = tools.check_refund_eligibility(1002)  # shipped
    assert result["eligible"] is False
    assert "delivered" in result["reason"]


def test_refund_eligibility_already_requested():
    result = tools.check_refund_eligibility(1004)  # delivered but refund already requested
    assert result["eligible"] is False
    assert "already" in result["reason"]


def test_initiate_refund_succeeds_when_eligible():
    result = tools.initiate_refund(1001, "Item not as described")
    assert result["success"] is True
    assert result["status"] == "requested"
    # DB actually changed
    assert tools.check_refund_eligibility(1001)["eligible"] is False


def test_initiate_refund_refuses_when_not_delivered():
    result = tools.initiate_refund(1002, "Changed my mind")
    assert result["success"] is False
    assert "delivered" in result["reason"]


def test_get_payment_status():
    result = tools.get_payment_status(1003)
    assert result["status"] == "pending"


def test_process_payment_succeeds_when_pending():
    result = tools.process_payment(1003, "paypal")
    assert result["success"] is True
    assert tools.get_payment_status(1003)["status"] == "paid"


def test_process_payment_refuses_when_already_paid():
    result = tools.process_payment(1001, "card")
    assert result["success"] is False


def test_answer_faq_shipping():
    answer = tools.answer_faq("How long does shipping take?")
    assert answer['status'] == 'ANSWERED'
    assert "3-5 business days" in answer['answer']
    assert answer['policies'][0]['policy_id'] == 'shipping'


def test_answer_faq_fallback():
    answer = tools.answer_faq("What is the meaning of life?")
    assert answer['status'] == 'NOT_FOUND'
    assert not answer['policies']


def test_escalate_to_human():
    result = tools.escalate_to_human("agent could not resolve", "user asked about order 9999")
    assert result['escalated'] is False and result['delivered'] is False
    assert result['mode'] == 'DEMO'
    assert "ticket_id" in result
