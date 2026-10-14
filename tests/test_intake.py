import pytest

from src.intake import (
    APPROVAL_CONFIRMED,
    APPROVAL_UNKNOWN,
    CLARIFICATION_REQUIRED,
    DIRECT,
    ESCALATION_REQUESTED,
    FAQ,
    LLM_REQUIRED,
    ORDER_DETAILS,
    ORDER_STATUS,
    PAYMENT_PROCESS,
    PAYMENT_STATUS,
    REFUND_ELIGIBILITY,
    REFUND_REQUEST,
    SHIPMENT_TRACKING,
    UNSUPPORTED,
    analyze_request,
    parse_approval_response,
)


@pytest.mark.parametrize(
    "message,mode,intents,order_id",
    [
        ("请查询订单 1004 的当前状态", DIRECT, (ORDER_STATUS,), 1004),
        ("追踪订单 1002 的物流进度", DIRECT, (SHIPMENT_TRACKING,), 1002),
        ("Where is order #1002?", DIRECT, (SHIPMENT_TRACKING,), 1002),
        ("查询订单 1001 是否可以退款", DIRECT, (REFUND_ELIGIBILITY,), 1001),
        ("订单 1001 商品损坏了，我想退款", DIRECT, (REFUND_REQUEST,), 1001),
        ("帮我支付订单 1003", DIRECT, (PAYMENT_PROCESS,), 1003),
        ("查询订单 1003 的支付状态", DIRECT, (PAYMENT_STATUS,), 1003),
        ("Has payment for order 1003 gone through?", DIRECT, (PAYMENT_STATUS,), 1003),
        ("查询订单 1004 的详情", DIRECT, (ORDER_DETAILS,), 1004),
        ("请告诉我退款规则", DIRECT, (FAQ,), None),
        ("What's your return policy?", DIRECT, (FAQ,), None),
    ],
)
def test_deterministic_intake(message, mode, intents, order_id):
    result = analyze_request(message)
    assert result.execution_mode == mode
    assert result.intents == intents
    assert result.entities.get("order_id") == order_id


def test_compound_intent_keeps_all_required_tasks():
    result = analyze_request("查询订单 1004 的状态和物流进度，再看看能不能退款")
    assert result.execution_mode == DIRECT
    assert set(result.intents) == {ORDER_STATUS, SHIPMENT_TRACKING, REFUND_ELIGIBILITY}


@pytest.mark.parametrize(
    "message",
    [
        "查询订单但未提供订单号",
        "帮我看看物流",
        "这个订单能不能退款",
        "查询支付状态",
    ],
)
def test_missing_order_id_requests_clarification(message):
    result = analyze_request(message)
    assert result.execution_mode == CLARIFICATION_REQUIRED
    assert result.missing_slots == ("order_id",)


def test_context_note_supplies_last_order_id():
    result = analyze_request("(Context: the customer was last discussing order 1007.)\n查一下刚才那个订单")
    assert result.execution_mode == DIRECT
    assert result.entities["order_id"] == 1007
    assert result.intents == (ORDER_DETAILS,)


def test_explicit_human_request_is_not_confused_with_failure():
    result = analyze_request("请直接帮我转人工客服")
    assert result.execution_mode == ESCALATION_REQUESTED


def test_out_of_domain_request_is_unsupported():
    result = analyze_request("帮我查一下今天的股票价格")
    assert result.execution_mode == UNSUPPORTED


def test_ambiguous_request_uses_model_fallback():
    result = analyze_request("你是谁")
    assert result.execution_mode == LLM_REQUIRED


@pytest.mark.parametrize(
    "message",
    ["我还没确认", "帮我确认一下订单 1003 的物流", "取消订单有什么政策", "not confirmed yet"],
)
def test_approval_requires_an_explicit_standalone_confirmation(message):
    pending = {"type": "PROCESS_PAYMENT", "order_id": 1003}
    assert parse_approval_response(message, pending) == APPROVAL_UNKNOWN


def test_explicit_confirmation_still_matches_pending_action():
    pending = {"type": "PROCESS_PAYMENT", "order_id": 1003}
    assert parse_approval_response("确认使用 paypal 支付", pending) == APPROVAL_CONFIRMED
