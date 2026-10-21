import json
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import api
from src import orchestrator, tools
from src.backend.db import Payment, Refund, get_engine, get_session_factory
from src.backend.seed import seed


@pytest.fixture(autouse=True)
def seeded_db():
    engine = get_engine("sqlite:///:memory:")
    tools.configure_db(engine)
    session = get_session_factory(engine)()
    seed(session)
    yield session


def test_chinese_order_1004_status_and_tracking_succeeds_without_llm(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM must not run")))

    result = orchestrator.run_orchestrator("请帮我查询订单 1004 的状态和物流进度")

    assert result["status"] == "SUCCEEDED"
    assert result["route"] == "DIRECT"
    assert result["agent"] == "order"
    assert [entry["tool"] for entry in result["trace"]] == ["get_order_status", "track_shipment"]
    assert "已送达" in result["reply"]
    assert result["usage"]["total_tokens"] == 0


def test_compound_order_and_refund_check_executes_safe_plan():
    result = orchestrator.run_orchestrator("查询订单 1001 的物流，再判断能不能退款")
    tools_called = [entry["tool"] for entry in result["trace"]]
    assert result["status"] == "SUCCEEDED"
    assert tools_called == ["track_shipment", "get_order_status", "check_refund_eligibility"]
    assert "符合退款条件" in result["reply"]


def test_refund_eligibility_check_does_not_mutate_refund(seeded_db):
    result = orchestrator.run_orchestrator("查询订单 1001 是否可以退款")
    refund = seeded_db.query(Refund).filter_by(order_id=1001).one()
    assert result["status"] == "SUCCEEDED"
    assert refund.status == "none"
    assert "initiate_refund" not in [entry["tool"] for entry in result["trace"]]


def test_refund_request_waits_for_approval_when_eligible(seeded_db):
    result = orchestrator.run_orchestrator("订单 1001 商品有问题，我想退款")
    refund = seeded_db.query(Refund).filter_by(order_id=1001).one()
    assert result["status"] == "WAITING_APPROVAL"
    assert result["pending_action"]["type"] == "INITIATE_REFUND"
    assert refund.status == "none"


def test_payment_request_waits_for_approval_without_charging(seeded_db):
    result = orchestrator.run_orchestrator("帮我支付订单 1003")
    payment = seeded_db.query(Payment).filter_by(order_id=1003).one()
    assert result["status"] == "WAITING_APPROVAL"
    assert result["terminal_reason"] == "APPROVAL_REQUIRED"
    assert result["pending_action"]["type"] == "PROCESS_PAYMENT"
    assert payment.status == "pending"
    assert "process_payment" not in [entry["tool"] for entry in result["trace"]]


def test_confirmed_payment_executes_once(seeded_db):
    first = orchestrator.run_orchestrator("帮我支付订单 1003")
    second = orchestrator.run_orchestrator("确认", pending_action=first["pending_action"])
    payment = seeded_db.query(Payment).filter_by(order_id=1003).one()

    assert second["status"] == "SUCCEEDED"
    assert second["trace"][0]["tool"] == "process_payment"
    assert payment.status == "paid"
    assert second["pending_action"] is None


def test_confirmed_refund_executes_after_approval(seeded_db):
    first = orchestrator.run_orchestrator("订单 1001 商品损坏了，我想退款")
    second = orchestrator.run_orchestrator("确认继续", pending_action=first["pending_action"])
    refund = seeded_db.query(Refund).filter_by(order_id=1001).one()

    assert first["status"] == "WAITING_APPROVAL"
    assert second["status"] == "SUCCEEDED"
    assert second["trace"][0]["tool"] == "initiate_refund"
    assert refund.status == "requested"


def test_cancelled_pending_action_makes_no_change(seeded_db):
    first = orchestrator.run_orchestrator("帮我支付订单 1003")
    second = orchestrator.run_orchestrator("取消，先不要支付", pending_action=first["pending_action"])
    payment = seeded_db.query(Payment).filter_by(order_id=1003).one()

    assert second["status"] == "CANCELLED"
    assert second["trace"] == []
    assert payment.status == "pending"


def test_missing_order_id_clarifies_instead_of_escalating():
    result = orchestrator.run_orchestrator("查询订单状态")
    assert result["status"] == "WAITING_CLARIFICATION"
    assert result["terminal_reason"] == "MISSING_ORDER_ID"
    assert result["agent"] == "intake"
    assert result["trace"] == []


def test_explicit_human_request_creates_a_traceable_ticket():
    result = orchestrator.run_orchestrator("请直接转人工客服")
    assert result["status"] == "WAITING_HUMAN"
    assert result["terminal_reason"] == "USER_REQUESTED_ESCALATION"
    assert result["trace"][0]["tool"] == "escalate_to_human"
    assert result["trace"][0]["result"]["ticket_id"]


def test_model_unavailable_is_failed_not_fake_escalation(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "get_llm",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("No LLM configured. Set a key")),
    )
    result = orchestrator.run_orchestrator("帮我看看这个")
    assert result["status"] == "FAILED"
    assert result["terminal_reason"] == "MODEL_UNAVAILABLE"
    assert result["agent"] == "runtime"
    assert result["trace"] == []
    assert "没有转交人工" in result["reply"]


@pytest.mark.parametrize(
    "message,expected",
    [
        ("你是谁", "我是 LingCart"),
        ("你能做什么？", "查订单与物流"),
        ("你好", "你好，我是 LingCart"),
    ],
)
def test_basic_conversation_succeeds_without_model_or_tools(monkeypatch, message, expected):
    kinds = {"你是谁": "IDENTITY", "你能做什么？": "CAPABILITIES", "你好": "GREETING"}
    response = AIMessage(content=json.dumps({
        "reason": "基础对话",
        "commands": [{"kind": kinds[message], "source_quote": message, "message": None}],
        "objectives": [],
    }, ensure_ascii=False))
    class Model:
        def invoke(self, _prompt):
            return response
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: Model())

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["route"] == "CONVERSATIONAL"
    assert result["agent"] == "intake"
    assert result["trace"] == []
    assert result["budget"]["consumed"]["model_calls"] == 1
    assert expected in result["reply"]


def test_unsupported_request_is_not_escalated(monkeypatch):
    response = AIMessage(content=json.dumps({
        "reason": "超出消费服务范围",
        "commands": [{"kind": "UNSUPPORTED", "source_quote": "帮我查一下今天的天气", "message": None}],
        "objectives": [],
    }, ensure_ascii=False))
    class Model:
        def invoke(self, _prompt):
            return response
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: Model())
    result = orchestrator.run_orchestrator("帮我查一下今天的天气")
    assert result["status"] == "FAILED"
    assert result["terminal_reason"] == "UNSUPPORTED_INTENT"
    assert result["route"] == "UNSUPPORTED"


def test_shared_budget_bypasses_single_item_fast_path_and_enters_semantic_layer(monkeypatch):
    response = AIMessage(content=json.dumps({
        "reason": "需要保留多个目标和共享约束",
        "commands": [{"kind": "CLARIFY", "source_quote": "有什么推荐吗", "message": "请确认是否各买一件。"}],
        "objectives": [],
    }, ensure_ascii=False))

    class Model:
        def invoke(self, _prompt):
            return response

    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: Model())

    result = orchestrator.run_orchestrator("我有400元预算，想买一个耳机一个键盘，有什么推荐吗")

    assert result["status"] == "WAITING_CLARIFICATION"
    assert result["terminal_reason"] == "MODEL_CLARIFICATION"
    assert result["route"] == "CLARIFY"
    assert result["trace"] == []
    assert result["budget"]["consumed"]["model_calls"] == 1


def test_every_event_is_ordered_and_linked_to_run():
    result = orchestrator.run_orchestrator("查询订单 1002 的状态和物流")
    assert result["run_id"]
    assert [event["sequence"] for event in result["events"]] == list(range(1, len(result["events"]) + 1))
    assert all(event["run_id"] == result["run_id"] for event in result["events"])
    assert result["events"][-1]["type"] == "terminal"


def test_api_exposes_v1_run_status_and_events():
    response = TestClient(api.app).post("/chat", json={"message": "请查询订单 1004 的状态和物流"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCEEDED"
    assert body["terminal_reason"] == "COMPLETED"
    assert body["run_id"]
    assert body["intake"]["execution_mode"] == "DIRECT"
    assert [item["tool"] for item in body["trace"]] == ["get_order_status", "track_shipment"]
    assert body["events"][-1]["type"] == "terminal"


def test_api_session_carries_pending_action_into_confirmation(seeded_db):
    client = TestClient(api.app)
    first = client.post(
        "/chat",
        json={"message": "帮我支付订单 1003", "session_id": "approval-session"},
    )
    second = client.post(
        "/chat",
        json={"message": "确认", "session_id": "approval-session"},
    )
    payment = seeded_db.query(Payment).filter_by(order_id=1003).one()

    assert first.json()["status"] == "WAITING_APPROVAL"
    assert second.json()["status"] == "SUCCEEDED"
    assert second.json()["trace"][0]["tool"] == "process_payment"
    assert payment.status == "paid"
