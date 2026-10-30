import json

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import select

from src import tools
from src import orchestrator
from src.agents import base
from src.agents.payment_agent import PAYMENT_TOOLS
from src.agents.refund_agent import REFUND_TOOLS
from src.backend.catalog import seed_catalog
from src.backend.db import Order, Payment, Product, Refund, get_engine, get_session_factory
from src.backend.seed import seed
from src.intake import LLM_REQUIRED, PRODUCT_SEARCH, analyze_request
from src.orchestrator import run_orchestrator
from src.persistence import RunStore
from src.workspace import list_orders, list_runs, save_run
from tests.test_agent_base import FakeLLM, FakeToolCallMessage


@pytest.fixture(autouse=True)
def seeded_workspace():
    engine = get_engine("sqlite:///:memory:")
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
    yield engine


def test_catalog_seed_is_additive_and_keeps_edited_stock(seeded_workspace):
    with get_session_factory(seeded_workspace)() as session:
        product = session.get(Product, "P1101")
        product.stock = 0
        product.price = 88
        session.commit()
        seed_catalog(session)
        session.commit()
        assert len(session.scalars(select(Product)).all()) == 60
        assert product.stock == 0
        assert product.price == 88


def test_catalog_metadata_observes_database_change_after_explicit_invalidation(seeded_workspace):
    provider = tools.get_catalog_metadata_provider()
    first = provider.get_snapshot()
    with get_session_factory(seeded_workspace)() as session:
        session.add(Product(
            product_id="PX001",
            name="澄光显示器",
            category="显示器",
            price=1299,
            summary="27 英寸 · 4K",
            specs={"分辨率": "4K"},
            stock=3,
        ))
        session.commit()

    assert provider.get_snapshot() is first
    assert "显示器" not in first.facets["categories"]

    provider.invalidate()
    changed = provider.get_snapshot()

    assert "显示器" in changed.facets["categories"]
    assert changed.fingerprint != first.fingerprint


def test_business_seed_is_additive_and_preserves_changed_state(seeded_workspace):
    with get_session_factory(seeded_workspace)() as session:
        order = session.get(Order, 1003)
        payment = session.scalar(select(Payment).where(Payment.order_id == 1003))
        refund = session.scalar(select(Refund).where(Refund.order_id == 1001))
        order.status = "shipped"
        payment.status = "paid"
        payment.method = "paypal"
        refund.status = "requested"
        refund.reason = "用户已确认的原因"
        session.commit()

        seed(session)

        session.refresh(order)
        session.refresh(payment)
        session.refresh(refund)
        assert order.status == "shipped"
        assert (payment.status, payment.method) == ("paid", "paypal")
        assert (refund.status, refund.reason) == ("requested", "用户已确认的原因")
        assert len(session.scalars(select(Order)).all()) == 10
        assert len(session.scalars(select(Payment)).all()) == 10
        assert len(session.scalars(select(Refund)).all()) == 10


def test_search_enforces_budget_category_features_and_availability(seeded_workspace):
    with get_session_factory(seeded_workspace)() as session:
        session.get(Product, "P1102").stock = 0
        session.commit()
    result = tools.search_products(query="降噪 通勤", category="耳机", max_price=500)
    assert result["products"]
    assert all(p["price"] <= 500 and p["category"] == "耳机" and p["stock"] > 0 for p in result["products"])
    assert "P1102" not in [p["product_id"] for p in result["products"]]
    assert tools.search_products(category="耳机", max_price=1)["products"] == []

    compact = tools.search_products(query="降噪耳机", category="耳机", max_price=500)
    assert compact["products"]
    assert compact["filters"]["query_terms"] == ["降噪", "耳机"]
    assert all(product["specs"]["降噪"] != "无" for product in compact["products"])

    discovered = tools.discover_products(
        query="降噪耳机", category="耳机", max_price=500, limit=1, context_seed="test"
    )
    assert discovered["products"][0]["specs"]["降噪"] != "无"
    assert "unbound_catalog_language" not in discovered


def test_invalid_budget_and_comparison_are_explicit():
    assert "error" in tools.search_products(max_price=-1)
    assert "error" in tools.compare_products(["P1101", "P1101"])
    assert "error" in tools.compare_products(["P1101", "P9999"])
    result = tools.compare_products(["P1101", "P1102"])
    assert [p["product_id"] for p in result["products"]] == ["P1101", "P1102"]
    assert result["products"][0]["specs"]["降噪"] != result["products"][1]["specs"]["降噪"]


def test_shopping_ignores_previous_order_context_and_retains_budget():
    intake = analyze_request("(Context: the customer was last discussing order 1002.)\n推荐通勤降噪耳机，预算 500 元以内")
    assert intake.execution_mode == LLM_REQUIRED
    assert intake.intents == ()
    assert intake.route_hints == (PRODUCT_SEARCH,)
    assert intake.entities["max_price"] == 500
    assert "order_id" not in intake.entities


def test_shopping_request_uses_semantic_contract_before_catalog_execution(monkeypatch):
    message = "推荐通勤降噪耳机，预算 500 元以内"

    class Model:
        def json_mode(self):
            return self

        def invoke(self, _prompt):
            return AIMessage(content=json.dumps({
                "reason": "需要保留商品目标、用途、属性和预算范围",
                "commands": [],
                "objectives": [{
                    "id": "headphones",
                    "kind": PRODUCT_SEARCH,
                    "source_quote": message,
                    "category": "耳机",
                    "query": "通勤 降噪",
                    "max_price": 500,
                }],
            }, ensure_ascii=False))

    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: Model())

    result = run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["agent"] == "shopping"
    assert result["budget"]["consumed"]["model_calls"] == 1
    assert result["trace"][0]["tool"] == "search_products"
    assert all(p["price"] <= 500 for p in result["trace"][0]["result"]["products"])


def test_product_comparison_uses_catalog_records():
    result = run_orchestrator("对比商品 P1101 和 P1102")
    assert result["trace"][0]["tool"] == "compare_products"
    assert result["status"] == "SUCCEEDED"
    assert "P1102" in result["reply"]


def test_order_filters_match_chinese_and_payment_updates():
    assert list_orders(query="键盘")[0]["order_id"] == 1002
    assert all(order["status"] == "shipped" for order in list_orders(status="shipped"))
    tools.process_payment(1003, "card")
    assert list_orders(query="1003")[0]["payment_status"] == "paid"


def test_shipment_events_are_ordered_and_match_terminal_state():
    events = tools.track_shipment(1004)["events"]
    assert events[-1]["description"] == "包裹已签收"
    assert [e["timestamp"] for e in events] == sorted(e["timestamp"] for e in events)
    cancelled = tools.track_shipment(1005)["events"]
    assert cancelled[-1]["description"] == "订单已取消"
    assert not any("发货" in e["description"] for e in cancelled)


def test_run_archive_survives_new_sessions_and_deduplicates():
    result = run_orchestrator("查询订单 1002 的物流")
    save_run("conversation-a", "查询订单 1002 的物流", result)
    save_run("conversation-a", "重复保存", result)
    runs = list_runs()
    assert len(runs) == 1
    assert runs[0]["request"] == "查询订单 1002 的物流"
    assert runs[0]["result"]["events"] == result["events"]
    assert runs[0]["result"]["trace"] == result["trace"]


def test_active_run_archive_uses_persisted_events_before_terminal_result():
    store = RunStore()
    claim = store.claim("live-inspector", "request", "查询订单", {})
    event = {"run_id": claim["run_id"], "sequence": 1, "timestamp": "2026-01-01T00:00:00Z",
             "type": "model_call", "purpose": "router"}
    store.save_checkpoint(claim["run_id"], claim["token"], claim["snapshot"], event=event)

    saved = next(run for run in list_runs() if run["run_id"] == claim["run_id"])
    assert saved["result"]["status"] == "RUNNING"
    assert saved["result"]["events"] == [event]
    assert saved["result"]["terminal_reason"] is None


def test_model_path_cannot_execute_unconfirmed_payment(monkeypatch):
    response = FakeToolCallMessage(tool_calls=[{"name": "process_payment", "args": {"order_id": 1003, "method": "card"}, "id": "1"}])
    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: FakeLLM([response]))
    result = base.run_tool_agent("payment", "system", PAYMENT_TOOLS, "处理这个付款")
    assert result["status"] == "WAITING_APPROVAL"
    assert result["pending_action"]["order_id"] == 1003
    assert tools.get_payment_status(1003)["status"] == "pending"
    assert not any(t["tool"] == "process_payment" for t in result["trace"])


def test_model_path_cannot_execute_unconfirmed_refund(monkeypatch, seeded_workspace):
    response = FakeToolCallMessage(
        tool_calls=[
            {"name": "initiate_refund", "args": {"order_id": 1001, "reason": "商品损坏"}, "id": "1"}
        ]
    )
    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: FakeLLM([response]))
    result = base.run_tool_agent("refund", "system", REFUND_TOOLS, "处理这个退款")
    with get_session_factory(seeded_workspace)() as session:
        refund = session.scalar(select(Refund).where(Refund.order_id == 1001))
        assert refund.status == "none"
    assert result["status"] == "WAITING_APPROVAL"
    assert result["pending_action"]["order_id"] == 1001
    assert not any(t["tool"] == "initiate_refund" for t in result["trace"])


def test_semantic_intake_carries_model_requested_approval_to_frontend(monkeypatch):
    import json
    from langchain_core.messages import AIMessage
    class Model:
        def invoke(self, _prompt):
            return AIMessage(content=json.dumps({"reason": "payment request", "commands": [], "objectives": [
                {"id": "payment", "kind": "PAYMENT_PROCESS",
                 "source_quote": "take care of the bill for order 1003", "order_id": 1003}
            ]}))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: Model())
    result = run_orchestrator("could you take care of the bill for order 1003")
    assert result["status"] == "WAITING_APPROVAL"
    assert result["pending_action"]["order_id"] == 1003
    assert tools.get_payment_status(1003)["status"] == "pending"


def test_read_only_browsing_keeps_existing_pending_approval():
    pending = run_orchestrator("帮我支付订单 1003")["pending_action"]
    result = run_orchestrator("查询订单 1002 的物流", pending_action=pending)
    assert result["pending_action"] == pending
    assert tools.get_payment_status(1003)["status"] == "pending"


def test_zero_tool_budget_cannot_propose_mutation(monkeypatch):
    response = FakeToolCallMessage(tool_calls=[{"name": "process_payment", "args": {"order_id": 1003, "method": "card"}, "id": "1"}])
    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: FakeLLM([response]))
    result = base.run_tool_agent("payment", "system", PAYMENT_TOOLS, "付款", max_tool_calls=0)
    assert result["terminal_reason"] == "TOOL_BUDGET_EXHAUSTED"
    assert result.get("pending_action") is None
    assert tools.get_payment_status(1003)["status"] == "pending"
