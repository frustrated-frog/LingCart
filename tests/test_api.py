from fastapi.testclient import TestClient
import pytest

import api
from src import tools
from src.backend.db import get_engine, get_session_factory
from src.backend.seed import seed


@pytest.fixture(autouse=True)
def isolated_database():
    engine = get_engine("sqlite:///:memory:")
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)

client = TestClient(api.app)

FAKE_RESULT = {
    "reply": "Your order 1001 has been delivered.",
    "agent": "order",
    "trace": [
        {
            "agent": "order",
            "tool": "get_order_status",
            "args": {"order_id": 1001},
            "result": {"order_id": 1001, "status": "delivered"},
        }
    ],
    "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    "cached": False,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_status_is_sanitized_and_does_not_probe_provider(monkeypatch):
    calls = {"count": 0}

    def fake_status():
        calls["count"] += 1
        return {
            "ready": True,
            "configured": True,
            "live_verified": False,
            "message": "模型配置已加载，尚未执行实时请求",
            "purpose": "default",
            "max_attempts": 2,
            "request_timeout": 10,
            "providers": [{"provider": "openai", "model": "gpt-test", "slot": 1, "custom_base_url": False}],
            "active": None,
            "last_error": None,
        }

    monkeypatch.setattr(api, "get_model_status", fake_status)
    response = client.get("/models/status")

    assert response.status_code == 200
    assert response.json()["providers"][0]["model"] == "gpt-test"
    assert "api_key" not in response.text.lower()
    assert calls["count"] == 1


def test_readiness_checks_database_and_model_configuration(monkeypatch):
    monkeypatch.setattr(api.tools, "get_database_status", lambda: {
        "ready": True, "message": "数据库连接正常", "latency_ms": 1.2,
    })
    monkeypatch.setattr(api, "get_model_status", lambda: {
        "ready": True, "configured": True, "message": "模型配置已加载",
    })

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["checks"]["database"]["latency_ms"] == 1.2


def test_readiness_returns_503_when_model_is_not_configured(monkeypatch):
    monkeypatch.setattr(api.tools, "get_database_status", lambda: {
        "ready": True, "message": "数据库连接正常", "latency_ms": 1.2,
    })
    monkeypatch.setattr(api, "get_model_status", lambda: {
        "ready": False, "configured": False, "message": "未配置可用模型",
    })

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"]["model_gateway"]["ready"] is False


def test_chat_returns_agent_reply_and_trace(monkeypatch):
    monkeypatch.setattr(api, "run_orchestrator", lambda *_a, **_k: FAKE_RESULT)

    response = client.post("/chat", json={"message": "where is order 1001?"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Your order 1001 has been delivered."
    assert body["agent"] == "order"
    assert body["trace"][0]["tool"] == "get_order_status"
    assert body["usage"]["total_tokens"] == 15
    assert "session_id" in body


def test_chat_generates_new_session_id_when_omitted(monkeypatch):
    monkeypatch.setattr(api, "run_orchestrator", lambda *_a, **_k: FAKE_RESULT)

    r1 = client.post("/chat", json={"message": "hello"})
    r2 = client.post("/chat", json={"message": "hello again"})

    assert r1.json()["session_id"] != r2.json()["session_id"]


def test_chat_reuses_provided_session_id_and_carries_history(monkeypatch):
    captured_history = []

    def fake_orchestrator(user_message, history=None):
        captured_history.append(list(history or []))
        return FAKE_RESULT

    monkeypatch.setattr(api, "run_orchestrator", fake_orchestrator)

    r1 = client.post("/chat", json={"message": "where is order 1001?", "session_id": "abc123"})
    assert r1.json()["session_id"] == "abc123"

    r2 = client.post("/chat", json={"message": "refund it", "session_id": "abc123"})
    assert r2.json()["session_id"] == "abc123"

    # second call's history should include the first turn (memory persisted across requests)
    assert len(captured_history[0]) == 0
    assert len(captured_history[1]) == 2


def test_chat_rejects_empty_message():
    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422


def test_api_request_id_deduplicates_and_get_run_is_session_scoped():
    message = {"message": "查询订单 1002 的物流", "session_id": "durable-api", "request_id": "stable"}
    first = client.post("/chat", json=message).json()
    repeated = client.post("/chat", json=message).json()
    assert first["run_id"] == repeated["run_id"]
    assert repeated["cached"]
    saved = client.get("/conversations/durable-api").json()
    assert len(saved["messages"]) == 2
    assert client.get(f"/runs/{first['run_id']}", params={"session_id": "other"}).status_code == 404
    restored = client.post(f"/runs/{first['run_id']}/resume", json={"session_id": "durable-api"})
    assert restored.status_code == 200 and restored.json()["cached"]
    conflicting = client.post("/chat", json={**message, "message": "different"})
    assert conflicting.status_code == 409
    assert conflicting.json()["detail"]["code"] == "REQUEST_ID_CONFLICT"


def test_api_stale_action_cannot_confirm_new_pending_action():
    first = client.post("/chat", json={"message": "帮我支付订单 1003", "session_id": "stale-api"}).json()
    client.post("/chat", json={"message": "取消", "session_id": "stale-api"})
    client.post("/chat", json={"message": "订单 1001 我想退款", "session_id": "stale-api"})
    response = client.post("/chat", json={"message": "确认", "session_id": "stale-api", "action_id": first["pending_action"]["action_id"]})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "APPROVAL_STALE"
    assert tools.get_payment_status(1003)["status"] == "pending"
    assert tools.check_refund_eligibility(1001)["eligible"]


def test_api_interruption_returns_handle_and_resumes_without_losing_the_request(monkeypatch):
    original = api.run_orchestrator
    def interrupted(*args, **kwargs):
        raise RuntimeError("injected process interruption")
    monkeypatch.setattr(api, "run_orchestrator", interrupted)
    failed = client.post("/chat", json={"message": "查询订单 1002 的状态", "request_id": "restart"})
    assert failed.status_code == 502
    handle = failed.json()["detail"]
    assert handle["session_id"] and handle["run_id"]
    monkeypatch.setattr(api, "run_orchestrator", original)
    resumed = client.post(f"/runs/{handle['run_id']}/resume", json={"session_id": handle["session_id"]})
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "SUCCEEDED"
    assert resumed.json()["run_id"] == handle["run_id"]
