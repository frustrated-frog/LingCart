from fastapi.testclient import TestClient
from pathlib import Path
import pytest

import api
from src import tools
from src.backend.db import get_engine, get_session_factory
from src.backend.seed import seed


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch):
    monkeypatch.setenv("LINGCART_ENGINEERING_KEY", "test-engineering-key")
    engine = get_engine("sqlite:///:memory:")
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)

client = TestClient(api.app, headers={"X-Engineering-Key": "test-engineering-key"})

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


def test_production_build_and_client_routes_are_served_when_available():
    if not (Path(api.__file__).parent / 'frontend' / 'dist').is_dir():
        pytest.skip('React production build is not present')
    home = client.get('/')
    deep_link = client.get('/mechanisms')
    assert home.status_code == 200
    assert deep_link.status_code == 200
    assert 'LingCart · Agent Engineering' in home.text
    assert '<div id="root"></div>' in deep_link.text


def test_engineering_overview_exposes_real_registry_and_aggregates_only():
    response = client.get('/engineering/overview')
    assert response.status_code == 200
    payload = response.json()
    assert payload['counts']['profiles'] >= 3
    assert payload['counts']['dataset_cases'] == 100
    assert {item['profile_id'] for item in payload['profiles']} >= {
        'baseline-v1', 'adaptive-v1', 'adaptive-objective-context-v1'}
    assert all('cases' not in item for item in payload['datasets'])
    assert all('result' not in item for item in payload['runs'])


def test_engineering_api_rejects_remote_requests_without_operator_key():
    untrusted = TestClient(api.app)
    response = untrusted.get('/engineering/overview')
    assert response.status_code == 403
    assert response.json()['detail']['code'] == 'ENGINEERING_ACCESS_DENIED'


def test_engineering_api_allows_only_direct_local_access_without_key(monkeypatch):
    monkeypatch.delenv('LINGCART_ENGINEERING_KEY')
    local = TestClient(api.app, base_url='http://127.0.0.1', client=('127.0.0.1', 51000))
    assert local.get('/engineering/overview').status_code == 200
    proxied = local.get('/engineering/overview', headers={'X-Forwarded-For': '203.0.113.10'})
    assert proxied.status_code == 403


def test_engineering_dataset_detail_and_business_catalog_are_api_backed():
    dataset = client.get('/engineering/datasets/offline-core-v1')
    assert dataset.status_code == 200
    assert dataset.json()['dataset']['case_count'] == 60
    assert len(dataset.json()['dataset']['cases']) == 60
    assert client.get('/engineering/datasets/missing').status_code == 404

    products = client.get('/business/products', params={'category': '耳机', 'max_price': 500})
    assert products.status_code == 200
    assert products.json()['total'] > 0
    assert all(item['category'] == '耳机' and item['price'] <= 500
               for item in products.json()['products'])
    orders = client.get('/business/orders', params={'status': 'shipped'})
    assert orders.status_code == 200
    assert all(item['status'] == 'shipped' for item in orders.json()['orders'])


def test_engineering_run_detail_is_session_independent_but_contains_runtime_evidence():
    result = client.post('/chat', json={
        'message': '查询订单 1002 的状态和物流',
        'session_id': 'engineering-inspector',
        'request_id': 'engineering-run',
        'harness_profile': 'adaptive-v1',
    }).json()
    detail = client.get(f"/engineering/runs/{result['run_id']}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload['run_id'] == result['run_id']
    assert payload['profile']['profile_id'] == 'adaptive-v1'
    assert payload['event_count'] == len(payload['result']['events'])
    assert payload['trace_projection']['integrity']['sequential'] is True
    assert client.get('/engineering/runs/missing').status_code == 404


def test_harness_profiles_are_discoverable_but_candidates_are_research_only():
    profiles = client.get('/harness/profiles')
    assert profiles.status_code == 200
    assert {'baseline-v1', 'adaptive-v1', 'adaptive-objective-context-v1'} <= {
        item['profile_id'] for item in profiles.json()['profiles']}
    candidate = next(item for item in profiles.json()['profiles']
                     if item['profile_id'] == 'adaptive-objective-context-v1')
    assert candidate['product_eligible'] is False
    result = client.post('/chat', json={'message': '查询订单1002状态',
                                        'harness_profile': 'adaptive-objective-context-v1'})
    assert result.status_code == 409
    assert result.json()['detail']['code'] == 'HARNESS_PROFILE_NOT_APPROVED'
    baseline = client.post('/chat', json={'message': '查询订单1002状态',
                                          'harness_profile': 'adaptive-v1'}).json()
    assert baseline['harness']['profile_id'] == 'adaptive-v1'


def test_async_product_api_also_rejects_candidate_profile():
    result = client.post('/runs', json={'message': '查询订单1002状态',
                                        'harness_profile': 'adaptive-objective-context-v1'})
    assert result.status_code == 409
    assert result.json()['detail']['code'] == 'HARNESS_PROFILE_NOT_APPROVED'


def test_product_resume_endpoints_cannot_reopen_a_candidate_run():
    from src.durable import run_conversation
    from src.harness import get_runner as get_research_runner
    from src.persistence import RunStore

    class Crash(BaseException):
        pass

    class CrashAfterBinding(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            if event and event['type'] == 'harness_profile_selected':
                raise Crash()

    with pytest.raises(Crash):
        run_conversation('查询订单1002状态', 'candidate-resume', store=CrashAfterBinding(),
                         runner=get_research_runner('adaptive-objective-context-v1'))
    run_id = RunStore().conversation('candidate-resume')['active_run_id']

    for suffix in ('resume', 'resume-async'):
        response = client.post(f'/runs/{run_id}/{suffix}', json={'session_id': 'candidate-resume'})
        assert response.status_code == 409
        assert response.json()['detail']['code'] == 'HARNESS_PROFILE_NOT_APPROVED'
    assert RunStore().run(run_id)['status'] == 'INTERRUPTED'


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
    monkeypatch.setattr(api, "get_runner", lambda _profile: lambda *_a, **_k: FAKE_RESULT)

    response = client.post("/chat", json={"message": "where is order 1001?"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Your order 1001 has been delivered."
    assert body["agent"] == "order"
    assert body["trace"][0]["tool"] == "get_order_status"
    assert body["usage"]["total_tokens"] == 15
    assert "session_id" in body


def test_chat_generates_new_session_id_when_omitted(monkeypatch):
    monkeypatch.setattr(api, "get_runner", lambda _profile: lambda *_a, **_k: FAKE_RESULT)

    r1 = client.post("/chat", json={"message": "hello"})
    r2 = client.post("/chat", json={"message": "hello again"})

    assert r1.json()["session_id"] != r2.json()["session_id"]


def test_chat_reuses_provided_session_id_and_carries_history(monkeypatch):
    captured_history = []

    def fake_orchestrator(user_message, history=None):
        captured_history.append(list(history or []))
        return FAKE_RESULT

    monkeypatch.setattr(api, "get_runner", lambda _profile: fake_orchestrator)

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
    trace = client.get(f"/runs/{first['run_id']}/trace", params={"session_id": "durable-api"})
    assert trace.status_code == 200
    assert trace.json()["run_id"] == first["run_id"]
    assert trace.json()["integrity"]["sequential"] is True
    assert client.get(f"/runs/{first['run_id']}/trace", params={"session_id": "other"}).status_code == 404
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
    def interrupted(*args, **kwargs):
        raise RuntimeError("injected process interruption")
    monkeypatch.setattr(api, "get_runner", lambda _profile: interrupted)
    failed = client.post("/chat", json={"message": "查询订单 1002 的状态", "request_id": "restart"})
    assert failed.status_code == 502
    handle = failed.json()["detail"]
    assert handle["session_id"] and handle["run_id"]
    resumed = client.post(f"/runs/{handle['run_id']}/resume", json={"session_id": handle["session_id"]})
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "SUCCEEDED"
    assert resumed.json()["run_id"] == handle["run_id"]


def test_events_reconnect_uses_persisted_sequence():
    run = client.post('/chat', json={'message': '查询订单1002物流', 'session_id': 'stream'}).json()
    response = client.get(f"/runs/{run['run_id']}/events", params={'session_id': 'stream'}, headers={'Last-Event-ID': '1'})
    assert response.status_code == 200
    assert 'id: 1\n' not in response.text
    assert 'event: terminal' in response.text
    assert client.get(f"/runs/{run['run_id']}/events", params={'session_id': 'other'}).status_code == 404


def test_preferences_require_explicit_save_and_are_scoped():
    profile = client.post('/profiles').json()['profile_id']
    other = client.post('/profiles').json()['profile_id']
    assert client.put(f'/profiles/{profile}/preferences', json={'key': 'max_price', 'value': 300}).status_code == 409
    assert client.put(f'/profiles/{profile}/preferences', json={'key': 'max_price', 'value': 300, 'confirmed': True}).status_code == 200
    assert client.get(f'/profiles/{other}/preferences').json()['preferences'] == []
    assert client.post(f'/profiles/{profile}/bindings', json={'session_id': 'profile-session'}).status_code == 200
    assert client.delete(f'/profiles/{profile}/preferences/max_price').json()['forgotten']


@pytest.mark.parametrize('path', ['steer', 'cancel', 'follow-up'])
@pytest.mark.parametrize('fields', [{}, {'command_id': 'id'}, {'expected_revision': 0},
                                    {'command_id': 'id', 'expected_revision': None},
                                    {'command_id': 'id', 'expected_revision': True}])
def test_control_api_requires_id_and_revision(path, fields):
    response = client.post(f'/runs/unknown/{path}', json={'session_id': 'session', 'message': '新需求', **fields})
    assert response.status_code == 422


def test_control_api_retries_are_idempotent_and_old_window_is_rejected():
    from src.session_control import ControlStore
    store = ControlStore()
    run_id = store.claim('two-windows', 'request', '原需求', {})['run_id']
    body = dict(session_id='two-windows', message='预算300', command_id='window-one', expected_revision=0)
    first = client.post(f'/runs/{run_id}/steer', json=body)
    assert first.status_code == 202
    assert client.post(f'/runs/{run_id}/steer', json=body).json() == first.json()
    stale = client.post(f'/runs/{run_id}/cancel', json={**body, 'command_id': 'window-two'})
    assert stale.status_code == 409
    assert stale.json()['detail']['code'] == 'STALE_CONTROL_REVISION'
    assert store.revision(run_id) == 1
