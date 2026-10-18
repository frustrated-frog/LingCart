import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import orchestrator, tools
from src.adaptive import run_adaptive
from src.backend.db import Payment, get_engine, get_session_factory
from src.backend.seed import seed
from src.dialogue_understanding import TurnUnderstanding, to_task_plan


class ScriptedModel:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.index = 0

    def json_mode(self):
        return self

    def invoke(self, _messages):
        payload = self.payloads[min(self.index, len(self.payloads) - 1)]
        self.index += 1
        content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return AIMessage(content=content)


def interpretation(*, commands=None, objectives=None, reason="test"):
    return {"reason": reason, "commands": commands or [], "objectives": objectives or []}


@pytest.fixture(autouse=True)
def database():
    engine = get_engine("sqlite:///:memory:")
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
    return engine


@pytest.mark.parametrize(
    "message,kind,expected",
    [
        ("方便说说你到底是什么助手吗", "IDENTITY", "我是 LingCart"),
        ("第一次来，我可以让你帮我干哪些事", "CAPABILITIES", "查询服务规则"),
        ("嗨，第一次见", "GREETING", "你好，我是 LingCart"),
        ("谢啦，明白了", "ACKNOWLEDGEMENT", "不客气"),
    ],
)
def test_open_language_dialogue_is_semantic_not_pattern_matched(monkeypatch, message, kind, expected):
    model = ScriptedModel(interpretation(commands=[{"kind": kind, "source_quote": message, "message": None}]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["route"] == "CONVERSATIONAL"
    assert result["trace"] == []
    assert result["understanding"]["commands"][0]["kind"] == kind
    assert expected in result["reply"]


def test_multi_intent_turn_compiles_independent_verified_steps(monkeypatch):
    message = "给我瞅瞅 1002 那单进展，再挑一副耳机"
    model = ScriptedModel(interpretation(objectives=[
        {"id": "status", "kind": "ORDER_STATUS", "source_quote": "1002 那单进展", "order_id": 1002},
        {"id": "shopping", "kind": "PRODUCT_SEARCH", "source_quote": "挑一副耳机", "category": "耳机"},
    ]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert [step["tool"] for step in result["trace"]] == ["get_order_status", "search_products"]
    assert result["plan"]["verification"]["all_goals_satisfied"] is True
    assert "订单 1002 当前运输中" in result["reply"]
    assert "本地演示目录" in result["reply"]


def test_social_command_can_accompany_a_business_objective(monkeypatch):
    message = "嗨，顺便替我瞅瞅 1002 那单现在是什么情况"
    model = ScriptedModel(interpretation(
        commands=[{"kind": "GREETING", "source_quote": "嗨", "message": None}],
        objectives=[{"id": "shipment", "kind": "SHIPMENT_TRACKING",
                     "source_quote": "1002 那单现在是什么情况", "order_id": 1002}],
    ))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["reply"].startswith("你好，我是 LingCart。")
    assert result["trace"][0]["tool"] == "track_shipment"


def test_open_set_request_is_rejected_without_fake_handoff(monkeypatch):
    message = "替我规划一条去冰岛看极光的路线"
    model = ScriptedModel(interpretation(commands=[
        {"kind": "UNSUPPORTED", "source_quote": message, "message": None}
    ]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "FAILED"
    assert result["terminal_reason"] == "UNSUPPORTED_INTENT"
    assert result["route"] == "UNSUPPORTED"
    assert result["trace"] == []


def test_semantic_handoff_requires_an_explicit_human_request(monkeypatch):
    message = "我想跟你们的人聊聊"
    model = ScriptedModel(interpretation(commands=[
        {"kind": "HUMAN_HANDOFF", "source_quote": message, "message": None}
    ]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "WAITING_HUMAN"
    assert result["terminal_reason"] == "USER_REQUESTED_ESCALATION"
    assert result["trace"][0]["tool"] == "escalate_to_human"


def test_missing_required_slot_clarifies_after_semantic_recognition(monkeypatch):
    message = "帮我看一下前几天买的那个包裹走到哪了"
    model = ScriptedModel(interpretation(objectives=[
        {"id": "shipment", "kind": "SHIPMENT_TRACKING", "source_quote": message, "order_id": None}
    ]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "WAITING_CLARIFICATION"
    assert result["terminal_reason"] == "MISSING_ORDER_ID"
    assert result["trace"] == []


def test_model_protocol_gets_one_bounded_repair(monkeypatch):
    message = "方便介绍下你吗"
    model = ScriptedModel(
        "not json",
        interpretation(commands=[{"kind": "IDENTITY", "source_quote": message, "message": None}]),
    )
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["budget"]["consumed"]["model_calls"] == 2
    assert any(event["type"] == "dialogue_protocol_retry" for event in result["events"])


def test_ungrounded_model_entity_never_reaches_a_tool(monkeypatch):
    message = "帮我处理一下那个订单"
    model = ScriptedModel(interpretation(objectives=[
        {"id": "payment", "kind": "PAYMENT_PROCESS", "source_quote": message, "order_id": 1003}
    ]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "FAILED"
    assert result["terminal_reason"] == "UNGROUNDED_PLAN_ENTITY"
    assert result["trace"] == []


def test_pending_action_does_not_block_an_unrelated_semantic_turn(monkeypatch, database):
    pending = orchestrator.run_orchestrator("帮我支付订单 1003")["pending_action"]
    message = "先等等，你具体能帮我做些什么"
    model = ScriptedModel(interpretation(commands=[
        {"kind": "CAPABILITIES", "source_quote": message, "message": None}
    ]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message, pending_action=pending)

    assert result["status"] == "SUCCEEDED"
    assert result["pending_action"] == pending
    with get_session_factory(database)() as session:
        assert session.query(Payment).filter_by(order_id=1003).one().status == "pending"


def test_semantic_confirmation_cannot_weaken_explicit_authorization(monkeypatch, database):
    pending = orchestrator.run_orchestrator("帮我支付订单 1003")["pending_action"]
    message = "那就照你说的办吧"
    model = ScriptedModel(interpretation(commands=[
        {"kind": "CONFIRM_PENDING", "source_quote": message, "message": None}
    ]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message, pending_action=pending)

    assert result["status"] == "WAITING_APPROVAL"
    assert result["pending_action"] == pending
    with get_session_factory(database)() as session:
        assert session.query(Payment).filter_by(order_id=1003).one().status == "pending"


def test_semantic_confirmation_without_pending_action_clarifies(monkeypatch):
    message = "那就照你说的办吧"
    model = ScriptedModel(interpretation(commands=[
        {"kind": "CONFIRM_PENDING", "source_quote": message, "message": None}
    ]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "WAITING_CLARIFICATION"
    assert result["terminal_reason"] == "NO_PENDING_ACTION"
    assert result["trace"] == []


def test_supported_and_unsupported_parts_finish_as_partial(monkeypatch):
    message = "查下订单 1002 的状态，再替我买一张去冰岛的机票"
    model = ScriptedModel(interpretation(
        commands=[{"kind": "UNSUPPORTED", "source_quote": "替我买一张去冰岛的机票", "message": None}],
        objectives=[{"id": "status", "kind": "ORDER_STATUS",
                     "source_quote": "订单 1002 的状态", "order_id": 1002}],
    ))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "PARTIAL"
    assert result["terminal_reason"] == "PARTIAL_UNSUPPORTED_INTENT"
    assert result["trace"][0]["tool"] == "get_order_status"
    assert result["events"][-1]["status"] == "PARTIAL"


def test_stale_history_cannot_supply_a_current_dialogue_command(monkeypatch):
    message = "订单 1002 现在怎么样"
    history = [HumanMessage(content="我想跟人工聊聊")]
    model = ScriptedModel(interpretation(commands=[
        {"kind": "HUMAN_HANDOFF", "source_quote": "我想跟人工聊聊", "message": None}
    ]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message, history=history)

    assert result["status"] == "FAILED"
    assert result["terminal_reason"] == "UNGROUNDED_INTENT"
    assert result["trace"] == []


def test_followup_may_use_a_grounded_order_from_human_history(monkeypatch):
    message = "那个包裹现在呢"
    history = [HumanMessage(content="我之前问的是订单 1002")]
    model = ScriptedModel(interpretation(objectives=[
        {"id": "shipment", "kind": "SHIPMENT_TRACKING", "source_quote": message, "order_id": 1002}
    ]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message, history=history)

    assert result["status"] == "SUCCEEDED"
    assert result["trace"][0]["args"]["order_id"] == 1002


def test_adaptive_mode_uses_the_same_dialogue_command_vocabulary(monkeypatch):
    message = "方便说说你到底是什么助手吗"
    payload = {
        "reason": "身份问题",
        "goal": None,
        "clarification": None,
        "next_steps": None,
        "commands": [{"kind": "IDENTITY", "source_quote": message, "message": None}],
    }
    monkeypatch.setattr("src.adaptive_decision.get_llm", lambda **_kwargs: ScriptedModel(payload))

    result = run_adaptive(message)

    assert result["status"] == "SUCCEEDED"
    assert result["route"] == "CONVERSATIONAL"
    assert result["trace"] == []
    assert "我是 LingCart" in result["reply"]


def test_compiler_deduplicates_shared_read_dependencies():
    parsed = TurnUnderstanding.model_validate(interpretation(objectives=[
        {"id": "status", "kind": "ORDER_STATUS", "source_quote": "订单 1002", "order_id": 1002},
        {"id": "refund", "kind": "REFUND_ELIGIBILITY", "source_quote": "订单 1002", "order_id": 1002},
    ]))

    _, plan = to_task_plan(parsed, "订单 1002")

    assert [step.tool for step in plan.steps] == ["get_order_status", "check_refund_eligibility"]
    assert plan.steps[1].depends_on == ["step-1"]
