import pytest
from sqlalchemy import select

from src import tools
from src.backend.db import get_engine, get_session_factory
from src.backend.seed import seed
from src.durable import run_conversation
from src.errors import AgentError
from src.orchestrator import run_orchestrator
from src.runtime import BudgetLimits
from src.task_plan import TaskPlan, TaskStep


@pytest.fixture(autouse=True)
def database():
    engine = get_engine("sqlite:///:memory:")
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)


def test_independent_results_survive_one_failed_order():
    result = run_orchestrator("分别查询订单 1002 和订单 9999 的状态")
    assert result["status"] == "PARTIAL"
    assert "运输中" in result["reply"]
    assert result["plan"]["verification"]["all_goals_satisfied"] is False
    assert len(result["plan"]["verification"]["incomplete_steps"]) == 1


def test_missing_dependency_blocks_only_dependent_steps():
    result = run_orchestrator("查询订单 9999 能不能退款，同时告诉我退货政策")
    steps = result["plan"]["steps"]
    assert result["status"] == "PARTIAL"
    assert next(s for s in steps if s["tool"] == "check_refund_eligibility")["status"] == "SKIPPED"
    assert next(s for s in steps if s["tool"] == "answer_faq")["status"] == "SUCCEEDED"
    assert not any(t["tool"] == "check_refund_eligibility" for t in result["trace"])


def test_plan_step_and_depth_limits_are_enforced_before_tools():
    result = run_orchestrator("查询订单 1001 的状态和物流", limits=BudgetLimits(max_steps=1))
    assert result["terminal_reason"] == "PLAN_STEP_BUDGET_EXHAUSTED"
    assert not result["trace"]
    result = run_orchestrator("订单 1001 我想退款", limits=BudgetLimits(max_depth=2))
    assert result["terminal_reason"] == "PLAN_DEPTH_BUDGET_EXHAUSTED"
    assert not result["trace"]
    with pytest.raises(AgentError):
        TaskPlan([], [TaskStep("cycle", "order", "get_order_status", {}, ["cycle"])]).validate(8, 4)


def test_approval_is_not_a_completed_goal():
    result = run_orchestrator("订单 1001 我想退款")
    assert result["status"] == "WAITING_APPROVAL"
    assert result["plan"]["verification"]["waiting_steps"]
    assert not result["plan"]["verification"]["all_goals_satisfied"]
    assert not result["plan"]["verification"]["side_effects_executed"]


def test_compound_goals_and_missing_slot_survive_new_service_instance():
    missing = run_conversation("查询订单的物流进度", "slots")
    assert missing["status"] == "WAITING_CLARIFICATION"
    completed = run_conversation("1002", "slots")
    assert completed["status"] == "SUCCEEDED"
    assert completed["trace"][0]["tool"] == "track_shipment"
    compound = run_conversation("查询订单 1002 的状态，同时推荐 500 元以内的耳机", "slots")
    assert {t["agent"] for t in compound["trace"]} == {"order", "shopping"}
    assert compound["plan"]["verification"]["all_goals_satisfied"]


def test_ambiguous_multi_order_goals_and_batch_writes_clarify():
    ambiguous = run_orchestrator("查询订单 1001 物流，订单 1002 是否能退款")
    assert ambiguous["status"] == "WAITING_CLARIFICATION"
    assert not ambiguous["trace"]
    batch = run_orchestrator("分别帮我支付订单 1003 和订单 1006")
    assert batch["status"] == "WAITING_CLARIFICATION"
    assert batch["terminal_reason"] == "MULTIPLE_ACTIONS_REQUIRE_SEPARATE_APPROVAL"
    assert not batch["trace"]


def test_structured_model_plan_dispatches_multiple_specialists(monkeypatch):
    import json
    from langchain_core.messages import AIMessage
    import src.planner as planner
    class PlannerModel:
        def invoke(self, message):
            return AIMessage(content=json.dumps({"goals": [
                {"intent": "ORDER_STATUS", "order_id": 1002},
                {"intent": "FAQ"},
            ]}))
    monkeypatch.setattr(planner, "get_llm", lambda **_kwargs: PlannerModel())
    result = run_orchestrator("订单 1002 的包裹怎么样，另外讲讲售后保障")
    assert result["status"] == "SUCCEEDED"
    assert {t["agent"] for t in result["trace"]} == {"order", "faq"}
    assert result["budget"]["consumed"]["model_calls"] == 1
    assert any(e["type"] == "verification" for e in result["events"])
    assert result["usage"]["total_tokens"] is None


def test_model_planner_cannot_invent_order_ids_or_write_without_approval(monkeypatch):
    import json
    from langchain_core.messages import AIMessage
    import src.planner as planner
    class PlannerModel:
        def __init__(self, order_id):
            self.order_id = order_id
        def invoke(self, message):
            return AIMessage(content=json.dumps({"goals": [{"intent": "PAYMENT_PROCESS", "order_id": self.order_id}]}))
    monkeypatch.setattr(planner, "get_llm", lambda **_kwargs: PlannerModel(1003))
    # Use an ambiguous compound phrasing so deterministic intake falls through.
    result = run_orchestrator("订单 1002 的包裹怎么样，另外处理一下账单")
    assert result["terminal_reason"] == "UNGROUNDED_PLAN_ENTITY"
    assert not result["trace"]
    monkeypatch.setattr(planner, "get_llm", lambda **_kwargs: PlannerModel(1003))
    result = run_orchestrator("订单 1003 的包裹怎么样，另外处理一下账单")
    assert result["status"] == "WAITING_APPROVAL"
    assert tools.get_payment_status(1003)["status"] == "pending"
    assert not any(t["tool"] == "process_payment" for t in result["trace"])


def test_tool_response_for_wrong_order_cannot_complete_a_goal(monkeypatch):
    import src.direct_executor as direct
    def wrong_order(order_id: int):
        return {"order_id": 1001, "status": "delivered"}
    monkeypatch.setitem(direct._READ_ONLY_TOOLS, "get_order_status", __import__("src.runtime", fromlist=["ToolSpec"]).ToolSpec(wrong_order))
    result = run_orchestrator("查询订单 1002 的状态")
    assert result["status"] == "FAILED"
    assert result["terminal_reason"] == "INVALID_TOOL_RESULT"
    assert not result["plan"]["verification"]["all_goals_satisfied"]
