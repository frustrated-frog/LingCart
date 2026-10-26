import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import orchestrator, tools
from src.adaptive import run_adaptive
from src.backend.db import Payment, get_engine, get_session_factory
from src.backend.seed import seed
from src.dialogue_understanding import TurnUnderstanding, _prompt, to_task_plan, understand_turn
from src.errors import AgentError
from src.runtime import BudgetLimits, RunRuntime, runtime_scope
from src.task_plan import execute_plan


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


def test_evidence_boundary_profile_adds_case_first_instruction_only_for_candidate():
    from src.harness import get_runner

    message = "订单 1001 商品与描述不符，请建立售后案件；原包装情况稍后确认。"

    def prompt_for(profile_id):
        runtime = RunRuntime(profile_id, BudgetLimits())
        with runtime_scope(runtime):
            get_runner(profile_id).bind(runtime)
            return _prompt(message, None, None)

    baseline_prompt = prompt_for("after-sales-policy-execution-v1")
    candidate_prompt = prompt_for("after-sales-evidence-boundary-v2")
    readiness_prompt = prompt_for("after-sales-evidence-readiness-v3")

    assert "Known evidence absence is not missing case-intake information" not in baseline_prompt
    assert "evidence_context_quotes" not in baseline_prompt
    assert "evidence_readiness_claims" not in baseline_prompt
    assert "EvidenceReadinessClaim" not in baseline_prompt
    assert "Known evidence absence is not missing case-intake information" in candidate_prompt
    assert "evidence_context_quotes" in candidate_prompt
    assert "evidence_readiness_claims" not in candidate_prompt
    assert "EvidenceReadinessClaim" not in candidate_prompt
    assert "Evidence-readiness ledger for this Harness Profile" in readiness_prompt
    assert "Known evidence absence is not missing case-intake information" in readiness_prompt
    assert "evidence_context_quotes" in readiness_prompt
    assert "evidence_readiness_claims" in readiness_prompt
    assert "after-sales-evidence-readiness/2026-01" in readiness_prompt


def test_evidence_protocol_fields_are_rejected_outside_the_bound_profile():
    from src.harness import get_runner

    message = "订单 1001 商品不想要了，请建立售后案件；原包装稍后确认"
    payload = interpretation(objectives=[{
        "id": "case",
        "kind": "AFTER_SALES_CASE",
        "source_quote": "订单 1001 商品不想要了，请建立售后案件",
        "evidence_readiness_claims": [{
            "evidence_type": "ORIGINAL_PACKAGING",
            "disposition": "DEFERRED",
            "source_quote": "原包装稍后确认",
        }],
        "order_id": 1001,
        "reason_code": "CHANGED_MIND",
    }])
    runtime = RunRuntime("after-sales-policy-execution-v1", BudgetLimits())
    with runtime_scope(runtime):
        get_runner("after-sales-policy-execution-v1").bind(runtime)
        with pytest.raises(AgentError) as blocked:
            understand_turn(message, None, None, model=ScriptedModel(payload, payload))

    assert blocked.value.code == "PROFILE_PROTOCOL_VIOLATION"


def test_evidence_context_quotes_are_scoped_to_after_sales_cases():
    with pytest.raises(ValueError, match="Only AFTER_SALES_CASE"):
        TurnUnderstanding.model_validate({
            "reason": "不能用证据上下文隐藏独立订单目标",
            "commands": [],
            "objectives": [{
                "id": "order-status",
                "kind": "ORDER_STATUS",
                "source_quote": "查询订单 1001",
                "evidence_context_quotes": ["再看看物流"],
                "order_id": 1001,
            }],
        })


def test_evidence_readiness_claims_are_scoped_and_compile_into_the_guarded_tool():
    with pytest.raises(ValueError, match="Only AFTER_SALES_CASE"):
        TurnUnderstanding.model_validate({
            "reason": "不能用证据延期隐藏订单查询目标",
            "commands": [],
            "objectives": [{
                "id": "order-status",
                "kind": "ORDER_STATUS",
                "source_quote": "查询订单 1001",
                "evidence_readiness_claims": [{
                    "evidence_type": "ORIGINAL_PACKAGING",
                    "disposition": "DEFERRED",
                    "source_quote": "原包装稍后确认",
                }],
                "order_id": 1001,
            }],
        })

    understanding = TurnUnderstanding.model_validate({
        "reason": "案件信息完整，政策材料被明确延期",
        "commands": [],
        "objectives": [{
            "id": "case",
            "kind": "AFTER_SALES_CASE",
            "source_quote": "订单 1001 商品不想要了，请建立售后案件",
            "evidence_readiness_claims": [{
                "evidence_type": "ORIGINAL_PACKAGING",
                "disposition": "DEFERRED",
                "source_quote": "原包装稍后确认",
            }],
            "order_id": 1001,
            "reason_code": "CHANGED_MIND",
        }],
    })

    _intake, plan = to_task_plan(
        understanding,
        "订单 1001 商品不想要了，请建立售后案件；原包装稍后确认",
    )

    assert plan.steps[0].tool == "prepare_after_sales_case"
    assert plan.steps[0].args["evidence_readiness_claims"] == [{
        "evidence_type": "ORIGINAL_PACKAGING",
        "disposition": "DEFERRED",
        "source_quote": "原包装稍后确认",
    }]

    with pytest.raises(ValueError, match="cannot add or revoke evidence"):
        TurnUnderstanding.model_validate({
            "reason": "同一目标不能混淆时间声明与证据写入",
            "commands": [],
            "objectives": [{
                "id": "case-update",
                "kind": "AFTER_SALES_CONTINUE",
                "source_quote": "原包装稍后确认，但先记为包装完整",
                "case_id": "case-1",
                "case_revision": 1,
                "evidence_text": "包装完整",
                "evidence_readiness_claims": [{
                    "evidence_type": "ORIGINAL_PACKAGING",
                    "disposition": "DEFERRED",
                    "source_quote": "原包装稍后确认",
                }],
            }],
        })


def catalog_extreme_spec(message: str, *, scope: str, with_category: bool) -> dict:
    return {
        "version": "catalog-query/v1",
        "scope": {"requirement_id": "scope", "source_quote": message, "kind": scope},
        "availability": {"requirement_id": "availability", "source_quote": message, "scope": "ALL"},
        "filters": ([{
            "requirement_id": "category",
            "source_quote": "耳机里面",
            "field": {"core": "category"},
            "operator": "EQ",
            "value": "耳机",
        }] if with_category else []),
        "partition": None,
        "ordering": [],
        "selector": {
            "requirement_id": "cheapest",
            "source_quote": "最便宜的是哪款",
            "mode": "ARGMIN",
            "field": {"core": "price"},
            "include_ties": False,
        },
        "metrics": [],
        "projection": {
            "requirement_id": "product",
            "source_quote": "哪款",
            "fields": [{"core": "product_id"}, {"core": "name"}, {"core": "price"}],
        },
    }


@pytest.fixture(autouse=True)
def database():
    engine = get_engine("sqlite:///:memory:")
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
    return engine


def test_explicit_current_category_repairs_stale_previous_result_scope():
    message = "耳机里面最便宜的是哪款"
    previous = catalog_extreme_spec(message, scope="PREVIOUS_RESULT_SET", with_category=True)
    current = catalog_extreme_spec(message, scope="CATALOG", with_category=True)
    model = ScriptedModel(
        interpretation(objectives=[{
            "id": "cheapest", "kind": "CATALOG_QUERY", "source_quote": message,
            "catalog_query": previous,
        }]),
        interpretation(objectives=[{
            "id": "cheapest", "kind": "CATALOG_QUERY", "source_quote": message,
            "catalog_query": current,
        }]),
    )

    result = understand_turn(
        message,
        [HumanMessage(content="把所有商品按类目展示")],
        None,
        model=model,
    )

    assert result.objectives[0].catalog_query.scope.kind == "CATALOG"
    assert model.index == 2


def test_elliptical_extreme_followup_can_reuse_previous_result_scope():
    message = "最便宜的是哪款"
    spec = catalog_extreme_spec(message, scope="PREVIOUS_RESULT_SET", with_category=False)
    model = ScriptedModel(interpretation(objectives=[{
        "id": "cheapest", "kind": "CATALOG_QUERY", "source_quote": message,
        "catalog_query": spec,
    }]))

    result = understand_turn(
        message,
        [HumanMessage(content="耳机都有哪些")],
        None,
        model=model,
    )

    assert result.objectives[0].catalog_query.scope.kind == "PREVIOUS_RESULT_SET"
    assert model.index == 1


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


def test_negated_handoff_executes_the_positive_goal_without_escalating(monkeypatch):
    message = "不要转人工，先帮我查订单 1002 的物流"
    model = ScriptedModel(interpretation(objectives=[
        {"id": "shipment", "kind": "SHIPMENT_TRACKING", "source_quote": "查订单 1002 的物流", "order_id": 1002}
    ]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["trace"][0]["tool"] == "track_shipment"
    assert all(entry["tool"] != "escalate_to_human" for entry in result["trace"])


def test_negated_handoff_command_from_model_is_removed_before_terminal_validation(monkeypatch):
    message = "不要转人工，先帮我查订单 1002 的物流"
    model = ScriptedModel(interpretation(
        commands=[{"kind": "HUMAN_HANDOFF", "source_quote": "不要转人工", "message": None}],
        objectives=[{"id": "shipment", "kind": "SHIPMENT_TRACKING",
                     "source_quote": "查订单 1002 的物流", "order_id": 1002}],
    ))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["trace"][0]["tool"] == "track_shipment"
    assert result["budget"]["consumed"]["model_calls"] == 1


def test_negative_product_constraint_is_a_hard_catalog_exclusion(monkeypatch):
    message = "推荐预算 500 元的耳机，不要主动降噪"
    model = ScriptedModel(interpretation(objectives=[{
        "id": "headphones", "kind": "PRODUCT_SEARCH", "source_quote": message,
        "category": "耳机", "max_price": 500, "excluded_terms": ["主动降噪"],
    }]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["trace"][0]["args"]["excluded_terms"] == ["主动降噪"]
    assert result["trace"][0]["result"]["products"]
    assert all("主动降噪" not in str(product) for product in result["trace"][0]["result"]["products"])


def test_missing_coordinated_goal_gets_one_repair_before_execution(monkeypatch):
    message = "瞅瞅 1002 那单进展，然后计算 2+2"
    model = ScriptedModel(
        interpretation(objectives=[
            {"id": "status", "kind": "ORDER_STATUS", "source_quote": "1002 那单进展", "order_id": 1002},
        ]),
        interpretation(objectives=[
            {"id": "status", "kind": "ORDER_STATUS", "source_quote": "1002 那单进展", "order_id": 1002},
            {"id": "math", "kind": "CALCULATE", "source_quote": "计算 2+2", "expression": "2+2"},
        ]),
    )
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert [item["tool"] for item in result["trace"]] == ["get_order_status", "calculate"]
    assert result["budget"]["consumed"]["model_calls"] == 2
    assert any(event["type"] == "dialogue_protocol_retry" for event in result["events"])


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


def test_numeric_grounding_accepts_sentence_punctuation_but_not_decimal_prefix(monkeypatch):
    message = "Find 耳机 under CNY 500."
    grounded = ScriptedModel(interpretation(objectives=[{
        "id": "headphones", "kind": "PRODUCT_SEARCH", "source_quote": message,
        "category": "耳机", "max_price": 500,
    }]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: grounded)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["trace"][0]["args"]["max_price"] == 500

    decimal_message = "Find 耳机 under CNY 500.5."
    ungrounded = ScriptedModel(interpretation(objectives=[{
        "id": "headphones", "kind": "PRODUCT_SEARCH", "source_quote": decimal_message,
        "category": "耳机", "max_price": 500,
    }]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: ungrounded)

    rejected = orchestrator.run_orchestrator(decimal_message)

    assert rejected["status"] == "FAILED"
    assert rejected["terminal_reason"] == "UNGROUNDED_PLAN_ENTITY"
    assert rejected["trace"] == []


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


def test_unregistered_model_capability_gets_one_repair_to_unsupported(monkeypatch):
    message = "查订单 1002 的状态，然后替我买一张去冰岛的机票"
    model = ScriptedModel(
        interpretation(objectives=[
            {"id": "status", "kind": "ORDER_STATUS", "source_quote": "订单 1002 的状态", "order_id": 1002},
            {"id": "travel", "kind": "BOOK_FLIGHT", "source_quote": "替我买一张去冰岛的机票"},
        ]),
        interpretation(
            commands=[{"kind": "UNSUPPORTED", "source_quote": "替我买一张去冰岛的机票", "message": None}],
            objectives=[{"id": "status", "kind": "ORDER_STATUS",
                         "source_quote": "订单 1002 的状态", "order_id": 1002}],
        ),
    )
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "PARTIAL"
    assert result["terminal_reason"] == "PARTIAL_UNSUPPORTED_INTENT"
    assert result["trace"][0]["tool"] == "get_order_status"
    assert result["budget"]["consumed"]["model_calls"] == 2


def test_reserved_unsupported_kind_is_canonicalized_to_a_command(monkeypatch):
    message = "查订单 1002 的状态，然后替我买一张去冰岛的机票"
    model = ScriptedModel(interpretation(objectives=[
        {"id": "status", "kind": "ORDER_STATUS", "source_quote": "订单 1002 的状态", "order_id": 1002},
        {"id": "travel", "kind": "UNSUPPORTED", "source_quote": "替我买一张去冰岛的机票"},
    ]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "PARTIAL"
    assert result["trace"][0]["tool"] == "get_order_status"
    assert result["budget"]["consumed"]["model_calls"] == 1


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
    payload = interpretation(
        reason="身份问题",
        commands=[{"kind": "IDENTITY", "source_quote": message, "message": None}],
    )
    monkeypatch.setattr("src.llm.get_llm", lambda **_kwargs: ScriptedModel(payload))

    result = run_adaptive(message)

    assert result["status"] == "SUCCEEDED"
    assert result["route"] == "CONVERSATIONAL"
    assert result["trace"] == []
    assert "我是 LingCart" in result["reply"]


@pytest.mark.parametrize(
    "message,objective,tool",
    [
        ("今天是哪年哪月哪日", {"id": "clock", "kind": "SYSTEM_TIME", "source_quote": "今天是哪年哪月哪日"}, "read_clock"),
        ("计算 7*8", {"id": "math", "kind": "CALCULATE", "source_quote": "计算 7*8", "expression": "7*8"}, "calculate"),
    ],
)
def test_adaptive_open_set_utility_turns_use_the_shared_capability_path(monkeypatch, message, objective, tool):
    monkeypatch.setattr("src.llm.get_llm", lambda **_kwargs: ScriptedModel(
        interpretation(objectives=[objective])
    ))

    result = run_adaptive(message)

    assert result["status"] == "SUCCEEDED"
    assert result["execution_mode"] == "auto"
    assert result["trace"][0]["tool"] == tool
    assert not result["autonomous_executed"]


def test_adaptive_general_answer_uses_the_shared_registered_capability(monkeypatch):
    message = "解释什么是机会成本"
    semantic_model = ScriptedModel(interpretation(objectives=[{
        "id": "knowledge", "kind": "GENERAL_ANSWER", "source_quote": message, "question": message,
    }]))
    monkeypatch.setattr("src.llm.get_llm", lambda **_kwargs: semantic_model)
    monkeypatch.setattr("src.general_answer.get_llm", lambda **_kwargs: ScriptedModel("机会成本是放弃的最佳替代方案的价值。"))

    result = run_adaptive(message)

    assert result["status"] == "SUCCEEDED"
    assert result["route"] == "KNOWLEDGE"
    assert result["execution_mode"] == "auto"
    assert "最佳替代方案" in result["reply"]


def test_compiler_deduplicates_shared_read_dependencies():
    parsed = TurnUnderstanding.model_validate(interpretation(objectives=[
        {"id": "status", "kind": "ORDER_STATUS", "source_quote": "订单 1002", "order_id": 1002},
        {"id": "refund", "kind": "REFUND_ELIGIBILITY", "source_quote": "订单 1002", "order_id": 1002},
    ]))

    _, plan = to_task_plan(parsed, "订单 1002")

    assert [step.tool for step in plan.steps] == ["get_order_status", "check_refund_eligibility"]
    assert plan.steps[1].depends_on == ["step-1"]


def test_deduplicated_evidence_is_attributed_to_every_declared_goal():
    parsed = TurnUnderstanding.model_validate(interpretation(objectives=[
        {"id": "status", "kind": "ORDER_STATUS", "source_quote": "订单 1002", "order_id": 1002},
        {"id": "refund", "kind": "REFUND_ELIGIBILITY", "source_quote": "订单 1002", "order_id": 1002},
    ]))
    intake, plan = to_task_plan(parsed, "订单 1002")

    result = execute_plan(intake, "shared-evidence", plan=plan, source="test")

    outcomes = {row["objective_id"]: row["outcome"]
                for row in result["plan"]["verification"]["goal_outcomes"]}
    assert outcomes == {"status": "SATISFIED", "refund": "SATISFIED"}
    assert result["plan"]["steps"][0]["objective_ids"] == ["status", "refund"]
