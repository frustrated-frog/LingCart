import json

import pytest
from langchain_core.messages import AIMessage
from sqlalchemy import update

from src import orchestrator, tools
from src.backend.db import Product, get_engine, get_session_factory
from src.backend.seed import seed
from src.capability_registry import (
    CapabilityRegistration,
    CapabilityRegistry,
    CapabilitySpec,
    CompiledOperation,
    GENERAL_ANSWER,
    PRODUCT_DISCOVERY,
    SYSTEM_TIME,
    CALCULATE,
    OperationContract,
    use_capability_registry,
)
from src.dialogue_understanding import TurnUnderstanding, to_task_plan
from src.errors import AgentError
from src.runtime import ToolSpec


class ScriptedModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.index = 0

    def json_mode(self):
        return self

    def invoke(self, _prompt):
        response = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        if isinstance(response, dict):
            response = json.dumps(response, ensure_ascii=False)
        return AIMessage(content=response)


@pytest.fixture(autouse=True)
def database():
    engine = get_engine("sqlite:///:memory:")
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
    return engine


def understanding(message, *, objectives=None, commands=None):
    return {"reason": "test", "commands": commands or [], "objectives": objectives or []}


def test_business_objective_schema_has_no_catalog_category_enum():
    schema = TurnUnderstanding.model_json_schema()
    objective = schema["$defs"]["BusinessObjective"]["properties"]
    category_schema = objective["category"]
    assert "enum" not in json.dumps(category_schema)
    parsed = TurnUnderstanding.model_validate(understanding("推荐显示器", objectives=[{
        "id": "display", "kind": PRODUCT_DISCOVERY, "source_quote": "推荐显示器", "category": "显示器"
    }]))
    assert parsed.objectives[0].category == "显示器"


def test_registry_can_describe_a_new_capability_without_parser_changes():
    spec = CapabilitySpec(
        capability_id="demo.echo",
        objective_kind="DEMO_ECHO",
        description="回显一段已验证文本",
        target_type="text",
        evidence_requirement="MODEL_KNOWLEDGE",
        required_fields=("question",),
        accepted_fields=("question",),
    )
    registration = CapabilityRegistration(
        spec,
        lambda objective, _request: (
            CompiledOperation("echo", "demo", "echo", {"text": objective["question"]}),
        ),
    )
    registry = CapabilityRegistry((registration,))

    assert registry.prompt_catalog()[0]["objective_kind"] == "DEMO_ECHO"
    assert registry.compile({"kind": "DEMO_ECHO", "question": "hello"}, "hello")[0].arguments == {
        "text": "hello"
    }
    parsed = TurnUnderstanding.model_validate(understanding("hello", objectives=[{
        "id": "echo", "kind": "DEMO_ECHO", "source_quote": "hello", "question": "hello"
    }]))
    assert parsed.objectives[0].kind == "DEMO_ECHO"


def test_registry_rejects_unsafe_ungrounded_field_policy():
    common = dict(
        capability_id="demo.echo",
        objective_kind="DEMO_ECHO",
        description="回显一段已验证文本",
        target_type="text",
        evidence_requirement="MODEL_KNOWLEDGE",
        accepted_fields=("question",),
    )

    with pytest.raises(ValueError, match="must be accepted"):
        CapabilityRegistry((CapabilityRegistration(
            CapabilitySpec(**common, ungrounded_optional_defaults={"query": ""}),
            lambda _objective, _request: (),
        ),))

    with pytest.raises(ValueError, match="Required fields cannot be dropped"):
        CapabilityRegistry((CapabilityRegistration(
            CapabilitySpec(**common, required_fields=("question",),
                           ungrounded_optional_defaults={"question": ""}),
            lambda _objective, _request: (),
        ),))


def test_registered_capability_executes_validates_renders_and_reports_outcome_without_task_plan_changes(monkeypatch):
    message = "回显 hello"

    def echo(text: str) -> dict:
        return {"echo": text, "source": "DEMO_ECHO"}

    registration = CapabilityRegistration(
        CapabilitySpec(
            capability_id="demo.echo.v2",
            objective_kind="DEMO_ECHO",
            description="回显一段已验证文本",
            target_type="text",
            evidence_requirement="DETERMINISTIC_COMPUTATION",
            accepted_fields=("parameters",),
            required_parameters=("text",),
            parameter_schema={"text": "string"},
        ),
        lambda objective, _request: (
            CompiledOperation("echo", "demo", "demo_echo", {"text": objective["parameters"]["text"]}),
        ),
        operations={
            "demo_echo": OperationContract(
                handler=ToolSpec(echo),
                required_result_fields=("echo", "source"),
                result_validator=lambda result, args: result["echo"] == args["text"] and result["source"] == "DEMO_ECHO",
                renderer=lambda result, _args: "回显结果：" + result["echo"],
                outcome_mapper=lambda _result, _args: "SATISFIED",
            )
        },
    )
    registry = CapabilityRegistry((registration,))
    model = ScriptedModel(understanding(message, objectives=[{
        "id": "echo", "kind": "DEMO_ECHO", "source_quote": message,
        "target_type": "text", "evidence_requirement": "DETERMINISTIC_COMPUTATION",
        "parameters": {"text": "hello"},
    }]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    with use_capability_registry(registry):
        result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["reply"] == "回显结果：hello"
    assert result["trace"][0]["tool"] == "demo_echo"
    assert result["plan"]["verification"]["goal_outcomes"] == [{
        "objective_id": "echo", "outcome": "SATISFIED", "step_ids": ["step-1"]
    }]


def test_registered_operation_cannot_invent_a_completion_outcome():
    contract = OperationContract(handler=ToolSpec(lambda: {}), outcome_mapper=lambda _result, _args: "MAYBE")

    with pytest.raises(AgentError, match="未知目标状态") as error:
        contract.outcome({}, {})

    assert error.value.code == "INVALID_CAPABILITY_OUTCOME"


def test_open_ended_recommendation_returns_verified_diverse_candidates(monkeypatch):
    message = "给我推荐买些什么"
    model = ScriptedModel(understanding(message, objectives=[{
        "id": "discover", "kind": PRODUCT_DISCOVERY, "source_quote": message,
        "desired_outcome": "发现值得考虑的商品", "target_type": "product",
        "evidence_requirement": "CATALOG", "query": message
    }]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["trace"][0]["tool"] == "discover_products"
    payload = result["trace"][0]["result"]
    assert payload["interaction_policy"] == "RECOMMEND_AND_ASK"
    assert payload["unbound_catalog_language"] == [message]
    assert len(payload["products"]) == 4
    assert len({product["category"] for product in payload["products"]}) == 3
    assert payload["candidate_set"]["selected_ids"] == [p["product_id"] for p in payload["products"]]
    assert "探索推荐" in result["reply"]
    assert payload["follow_up_question"] in result["reply"]


def test_discovery_drops_model_invented_optional_filters_instead_of_rejecting_or_narrowing(monkeypatch):
    message = "给我推荐买些什么"
    model = ScriptedModel(understanding(message, objectives=[{
        "id": "discover", "kind": PRODUCT_DISCOVERY, "source_quote": message,
        "query": "热门商品", "category": "数码", "max_price": 999,
        "soft_preferences": ["高性价比"], "excluded_terms": ["低质量"],
    }]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["trace"][0]["tool"] == "discover_products"
    assert set(result["trace"][0]["args"]) == {"context_seed"}
    provenance = result["plan"]["steps"][0]["argument_provenance"]
    assert set(provenance) == {"query", "category", "max_price", "soft_preferences", "excluded_terms"}
    assert all(item["resolver"] == "ungrounded_optional_filter" for item in provenance.values())


def test_discovery_does_not_ground_invented_budget_inside_an_order_number(monkeypatch):
    message = "给我推荐点东西，我的会员号是 1999"
    model = ScriptedModel(understanding(message, objectives=[{
        "id": "discover", "kind": PRODUCT_DISCOVERY, "source_quote": "给我推荐点东西",
        "max_price": 999,
    }]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert "max_price" not in result["trace"][0]["args"]
    assert result["plan"]["steps"][0]["argument_provenance"]["max_price"]["normalized_value"] is None


def test_explicit_generic_recommendation_fast_path_uses_discovery(monkeypatch):
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: pytest.fail("model should not be called"))

    result = orchestrator.run_orchestrator("推荐商品")

    assert result["status"] == "SUCCEEDED"
    assert result["intake"]["intents"] == [PRODUCT_DISCOVERY]
    assert result["trace"][0]["tool"] == "discover_products"
    assert result["trace"][0]["result"]["interaction_policy"] == "RECOMMEND_AND_ASK"


def test_catalog_addition_is_discovered_without_source_changes(database):
    with get_session_factory(database)() as session:
        session.add(Product(product_id="PX001", name="澄光显示器", category="显示器", price=1299,
                            summary="27 英寸 · 4K · 办公创作",
                            specs={"尺寸": "27 英寸", "分辨率": "4K", "适用场景": "办公创作"}, stock=5))
        session.commit()

    facets = tools.get_catalog_facets()
    result = tools.discover_products(limit=4, context_seed="dynamic-category")

    assert "显示器" in facets["categories"]
    assert "显示器" in {product["category"] for product in result["products"]}


def test_discovery_is_not_the_cheapest_rows_and_is_reproducible():
    first = tools.discover_products(limit=4, context_seed="same-session")
    second = tools.discover_products(limit=4, context_seed="same-session")
    cheapest = tools.search_products(limit=4)

    assert first["candidate_set"]["selected_ids"] == second["candidate_set"]["selected_ids"]
    assert first["candidate_set"]["selected_ids"] != [p["product_id"] for p in cheapest["products"]]
    assert all("total" in first["candidate_set"]["score_components"][p["product_id"]]
               for p in first["products"])


def test_empty_catalog_is_a_verified_no_candidate_outcome(monkeypatch, database):
    with get_session_factory(database)() as session:
        session.execute(update(Product).values(stock=0))
        session.commit()
    message = "随便推荐点东西"
    model = ScriptedModel(understanding(message, objectives=[{
        "id": "discover", "kind": PRODUCT_DISCOVERY, "source_quote": message, "query": message
    }]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["terminal_reason"] == "NO_CANDIDATE_EVIDENCE"
    assert result["trace"][0]["result"]["interaction_policy"] == "NO_CANDIDATE_EVIDENCE"
    assert "unbound_catalog_language" not in result["trace"][0]["result"]
    assert "没有可验证的候选商品" in result["reply"]


def test_capability_contract_repairs_irrelevant_model_arguments(monkeypatch):
    message = "解释一下机会成本"
    model = ScriptedModel(
        understanding(message, objectives=[{
            "id": "knowledge", "kind": GENERAL_ANSWER, "source_quote": message,
            "question": message, "query": message,
        }]),
        understanding(message, objectives=[{
            "id": "knowledge", "kind": GENERAL_ANSWER, "source_quote": message,
            "question": message,
        }]),
        "机会成本是放弃的最佳替代方案的价值。",
    )
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)
    monkeypatch.setattr("src.general_answer.get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["budget"]["consumed"]["model_calls"] == 3
    assert any(event["type"] == "dialogue_protocol_retry" for event in result["events"])


def test_stable_general_question_uses_general_answer_capability(monkeypatch):
    message = "解释一下什么是机会成本"
    model = ScriptedModel(
        understanding(message, objectives=[{
            "id": "knowledge", "kind": GENERAL_ANSWER, "source_quote": message,
            "question": message, "evidence_requirement": "MODEL_KNOWLEDGE"
        }]),
        "机会成本是选择一个方案时放弃的最佳替代方案的价值。",
    )
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)
    monkeypatch.setattr("src.general_answer.get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["route"] == "KNOWLEDGE"
    assert result["trace"] == []
    assert "最佳替代方案" in result["reply"]
    assert result["budget"]["consumed"]["model_calls"] == 2


def test_realtime_question_reports_missing_evidence_instead_of_guessing(monkeypatch):
    message = "今天上海金价是多少"
    model = ScriptedModel(understanding(message, objectives=[{
        "id": "current", "kind": GENERAL_ANSWER, "source_quote": message,
        "question": message, "evidence_requirement": "REALTIME_EXTERNAL"
    }]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "PARTIAL"
    assert result["terminal_reason"] == "EVIDENCE_UNAVAILABLE"
    assert result["route"] == "KNOWLEDGE"
    assert "没有可验证的联网数据源" in result["reply"]
    assert result["budget"]["consumed"]["model_calls"] == 1
    assert result["plan"]["verification"]["goal_outcomes"][0]["outcome"] == "UNAVAILABLE_EVIDENCE"


def test_current_date_uses_runtime_clock_not_model_memory(monkeypatch):
    message = "今天是哪年哪月哪日"
    model = ScriptedModel(understanding(message, objectives=[{
        "id": "clock", "kind": SYSTEM_TIME, "source_quote": message,
        "desired_outcome": "获取当前日期", "evidence_requirement": "RUNTIME_ENVIRONMENT"
    }]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["trace"][0]["tool"] == "read_clock"
    assert result["trace"][0]["result"]["data_source"] == "RUNTIME_CLOCK"
    assert result["trace"][0]["result"]["timezone"] == "Asia/Shanghai"
    assert result["trace"][0]["result"]["date"] in result["reply"]
    assert result["budget"]["consumed"]["model_calls"] == 1


@pytest.mark.parametrize("model_timezone", [None, "America/New_York", "纽约"])
def test_location_is_resolved_to_a_grounded_iana_timezone(monkeypatch, model_timezone):
    message = "纽约现在几点"
    objective = {
        "id": "clock", "kind": SYSTEM_TIME, "source_quote": message,
        "evidence_requirement": "RUNTIME_ENVIRONMENT",
    }
    if model_timezone is not None:
        objective["timezone"] = model_timezone
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: ScriptedModel(
        understanding(message, objectives=[objective])
    ))

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["trace"][0]["args"]["timezone"] == "America/New_York"
    provenance = result["plan"]["steps"][0]["argument_provenance"]["timezone"]
    assert provenance["resolver"] == "trusted_timezone"
    assert provenance["source_quote"] == message


def test_timezone_resolver_uses_current_turn_when_model_quote_is_too_narrow(monkeypatch):
    message = "纽约现在几点"
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: ScriptedModel(
        understanding(message, objectives=[{
            "id": "clock", "kind": SYSTEM_TIME, "source_quote": "现在几点",
            "evidence_requirement": "RUNTIME_ENVIRONMENT",
        }])
    ))

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["trace"][0]["args"]["timezone"] == "America/New_York"
    assert result["plan"]["steps"][0]["argument_provenance"]["timezone"]["source_quote"] == message


def test_arithmetic_uses_bounded_calculator(monkeypatch):
    message = "计算 (18 + 6) / 3"
    model = ScriptedModel(understanding(message, objectives=[{
        "id": "math", "kind": CALCULATE, "source_quote": message,
        "expression": "(18 + 6) / 3", "evidence_requirement": "DETERMINISTIC_COMPUTATION"
    }]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["trace"][0]["tool"] == "calculate"
    assert result["trace"][0]["result"]["result"] == 8
    assert "= 8" in result["reply"]


def test_general_answer_and_business_lookup_can_share_one_turn(monkeypatch):
    message = "解释机会成本，再查订单 1002 的状态"
    model = ScriptedModel(
        understanding(message, objectives=[
            {"id": "knowledge", "kind": GENERAL_ANSWER, "source_quote": "解释机会成本",
             "question": "解释机会成本", "evidence_requirement": "MODEL_KNOWLEDGE"},
            {"id": "order", "kind": "ORDER_STATUS", "source_quote": "订单 1002 的状态",
             "order_id": 1002, "evidence_requirement": "BUSINESS_DATA"},
        ]),
        "机会成本是被放弃的最佳替代方案的价值。",
    )
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)
    monkeypatch.setattr("src.general_answer.get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["route"] == "MIXED"
    assert [step["tool"] for step in result["trace"]] == ["get_order_status"]
    assert "最佳替代方案" in result["reply"] and "订单 1002 当前运输中" in result["reply"]


def test_unknown_model_capability_never_reaches_execution(monkeypatch):
    message = "做一件系统从未注册过的事"
    model = ScriptedModel(understanding(message, objectives=[{
        "id": "unknown", "kind": "INVENTED_CAPABILITY", "source_quote": message
    }]))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "FAILED"
    assert result["terminal_reason"] == "UNSUPPORTED_INTENT"
    assert result["route"] == "UNSUPPORTED"
    assert result["trace"] == []
    assert result["budget"]["consumed"]["model_calls"] == 2
    assert any(event["type"] == "unknown_capability_downgraded" for event in result["events"])


def test_compiler_records_capability_and_objective_identity():
    parsed = TurnUnderstanding.model_validate(understanding("给我看看", objectives=[{
        "id": "discover", "kind": PRODUCT_DISCOVERY, "source_quote": "给我看看"
    }]))

    _intake, plan = to_task_plan(parsed, "给我看看")

    assert plan.steps[0].capability_id == "catalog.discover"
    assert plan.steps[0].objective_id == "discover"

