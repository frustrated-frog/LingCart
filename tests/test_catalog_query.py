import json

import pytest
from langchain_core.messages import AIMessage

from src import orchestrator, tools
from src.backend.db import get_engine, get_session_factory
from src.backend.seed import seed
from src.catalog_query import (
    CatalogQuerySpec,
    compilation_coverage,
    query_catalog,
    verify_catalog_result,
)
from src.durable import run_conversation
from src.errors import AgentError
from src.persistence import RunStore


class ScriptedModel:
    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.index = 0

    def json_mode(self):
        return self

    def invoke(self, _prompt):
        payload = self.payloads[self.index]
        self.index += 1
        return AIMessage(content=json.dumps(payload, ensure_ascii=False))


@pytest.fixture(autouse=True)
def database():
    engine = get_engine("sqlite:///:memory:")
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
    return engine


def grouped_catalog_spec():
    return {
        "version": "catalog-query/v1",
        "scope": {"requirement_id": "scope", "source_quote": "所有商品", "kind": "CATALOG"},
        "availability": {"requirement_id": "availability", "source_quote": "所有商品", "scope": "ALL"},
        "filters": [],
        "partition": {
            "requirement_id": "partition",
            "source_quote": "按照类目全部展示",
            "field": {"core": "category"},
        },
        "ordering": [{
            "requirement_id": "price_order",
            "source_quote": "价格从小到大排序",
            "field": {"core": "price"},
            "direction": "ASC",
        }],
        "selector": {
            "requirement_id": "complete",
            "source_quote": "全部展示",
            "mode": "ALL",
            "include_ties": False,
        },
        "metrics": [],
        "projection": {
            "requirement_id": "columns",
            "source_quote": "所有商品",
            "fields": [
                {"core": "product_id"}, {"core": "name"}, {"core": "category"},
                {"core": "price"}, {"core": "summary"},
            ],
        },
    }


def query_args(spec_payload, *, base=None):
    spec = CatalogQuerySpec.model_validate(spec_payload, strict=True)
    return {
        "spec": spec.model_dump(mode="json"),
        "base_result_set": base,
        "compilation_coverage": compilation_coverage(spec),
    }


def test_complete_grouped_query_preserves_every_requirement():
    args = query_args(grouped_catalog_spec())

    result = query_catalog(**args)
    goal, delivery = verify_catalog_result(result, args)

    artifact = result["artifact"]
    assert artifact["matched_count"] == artifact["delivered_count"] == 60
    assert [(group["key"], group["count"]) for group in artifact["groups"]] == [
        ("扩展坞", 20), ("耳机", 20), ("键盘", 20),
    ]
    for group in artifact["groups"]:
        allowed = set(group["product_ids"])
        prices = [row["price"] for row in artifact["products"] if row["product_id"] in allowed]
        assert prices == sorted(prices)
    assert {row["requirement_id"] for row in artifact["compilation_coverage"]} == {
        "scope", "availability", "partition", "price_order", "complete", "columns",
    }
    assert goal["verdict"] == delivery["verdict"] == "SATISFIED"


def test_delivery_verifier_rejects_a_truncated_complete_result():
    args = query_args(grouped_catalog_spec())
    result = query_catalog(**args)
    result["artifact"]["products"] = result["artifact"]["products"][:4]
    result["artifact"]["member_ids"] = result["artifact"]["member_ids"][:4]
    result["artifact"]["delivered_count"] = 4
    result["artifact"]["delivery"]["delivered_count"] = 4
    result["artifact"]["delivery"]["complete"] = False

    with pytest.raises(AgentError, match="完整交付") as captured:
        verify_catalog_result(result, args)

    assert captured.value.code == "CATALOG_DELIVERY_INCOMPLETE"


def test_catalog_query_rejects_a_forged_compilation_coverage():
    args = query_args(grouped_catalog_spec())
    args["compilation_coverage"] = args["compilation_coverage"][:-1]

    result = query_catalog(**args)

    assert result == {"error": "目录查询的编译覆盖证明无效"}


def test_semantic_catalog_request_uses_query_ast_instead_of_discovery(monkeypatch):
    message = "我想让你把所有商品，按照类目全部展示，每个类目下的商品按照价格从小到大排序"
    model = ScriptedModel({
        "reason": "目录范围、分组、排序和完整性均需保留",
        "commands": [],
        "objectives": [{
            "id": "catalog_view",
            "kind": "CATALOG_QUERY",
            "source_quote": message,
            "catalog_query": grouped_catalog_spec(),
        }],
    })
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert [entry["tool"] for entry in result["trace"]] == ["query_catalog"]
    assert result["artifacts"][0]["delivered_count"] == 60
    assert result["plan"]["verification"]["all_goals_satisfied"] is True
    assert any(event["type"] == "compilation_coverage" for event in result["events"])
    assert any(event["type"] == "catalog_goal_verified" for event in result["events"])
    assert any(event["type"] == "catalog_delivery_verified" for event in result["events"])


def headphones_spec(message):
    return {
        "version": "catalog-query/v1",
        "scope": {"requirement_id": "scope", "source_quote": message, "kind": "CATALOG"},
        "availability": {"requirement_id": "availability", "source_quote": message, "scope": "ALL"},
        "filters": [{
            "requirement_id": "headphones",
            "source_quote": "耳机",
            "field": {"core": "category"},
            "operator": "EQ",
            "value": "耳机",
        }],
        "partition": None,
        "ordering": [],
        "selector": {"requirement_id": "all", "source_quote": "都有哪些", "mode": "ALL", "include_ties": False},
        "metrics": [],
        "projection": {
            "requirement_id": "products",
            "source_quote": message,
            "fields": [{"core": "product_id"}, {"core": "name"}, {"core": "price"}],
        },
    }


def cheapest_previous_spec(message):
    return {
        "version": "catalog-query/v1",
        "scope": {"requirement_id": "scope", "source_quote": message, "kind": "PREVIOUS_RESULT_SET"},
        "availability": {"requirement_id": "availability", "source_quote": message, "scope": "ALL"},
        "filters": [],
        "partition": None,
        "ordering": [],
        "selector": {
            "requirement_id": "cheapest",
            "source_quote": "最便宜",
            "mode": "ARGMIN",
            "field": {"core": "price"},
            "include_ties": False,
        },
        "metrics": [],
        "projection": {
            "requirement_id": "which",
            "source_quote": "哪款",
            "fields": [{"core": "product_id"}, {"core": "name"}, {"core": "price"}],
        },
    }


def test_followup_argmin_is_evaluated_over_persisted_previous_result_set(monkeypatch):
    first_message = "耳机都有哪些"
    second_message = "最便宜的是哪款"
    model = ScriptedModel(
        {"reason": "列出耳机", "commands": [], "objectives": [{
            "id": "headphones", "kind": "CATALOG_QUERY", "source_quote": first_message,
            "catalog_query": headphones_spec(first_message),
        }]},
        {"reason": "在上一结果集取价格最小值", "commands": [], "objectives": [{
            "id": "cheapest", "kind": "CATALOG_QUERY", "source_quote": second_message,
            "catalog_query": cheapest_previous_spec(second_message),
        }]},
    )
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)
    session_id = "catalog-followup"

    first = run_conversation(first_message, session_id, request_id="catalog-first", store=RunStore())
    second = run_conversation(second_message, session_id, request_id="catalog-second", store=RunStore())

    assert first["artifacts"][0]["delivered_count"] == 20
    assert second["artifacts"][0]["source_result_set_id"] == first["artifacts"][0]["result_set_id"]
    selected = second["artifacts"][0]["products"]
    assert len(selected) == 1
    assert selected[0]["category"] == "耳机"
    assert selected[0]["price"] == min(row["price"] for row in first["artifacts"][0]["products"])
    state = RunStore().conversation(session_id)["state"]
    assert state["catalog_result_set"]["result_set_id"] == second["artifacts"][0]["result_set_id"]


def test_conversation_projection_restores_persisted_artifacts(monkeypatch):
    message = "耳机都有哪些"
    model = ScriptedModel({"reason": "列出耳机", "commands": [], "objectives": [{
        "id": "headphones", "kind": "CATALOG_QUERY", "source_quote": message,
        "catalog_query": headphones_spec(message),
    }]})
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)
    session_id = "catalog-conversation"
    result = run_conversation(message, session_id, request_id="catalog-conversation-run", store=RunStore())

    conversation = RunStore().conversation(session_id)

    assistant = next(item for item in conversation["messages"] if item["role"] == "assistant")
    assert assistant["result"]["artifacts"] == result["artifacts"]
