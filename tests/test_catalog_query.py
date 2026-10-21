from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from src import orchestrator, tools
from src.backend.db import get_engine, get_session_factory
from src.backend.seed import seed
from src.catalog_query import (
    CatalogFieldRef,
    CatalogQuerySpec,
    PredicateTruth,
    _catalog_data_fingerprint,
    _compare,
    compilation_coverage,
    query_catalog,
    render_catalog_result,
    verify_catalog_result,
)
from src.catalog_request_audit import review_catalog_request
from src.dialogue_understanding import BusinessObjective
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


def approved_audit(spec_payload, source_quote=None, objective_id="catalog_view"):
    spec = CatalogQuerySpec.model_validate(spec_payload, strict=True)
    requirements = spec.requirements()
    return {
        "verdict": "APPROVED",
        "requirements": [{
            "source_quote": source_quote or requirements[0].source_quote,
            "interpretation": "原始目录请求的全部要求均已进入受限查询 AST",
            "requirement_ids": [f"{objective_id}:{row.requirement_id}" for row in requirements],
            "status": "COVERED",
        }],
        "issues": [],
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
    result["artifact"]["selected_ids"] = result["artifact"]["selected_ids"][:4]
    result["artifact"]["delivered_ids"] = result["artifact"]["delivered_ids"][:4]
    result["artifact"]["delivered_count"] = 4
    result["artifact"]["delivery"]["delivered_count"] = 4
    result["artifact"]["delivery"]["complete"] = False
    result["artifact"]["result_data_fingerprint"] = _catalog_data_fingerprint(result["artifact"]["products"])

    with pytest.raises(AgentError, match="完整交付") as captured:
        verify_catalog_result(result, args)

    assert captured.value.code == "CATALOG_DELIVERY_INCOMPLETE"
    assert result["artifact"]["verification"]["status"] == "REJECTED"


def test_catalog_query_rejects_a_forged_compilation_coverage():
    args = query_args(grouped_catalog_spec())
    args["compilation_coverage"] = args["compilation_coverage"][:-1]

    result = query_catalog(**args)

    assert result == {"error": "目录查询的编译覆盖证明无效"}


def test_concept_filter_and_descending_sort_use_registered_codec():
    message = "至少 84 键的键盘按价格从高到低"
    spec = {
        "version": "catalog-query/v1",
        "scope": {"requirement_id": "scope", "source_quote": message, "kind": "CATALOG"},
        "availability": {"requirement_id": "availability", "source_quote": message, "scope": "ALL"},
        "filters": [
            {"requirement_id": "keyboard", "source_quote": "键盘", "field": {"core": "category"},
             "operator": "EQ", "value": "键盘"},
            {"requirement_id": "keys", "source_quote": "至少 84 键", "concept": {
                "schema_revision": "demo-catalog/product-concepts/v1",
                "concept_id": "keyboard.key_count", "operator": "gte", "canonical_value": 84,
            }},
        ],
        "partition": None,
        "ordering": [{"requirement_id": "price_desc", "source_quote": "价格从高到低",
                      "field": {"core": "price"}, "direction": "DESC"}],
        "selector": {"requirement_id": "all", "source_quote": message, "mode": "ALL", "include_ties": False},
        "metrics": [],
        "projection": {"requirement_id": "products", "source_quote": message,
                       "fields": [{"core": "product_id"}, {"core": "price"},
                                  {"concept_id": "keyboard.key_count"}]},
    }
    args = query_args(spec)

    result = query_catalog(**args)
    verify_catalog_result(result, args)

    rows = result["artifact"]["products"]
    assert rows
    assert all(row["category"] == "键盘" and row["values"]["keyboard.key_count"] >= 84 for row in rows)
    assert [row["price"] for row in rows] == sorted((row["price"] for row in rows), reverse=True)
    assert result["artifact"]["resolved_filters"][1]["field"]["concept_id"] == "keyboard.key_count"


def test_count_metric_is_computed_over_the_filtered_population():
    message = "一共有多少款耳机"
    spec = headphones_spec(message)
    spec["selector"]["source_quote"] = message
    spec["metrics"] = [{"requirement_id": "count", "source_quote": "多少款",
                        "function": "COUNT", "field": None, "alias": "product_count"}]
    args = query_args(spec)

    result = query_catalog(**args)
    verify_catalog_result(result, args)

    assert result["artifact"]["metrics"] == {"product_count": 20}
    assert render_catalog_result(result) == "已完成目录聚合：数量：20。"


def test_goal_verifier_rejects_result_order_tampering():
    args = query_args(grouped_catalog_spec())
    result = query_catalog(**args)
    rows = result["artifact"]["products"]
    rows[0], rows[1] = rows[1], rows[0]
    result["artifact"]["member_ids"][0:2] = [rows[0]["product_id"], rows[1]["product_id"]]
    result["artifact"]["groups"][0]["product_ids"][0:2] = [rows[0]["product_id"], rows[1]["product_id"]]

    with pytest.raises(AgentError) as captured:
        verify_catalog_result(result, args)

    assert captured.value.code == "CATALOG_GOAL_NOT_SATISFIED"


def test_goal_verifier_rejects_an_internally_consistent_non_minimum_selection():
    first_args = query_args(headphones_spec("耳机都有哪些"))
    first = query_catalog(**first_args)
    base = first["artifact"]
    args = query_args(cheapest_previous_spec("最便宜的是哪款"), base=base)
    result = query_catalog(**args)
    wrong = sorted(base["products"], key=lambda row: row["price"])[-1]
    artifact = result["artifact"]
    artifact["products"] = [wrong]
    artifact["member_ids"] = [wrong["product_id"]]
    artifact["selected_ids"] = [wrong["product_id"]]
    artifact["delivered_ids"] = [wrong["product_id"]]
    artifact["delivered_count"] = 1
    artifact["delivery"]["delivered_count"] = 1
    artifact["result_data_fingerprint"] = _catalog_data_fingerprint([wrong])

    with pytest.raises(AgentError) as captured:
        verify_catalog_result(result, args)

    assert captured.value.code == "CATALOG_GOAL_NOT_SATISFIED"
    assert artifact["verification"]["status"] == "REJECTED"


def test_goal_verifier_rejects_self_hashed_tampering_in_undelivered_population_rows():
    spec = grouped_catalog_spec()
    spec["partition"] = None
    spec["ordering"] = [{
        "requirement_id": "price_order", "source_quote": "价格升序",
        "field": {"core": "price"}, "direction": "ASC",
    }]
    spec["selector"] = {
        "requirement_id": "first_page", "source_quote": "第一页每页七款",
        "mode": "PAGE", "page": 1, "page_size": 7, "include_ties": False,
    }
    args = query_args(spec)
    result = query_catalog(**args)
    artifact = result["artifact"]
    assert artifact["population_products"][8]["product_id"] not in artifact["member_ids"]
    artifact["population_products"][8]["price"] = 0.01
    artifact["population_data_fingerprint"] = _catalog_data_fingerprint(artifact["population_products"])

    with pytest.raises(AgentError) as captured:
        verify_catalog_result(result, args)

    assert captured.value.code == "CATALOG_GOAL_NOT_SATISFIED"
    assert artifact["verification"]["status"] == "REJECTED"


def test_missing_attribute_is_unknown_and_cannot_satisfy_negative_filters():
    assert _compare(None, "NE", "静音轴") is PredicateTruth.UNKNOWN
    assert _compare(None, "NOT_CONTAINS", "主动") is PredicateTruth.UNKNOWN
    assert _compare("红轴", "NE", "静音轴") is PredicateTruth.TRUE

    message = "不是静音轴的商品"
    spec = grouped_catalog_spec()
    spec["scope"]["source_quote"] = message
    spec["availability"]["source_quote"] = message
    spec["partition"] = None
    spec["ordering"] = []
    spec["filters"] = [{
        "requirement_id": "not_silent",
        "source_quote": "不是静音轴",
        "concept": {
            "schema_revision": "demo-catalog/product-concepts/v1",
            "concept_id": "keyboard.switch_type",
            "operator": "ne",
            "canonical_value": "silent",
        },
    }]
    spec["selector"] = {
        "requirement_id": "all", "source_quote": message, "mode": "ALL", "include_ties": False,
    }
    spec["projection"]["source_quote"] = message
    args = query_args(spec)

    result = query_catalog(**args)
    verify_catalog_result(result, args)

    assert result["artifact"]["products"]
    assert all(row["category"] == "键盘" for row in result["artifact"]["products"])
    assert all(row["specs"]["轴体"] != "静音轴" for row in result["artifact"]["products"])


def test_previous_population_continues_pagination_without_repaginating_the_delivered_page():
    first_spec = grouped_catalog_spec()
    first_spec["partition"] = None
    first_spec["ordering"] = [{
        "requirement_id": "price_order", "source_quote": "价格升序",
        "field": {"core": "price"}, "direction": "ASC",
    }]
    first_spec["selector"] = {
        "requirement_id": "first_page", "source_quote": "第一页每页七款",
        "mode": "PAGE", "page": 1, "page_size": 7, "include_ties": False,
    }
    first_args = query_args(first_spec)
    first = query_catalog(**first_args)
    verify_catalog_result(first, first_args)

    second_spec = grouped_catalog_spec()
    second_spec["scope"] = {
        "requirement_id": "scope", "source_quote": "下一页",
        "kind": "PREVIOUS_RESULT_SET", "result_set_view": "POPULATION",
    }
    second_spec["availability"]["source_quote"] = "下一页"
    second_spec["partition"] = None
    second_spec["ordering"] = [{
        "requirement_id": "price_order", "source_quote": "下一页",
        "field": {"core": "price"}, "direction": "ASC",
    }]
    second_spec["selector"] = {
        "requirement_id": "next_page", "source_quote": "下一页",
        "mode": "PAGE", "page": 2, "page_size": 7, "include_ties": False,
    }
    second_spec["projection"]["source_quote"] = "下一页"
    second_args = query_args(second_spec, base=first["artifact"])

    second = query_catalog(**second_args)
    verify_catalog_result(second, second_args)

    expected_ids = first["artifact"]["population_ids"][7:14]
    assert second["artifact"]["member_ids"] == expected_ids
    assert second["artifact"]["source_result_set_view"] == "POPULATION"
    assert second["artifact"]["delivered_count"] == 7
    assert second["artifact"]["verification"]["status"] == "VERIFIED"


def test_query_protocol_rejects_ambiguous_or_untrusted_shapes():
    spec = grouped_catalog_spec()
    spec["partition"] = None
    spec["ordering"] = []
    spec["selector"] = {"requirement_id": "top", "source_quote": "前 3 个", "mode": "TOP_K",
                        "limit": 3, "include_ties": False}
    with pytest.raises(ValidationError, match="deterministic ordering"):
        CatalogQuerySpec.model_validate(spec, strict=True)
    with pytest.raises(ValidationError):
        CatalogFieldRef.model_validate({"core": "price", "raw_sql": "DROP TABLE products"}, strict=True)


def test_page_delivery_exposes_truthful_continuation_metadata():
    spec = grouped_catalog_spec()
    spec["partition"] = None
    spec["ordering"] = [{"requirement_id": "price_order", "source_quote": "价格升序",
                         "field": {"core": "price"}, "direction": "ASC"}]
    spec["selector"] = {"requirement_id": "page", "source_quote": "第二页每页七款", "mode": "PAGE",
                        "page": 2, "page_size": 7, "include_ties": False}
    args = query_args(spec)

    result = query_catalog(**args)
    _, delivery = verify_catalog_result(result, args)

    assert result["artifact"]["delivered_count"] == 7
    assert delivery == {
        "verdict": "SATISFIED", "failures": [], "mode": "PAGE", "complete": True,
        "population_complete": False, "has_more": True, "next_page": 3,
    }


@pytest.mark.parametrize(("field", "mode", "aggregate"), [
    ("price", "ARGMIN", min),
    ("price", "ARGMAX", max),
    ("stock", "ARGMIN", min),
    ("stock", "ARGMAX", max),
])
def test_extreme_selectors_are_verified_for_multiple_fields(field, mode, aggregate):
    message = "查询目录极值"
    spec = grouped_catalog_spec()
    spec["scope"]["source_quote"] = message
    spec["availability"]["source_quote"] = message
    spec["partition"] = None
    spec["ordering"] = []
    spec["selector"] = {"requirement_id": "extreme", "source_quote": message, "mode": mode,
                        "field": {"core": field}, "include_ties": True}
    spec["projection"] = {"requirement_id": "columns", "source_quote": message,
                          "fields": [{"core": "product_id"}, {"core": "name"}, {"core": field}]}
    args = query_args(spec)

    result = query_catalog(**args)
    verify_catalog_result(result, args)

    rows = result["artifact"]["products"]
    with tools._SessionLocal() as session:
        target = aggregate(getattr(product, field) for product in session.query(tools.Product).all())
    assert rows and all(row[field] == target for row in rows)


def test_semantic_catalog_request_uses_query_ast_instead_of_discovery(monkeypatch):
    message = "我想让你把所有商品，按照类目全部展示，每个类目下的商品按照价格从小到大排序"
    spec = grouped_catalog_spec()
    model = ScriptedModel(
        {
            "reason": "目录范围、分组、排序和完整性均需保留",
            "commands": [],
            "objectives": [{
                "id": "catalog_view",
                "kind": "CATALOG_QUERY",
                "source_quote": message,
                "catalog_query": spec,
            }],
        },
        approved_audit(spec, message),
    )
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert [entry["tool"] for entry in result["trace"]] == ["query_catalog"]
    assert result["artifacts"][0]["delivered_count"] == 60
    assert result["plan"]["verification"]["all_goals_satisfied"] is True
    assert any(event["type"] == "compilation_coverage" for event in result["events"])
    assert any(event["type"] == "catalog_goal_verified" for event in result["events"])
    assert any(event["type"] == "catalog_delivery_verified" for event in result["events"])
    assert result["catalog_request_audit"]["verdict"] == "APPROVED"
    assert any(event["type"] == "catalog_request_audited" for event in result["events"])


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
        approved_audit(headphones_spec(first_message), first_message, "headphones"),
        {"reason": "在上一结果集取价格最小值", "commands": [], "objectives": [{
            "id": "cheapest", "kind": "CATALOG_QUERY", "source_quote": second_message,
            "catalog_query": cheapest_previous_spec(second_message),
        }]},
        approved_audit(cheapest_previous_spec(second_message), second_message, "cheapest"),
    )
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)
    session_id = "catalog-followup"

    first = run_conversation(first_message, session_id, request_id="catalog-first", store=RunStore())
    first_minimum = min(first["artifacts"][0]["products"], key=lambda row: row["price"])
    # The next turn must operate on the persisted result snapshot, not silently
    # reinterpret the prior set against mutable current catalog values.
    with tools._SessionLocal() as session:
        product = session.get(tools.Product, first_minimum["product_id"])
        product.price = 9999.0
        session.commit()
    second = run_conversation(second_message, session_id, request_id="catalog-second", store=RunStore())

    assert first["artifacts"][0]["delivered_count"] == 20
    assert second["artifacts"][0]["source_result_set_id"] == first["artifacts"][0]["result_set_id"]
    selected = second["artifacts"][0]["products"]
    assert len(selected) == 1
    assert selected[0]["category"] == "耳机"
    assert selected[0]["product_id"] == first_minimum["product_id"]
    assert selected[0]["price"] == min(row["price"] for row in first["artifacts"][0]["products"])
    state = RunStore().conversation(session_id)["state"]
    assert state["catalog_result_set"]["result_set_id"] == second["artifacts"][0]["result_set_id"]
    assert state["catalog_result_set"]["source_run_id"] == second["run_id"]


def test_conversation_projection_restores_persisted_artifacts(monkeypatch):
    message = "耳机都有哪些"
    spec = headphones_spec(message)
    model = ScriptedModel(
        {"reason": "列出耳机", "commands": [], "objectives": [{
            "id": "headphones", "kind": "CATALOG_QUERY", "source_quote": message,
            "catalog_query": spec,
        }]},
        approved_audit(spec, message, "headphones"),
    )
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)
    session_id = "catalog-conversation"
    result = run_conversation(message, session_id, request_id="catalog-conversation-run", store=RunStore())

    conversation = RunStore().conversation(session_id)

    assistant = next(item for item in conversation["messages"] if item["role"] == "assistant")
    assert assistant["result"]["artifacts"] == result["artifacts"]


def test_rejected_catalog_artifact_cannot_replace_the_verified_conversation_scope(monkeypatch):
    message = "耳机都有哪些"
    spec = headphones_spec(message)
    model = ScriptedModel(
        {"reason": "列出耳机", "commands": [], "objectives": [{
            "id": "headphones", "kind": "CATALOG_QUERY", "source_quote": message,
            "catalog_query": spec,
        }]},
        approved_audit(spec, message, "headphones"),
    )
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)
    session_id = "catalog-rejected-scope"
    first = run_conversation(message, session_id, request_id="verified-catalog", store=RunStore())
    verified_scope = deepcopy(RunStore().conversation(session_id)["state"]["catalog_result_set"])
    rejected = deepcopy(first["artifacts"][0])
    rejected["result_set_id"] = "crs_rejected_scope"
    rejected["verification"] = {
        "status": "REJECTED",
        "error": {"code": "CATALOG_GOAL_NOT_SATISFIED", "message": "目录目标未满足", "retryable": False},
    }

    def rejected_runner(_request, **_kwargs):
        return {
            "reply": "目录核验失败。", "agent": "shopping", "route": "CATALOG",
            "status": "FAILED", "terminal_reason": "CATALOG_GOAL_NOT_SATISFIED",
            "error": rejected["verification"]["error"], "trace": [], "pending_action": None,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "artifacts": [rejected],
        }

    second = run_conversation(
        "给我一个核验失败的目录产物", session_id,
        request_id="rejected-catalog", store=RunStore(), runner=rejected_runner,
    )

    assert second["status"] == "FAILED"
    assert RunStore().conversation(session_id)["state"]["catalog_result_set"] == verified_scope


def test_independent_catalog_audit_repairs_an_omitted_order_before_execution(monkeypatch):
    message = "把所有商品按类目展示，每个类目按价格从小到大排序"
    incomplete = grouped_catalog_spec()
    incomplete["partition"]["source_quote"] = "按类目展示"
    incomplete["scope"]["source_quote"] = "所有商品"
    incomplete["availability"]["source_quote"] = "所有商品"
    incomplete["selector"]["source_quote"] = "所有商品"
    incomplete["projection"]["source_quote"] = "所有商品"
    incomplete["ordering"] = []
    complete = grouped_catalog_spec()
    complete["scope"]["source_quote"] = "所有商品"
    complete["availability"]["source_quote"] = "所有商品"
    complete["partition"]["source_quote"] = "按类目展示"
    complete["ordering"][0]["source_quote"] = "价格从小到大排序"
    complete["selector"]["source_quote"] = "所有商品"
    complete["projection"]["source_quote"] = "所有商品"
    model = ScriptedModel(
        {"reason": "初次理解遗漏排序", "commands": [], "objectives": [{
            "id": "catalog_view", "kind": "CATALOG_QUERY", "source_quote": message,
            "catalog_query": incomplete,
        }]},
        {
            "verdict": "REVISE",
            "requirements": [{
                "source_quote": "价格从小到大排序",
                "interpretation": "价格升序没有进入查询 AST",
                "requirement_ids": [],
                "status": "MISSING",
            }],
            "issues": ["补充价格升序要求"],
        },
        {"reason": "修复后保留排序", "commands": [], "objectives": [{
            "id": "catalog_view", "kind": "CATALOG_QUERY", "source_quote": message,
            "catalog_query": complete,
        }]},
        approved_audit(complete, message),
    )
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert [row["direction"] for row in result["artifacts"][0]["query_spec"]["ordering"]] == ["ASC"]
    assert sum(event["type"] == "catalog_request_audited" for event in result["events"]) == 2
    assert model.index == 4


def test_catalog_audit_normalizes_unique_ast_requirement_ids():
    message = "把所有商品按类目完整展示，每个类目内按库存从高到低排序"
    spec = grouped_catalog_spec()
    objective = BusinessObjective.model_validate({
        "id": "catalog_view",
        "kind": "CATALOG_QUERY",
        "source_quote": message,
        "catalog_query": spec,
    })
    raw_ids = [row.requirement_id for row in objective.catalog_query.requirements()]
    model = ScriptedModel({
        "verdict": "APPROVED",
        "requirements": [{
            "source_quote": message,
            "interpretation": "全部要求均已覆盖",
            "requirement_ids": raw_ids,
            "status": "COVERED",
        }],
        "issues": [],
    })

    audit = review_catalog_request(message, [], [objective], model=model)

    assert audit["requirements"][0]["requirement_ids"] == [
        f"catalog_view:{requirement_id}" for requirement_id in raw_ids
    ]
    assert audit["protocol_repairs"] == [
        {"received": requirement_id, "canonical": f"catalog_view:{requirement_id}"}
        for requirement_id in raw_ids
    ]


def test_catalog_audit_rejects_ambiguous_ast_requirement_id_aliases():
    message = "所有商品"
    spec = grouped_catalog_spec()
    objectives = [
        BusinessObjective.model_validate({
            "id": objective_id,
            "kind": "CATALOG_QUERY",
            "source_quote": message,
            "catalog_query": spec,
        })
        for objective_id in ("first", "second")
    ]
    model = ScriptedModel({
        "verdict": "REVISE",
        "requirements": [{
            "source_quote": message,
            "interpretation": "无法唯一定位范围要求",
            "requirement_ids": ["scope"],
            "status": "MISINTERPRETED",
        }],
        "issues": ["requirement_id 不明确"],
    })

    with pytest.raises(AgentError, match="unknown or ambiguous") as captured:
        review_catalog_request(message, [], objectives, model=model, allow_protocol_repair=False)

    assert captured.value.code == "CATALOG_AUDIT_FORMAT_ERROR"


def test_catalog_audit_repairs_an_irrelevant_history_row_once(monkeypatch):
    message = "我想让你把所有商品，按照类目全部展示，每个类目下的商品按照价格从小到大排序"
    prior_message = "推荐一个适合出差的降噪耳机"
    spec = grouped_catalog_spec()
    requirement_ids = [
        f"catalog_view:{row.requirement_id}"
        for row in CatalogQuerySpec.model_validate(spec, strict=True).requirements()
    ]
    understanding = {
        "reason": "完整保留范围、分组和排序要求",
        "commands": [],
        "objectives": [{
            "id": "catalog_view",
            "kind": "CATALOG_QUERY",
            "source_quote": message,
            "catalog_query": spec,
        }],
    }
    malformed_audit = {
        "verdict": "APPROVED",
        "requirements": [
            {
                "source_quote": message,
                "source_turn": "LATEST",
                "interpretation": "本轮目录要求全部覆盖",
                "requirement_ids": requirement_ids,
                "status": "COVERED",
            },
            {
                "source_quote": prior_message,
                "source_turn": "HISTORY",
                "interpretation": "上一轮推荐目标不需要继承",
                "requirement_ids": [],
                "status": "COVERED",
            },
        ],
        "issues": [],
    }
    repaired_audit = {
        **malformed_audit,
        "requirements": [
            malformed_audit["requirements"][0],
            {**malformed_audit["requirements"][1], "status": "NOT_APPLICABLE"},
        ],
    }
    model = ScriptedModel(understanding, malformed_audit, repaired_audit)
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message, [HumanMessage(content=prior_message)])

    assert result["status"] == "SUCCEEDED"
    assert result["catalog_request_audit"]["protocol_attempts"] == 2
    assert result["catalog_request_audit"]["requirements"][-1]["status"] == "NOT_APPLICABLE"
    assert sum(event["type"] == "catalog_audit_protocol_retry" for event in result["events"]) == 1
    assert sum(event["type"] == "catalog_audit_protocol_repaired" for event in result["events"]) == 1
    assert model.index == 3


def test_catalog_audit_cannot_dismiss_the_latest_request_as_not_applicable():
    message = "所有商品"
    spec = grouped_catalog_spec()
    objective = BusinessObjective.model_validate({
        "id": "catalog_view",
        "kind": "CATALOG_QUERY",
        "source_quote": message,
        "catalog_query": spec,
    })
    model = ScriptedModel({
        "verdict": "APPROVED",
        "requirements": [{
            "source_quote": message,
            "source_turn": "LATEST",
            "interpretation": "错误地忽略本轮请求",
            "requirement_ids": [],
            "status": "NOT_APPLICABLE",
        }],
        "issues": [],
    })

    with pytest.raises(AgentError, match="Only history may be NOT_APPLICABLE") as captured:
        review_catalog_request(message, [], [objective], model=model, allow_protocol_repair=False)

    assert captured.value.code == "CATALOG_AUDIT_FORMAT_ERROR"
