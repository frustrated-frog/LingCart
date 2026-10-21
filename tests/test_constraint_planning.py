from copy import deepcopy
import hashlib
import json

import pytest
from langchain_core.messages import AIMessage

from src import orchestrator, tools
from src.backend.db import Product, get_engine, get_session_factory
from src.backend.seed import seed
from src.constraint_planning import (
    CandidateValue,
    ConstraintPlanSpec,
    DomainProviderRegistration,
    DomainProviderRegistry,
    DomainResolution,
    compile_constraint_plan,
    compilation_coverage,
    solve_constraint_plan,
    solve_finite_domain,
    use_domain_provider_registry,
    verify_constraint_plan_result,
)
from src.errors import AgentError


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


def catalog_domain(category: str, quote: str, prefix: str) -> dict:
    return {
        "requirement_id": f"{prefix}_domain",
        "source_quote": quote,
        "provider": "CATALOG",
        "parameters": {"query": {
            "version": "catalog-query/v1",
            "scope": {"requirement_id": f"{prefix}_scope", "source_quote": quote, "kind": "CATALOG"},
            "availability": {"requirement_id": f"{prefix}_available", "source_quote": quote, "scope": "AVAILABLE"},
            "filters": [{
                "requirement_id": f"{prefix}_category", "source_quote": quote,
                "field": {"core": "category"}, "operator": "EQ", "value": category,
            }],
            "partition": None,
            "ordering": [],
            "selector": {"requirement_id": f"{prefix}_all", "source_quote": quote,
                         "mode": "ALL", "include_ties": False},
            "metrics": [],
            "projection": {"requirement_id": f"{prefix}_fields", "source_quote": quote, "fields": [
                {"core": "product_id"}, {"core": "name"}, {"core": "category"},
                {"core": "price"}, {"core": "stock"}, {"core": "summary"},
            ]},
        }},
    }


def shared_budget_spec(*, budget: float = 400) -> dict:
    return {
        "version": "constraint-plan/v1",
        "goals": [
            {"requirement_id": "headphones_goal", "source_quote": "一个耳机", "goal_id": "headphones",
             "quantity": 1, "domain": catalog_domain("耳机", "一个耳机", "headphones")},
            {"requirement_id": "keyboard_goal", "source_quote": "一个键盘", "goal_id": "keyboard",
             "quantity": 1, "domain": catalog_domain("键盘", "一个键盘", "keyboard")},
        ],
        "constraints": [{
            "requirement_id": "shared_budget", "source_quote": f"{budget:g}元预算", "kind": "AGGREGATE",
            "scope_goal_ids": ["headphones", "keyboard"], "attribute": "price", "function": "SUM",
            "operator": "LTE", "value": budget,
        }],
        "optimization": None,
    }


def approved_audit(spec: dict, message: str) -> dict:
    parsed = ConstraintPlanSpec.model_validate(spec, strict=True)
    return {
        "verdict": "APPROVED",
        "requirements": [{
            "source_quote": message,
            "source_turn": "LATEST",
            "interpretation": "两个商品目标及共享总预算均已进入同一个约束图",
            "requirement_ids": [f"bundle:{row['requirement_id']}" for row in compilation_coverage(parsed)],
            "status": "COVERED",
        }],
        "issues": [],
    }


def test_shared_budget_is_solved_and_independently_verified():
    arguments = compile_constraint_plan(shared_budget_spec())

    result = solve_constraint_plan(**arguments)
    verification = verify_constraint_plan_result(result, arguments)

    assert result["feasibility"] == "FEASIBLE"
    assert verification["status"] == "VERIFIED"
    assert verification["global_goal"] is True
    assert result["artifact"]["search"]["complete"] is True
    for solution in result["solutions"]:
        assert len(solution["selections"]) == 2
        assert solution["constraint_checks"][0]["satisfied"] is True
        assert solution["constraint_checks"][0]["observed"] <= 400


def test_infeasible_claim_requires_complete_search():
    arguments = compile_constraint_plan(shared_budget_spec(budget=1))

    result = solve_constraint_plan(**arguments)
    verification = verify_constraint_plan_result(result, arguments)

    assert result["feasibility"] == "INFEASIBLE"
    assert result["solutions"] == []
    assert result["artifact"]["search"]["complete"] is True
    assert verification["verdict"] == "PROVEN_INFEASIBLE"


def test_global_verifier_rejects_tampered_price_or_constraint_evidence():
    arguments = compile_constraint_plan(shared_budget_spec())
    result = solve_constraint_plan(**arguments)
    tampered = deepcopy(result)
    tampered["artifact"]["solutions"][0]["constraint_checks"][0]["observed"] = 1

    with pytest.raises(AgentError) as error:
        verify_constraint_plan_result(tampered, arguments)

    assert error.value.code == "CONSTRAINT_RESULT_EVIDENCE_MISMATCH"
    assert tampered["artifact"]["verification"]["status"] == "REJECTED"


def test_catalog_changes_invalidate_previous_constraint_evidence(database):
    arguments = compile_constraint_plan(shared_budget_spec())
    result = solve_constraint_plan(**arguments)
    with get_session_factory(database)() as session:
        product = session.get(Product, "P1108S")
        product.price += 10
        session.commit()

    with pytest.raises(AgentError) as error:
        verify_constraint_plan_result(result, arguments)

    assert error.value.code == "CONSTRAINT_RESULT_EVIDENCE_MISMATCH"


def test_solver_is_domain_agnostic_and_accepts_a_registered_non_catalog_provider():
    values = {
        "origin": (CandidateValue("a", "甲", {"cost": 3}), CandidateValue("b", "乙", {"cost": 7})),
        "destination": (CandidateValue("x", "丙", {"cost": 4}), CandidateValue("y", "丁", {"cost": 8})),
    }
    holder = {}

    def validate(parameters):
        if set(parameters) != {"name"} or parameters["name"] not in values:
            raise ValueError("unknown fixture domain")
        return dict(parameters)

    def resolve(parameters):
        registration = holder["registry"].registration("FINITE_FIXTURE")
        return DomainResolution("FINITE_FIXTURE", registration.fingerprint,
                                values[parameters["name"]], {"source": "TEST_FIXTURE"})

    registration = DomainProviderRegistration(
        "FINITE_FIXTURE", "通用有限域测试", {"name": "string"}, validate, resolve,
        lambda _parameters, _sources: None,
    )
    registry = DomainProviderRegistry((registration,))
    holder["registry"] = registry
    spec = ConstraintPlanSpec.model_validate({
        "version": "constraint-plan/v1",
        "goals": [
            {"requirement_id": "origin_goal", "source_quote": "起点", "goal_id": "origin", "quantity": 1,
             "domain": {"requirement_id": "origin_domain", "source_quote": "起点",
                        "provider": "FINITE_FIXTURE", "parameters": {"name": "origin"}}},
            {"requirement_id": "destination_goal", "source_quote": "终点", "goal_id": "destination", "quantity": 1,
             "domain": {"requirement_id": "destination_domain", "source_quote": "终点",
                        "provider": "FINITE_FIXTURE", "parameters": {"name": "destination"}}},
        ],
        "constraints": [{"requirement_id": "total", "source_quote": "总成本不超过10", "kind": "AGGREGATE",
                         "scope_goal_ids": ["origin", "destination"], "attribute": "cost",
                         "function": "SUM", "operator": "LTE", "value": 10}],
        "optimization": None,
    }, strict=True)
    with use_domain_provider_registry(registry):
        domains = {goal.goal_id: registry.resolve(goal.domain) for goal in spec.goals}
        result = solve_finite_domain(spec, domains)

    assert result["feasibility"] == "FEASIBLE"
    assert all(row["constraint_checks"][0]["observed"] <= 10 for row in result["solutions"])


def test_search_budget_never_turns_incomplete_search_into_false_infeasible():
    candidates = tuple(CandidateValue(str(index), str(index), {"cost": index}) for index in range(5))
    domains = {
        "left": DomainResolution("TEST", "fp", candidates, {}),
        "right": DomainResolution("TEST", "fp", candidates, {}),
    }
    spec = ConstraintPlanSpec.model_validate({
        "version": "constraint-plan/v1",
        "goals": [
            {"requirement_id": "left_goal", "source_quote": "左", "goal_id": "left", "quantity": 1,
             "domain": {"requirement_id": "left_domain", "source_quote": "左", "provider": "TEST", "parameters": {}}},
            {"requirement_id": "right_goal", "source_quote": "右", "goal_id": "right", "quantity": 1,
             "domain": {"requirement_id": "right_domain", "source_quote": "右", "provider": "TEST", "parameters": {}}},
        ],
        "constraints": [{"requirement_id": "impossible", "source_quote": "总成本等于999", "kind": "AGGREGATE",
                         "scope_goal_ids": ["left", "right"], "attribute": "cost",
                         "function": "SUM", "operator": "EQ", "value": 999}],
        "optimization": None,
    }, strict=True)

    with pytest.raises(AgentError) as error:
        solve_finite_domain(spec, domains, max_search_nodes=1)

    assert error.value.code == "CONSTRAINT_SEARCH_INCONCLUSIVE"


def test_shared_budget_request_runs_through_registered_capability(monkeypatch):
    message = "我有400元预算，想买一个耳机一个键盘，有什么推荐吗"
    spec = shared_budget_spec()
    dialogue = {
        "reason": "共享预算需要约束规划",
        "commands": [],
        "objectives": [{
            "id": "bundle", "kind": "CONSTRAINT_PLAN", "source_quote": message,
            "target_type": "constraint_solution", "evidence_requirement": "VERIFIED_CONSTRAINT_SOLUTION",
            "constraint_plan": spec,
        }],
    }
    model = ScriptedModel(dialogue, approved_audit(spec, message))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "SUCCEEDED"
    assert result["terminal_reason"] == "COMPLETED"
    assert result["trace"][0]["tool"] == "solve_constraint_plan"
    assert result["artifacts"][0]["artifact_type"] == "constraint_plan_result"
    assert result["artifacts"][0]["verification"]["status"] == "VERIFIED"
    assert result["plan"]["verification"]["all_goals_satisfied"] is True
    assert any(event["type"] == "global_goal_verified" for event in result["events"])
    assert model.index == 2
