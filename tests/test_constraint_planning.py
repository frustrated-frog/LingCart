from copy import deepcopy
import json
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src import orchestrator, tools
from src.backend.db import Product, get_engine, get_session_factory
from src.backend.seed import seed
from src.constraint_planning import (
    CandidateValue,
    ConstraintPlanSpec,
    DomainProviderRegistration,
    DomainProviderRegistry,
    DomainResolution,
    ResourceClaim,
    compile_constraint_plan,
    compilation_coverage,
    constraint_plan_outcome,
    solve_constraint_plan,
    solve_finite_domain,
    use_domain_provider_registry,
    validate_constraint_plan_grounding,
    verify_constraint_plan_result,
)
from src.dialogue_understanding import TurnUnderstanding, understand_turn
from src.errors import AgentError
from src.durable import run_conversation
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


def test_chinese_quantity_followup_and_empty_optimization_are_canonicalized():
    current = "两个耳机一个键盘"
    history = [HumanMessage(content="我有400元预算，想选几样东西")]
    spec = shared_budget_spec()
    spec["goals"][0].update(source_quote="两个耳机", quantity=2)
    spec["goals"][0]["domain"] = catalog_domain("耳机", "耳机", "headphones")
    spec["optimization"] = []
    dialogue = {
        "reason": "按上一轮总预算搭配本轮明确数量",
        "commands": [],
        "objectives": [{
            "id": "bundle",
            "kind": "CONSTRAINT_PLAN",
            "source_quote": current,
            "constraint_plan": spec,
        }],
    }

    understanding = understand_turn(
        current,
        history,
        None,
        model=ScriptedModel(dialogue),
    )

    plan = understanding.objectives[0].constraint_plan
    assert plan is not None
    assert plan.optimization is None
    assert [goal.quantity for goal in plan.goals] == [2, 1]
    validate_constraint_plan_grounding(plan, [current, history[0].content])


def test_persisted_clarification_repairs_dropped_shared_budget_before_execution(monkeypatch):
    first = "400元帮我选几样东西"
    current = "两个耳机一个键盘"
    clarification = {
        "reason": "缺少品类和数量",
        "commands": [{
            "kind": "CLARIFY",
            "source_quote": first,
            "message": "请告诉我需要哪些品类，以及每个品类各要几件。",
        }],
        "objectives": [],
    }
    dropped_budget = {
        "reason": "错误地拆成两个独立探索目标",
        "commands": [],
        "objectives": [
            {"id": "headphones", "kind": "PRODUCT_DISCOVERY", "source_quote": "两个耳机",
             "category": "耳机", "quantity": 2},
            {"id": "keyboard", "kind": "PRODUCT_DISCOVERY", "source_quote": "一个键盘",
             "category": "键盘", "quantity": 1},
        ],
    }
    spec = shared_budget_spec()
    spec["goals"][0].update(source_quote="两个耳机", quantity=2)
    spec["goals"][0]["domain"] = catalog_domain("耳机", "耳机", "headphones")
    spec["goals"][1].update(source_quote="一个键盘", quantity=1)
    spec["goals"][1]["domain"] = catalog_domain("键盘", "键盘", "keyboard")
    spec["constraints"][0].update(source_quote="400元", value=400)
    repaired = {
        "reason": "保留上一轮共享预算并合并本轮数量",
        "commands": [],
        "objectives": [{
            "id": "bundle", "kind": "CONSTRAINT_PLAN", "source_quote": current,
            "constraint_plan": spec,
        }],
    }
    model = ScriptedModel(clarification, dropped_budget, repaired, approved_audit(spec, current))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)
    session_id = f"constraint-followup-{uuid.uuid4().hex}"

    opened = run_conversation(first, session_id, request_id="request-1", store=RunStore())
    result = run_conversation(current, session_id, request_id="request-2", store=RunStore())

    assert opened["status"] == "WAITING_CLARIFICATION"
    assert result["status"] == "SUCCEEDED"
    assert result["trace"][0]["tool"] == "solve_constraint_plan"
    assert result["trace"][0]["args"]["spec"]["constraints"][0]["value"] == 400
    assert [goal["quantity"] for goal in result["trace"][0]["args"]["spec"]["goals"]] == [2, 1]
    assert any(event["type"] == "shared_constraint_followup_guarded" for event in result["events"])
    assert "pending_dialogue_clarification" not in RunStore().conversation(session_id)["state"]
    assert model.index == 4


def test_persisted_shared_budget_guard_repairs_redundant_model_clarification(monkeypatch):
    first = "400元帮我选几样东西"
    current = "两个耳机一个键盘"
    runtime = type("Runtime", (), {
        "snapshot": {"conversation": {"pending_dialogue_clarification": {
            "source_run_id": "run-1",
            "source_request": first,
            "question": "请告诉我品类和数量。",
            "terminal_reason": "MODEL_CLARIFICATION",
        }}},
    })()
    monkeypatch.setattr(orchestrator, "current_runtime", lambda: runtime)
    understanding = TurnUnderstanding.model_validate({
        "reason": "重复追问可选偏好",
        "commands": [{
            "kind": "CLARIFY",
            "source_quote": current,
            "message": "请补充耳机和键盘偏好。",
        }],
        "objectives": [],
    })

    feedback = orchestrator._shared_constraint_followup_feedback(current, understanding)

    assert feedback is not None
    assert feedback["requirements"][0]["source_turn"] == "HISTORY"
    assert "400 元" in feedback["requirements"][0]["interpretation"]


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


def test_explicit_optimization_proves_the_true_optimum_beyond_the_storage_limit():
    candidates = tuple(
        CandidateValue(f"candidate-{index:02d}", str(index), {"score": index})
        for index in range(71)
    )
    domains = {
        "left": DomainResolution("TEST", "fp", candidates, {}, ("score",)),
        "right": DomainResolution("TEST", "fp", candidates, {}, ("score",)),
    }
    spec = ConstraintPlanSpec.model_validate({
        "version": "constraint-plan/v1",
        "goals": [
            {"requirement_id": "left_goal", "source_quote": "左", "goal_id": "left", "quantity": 1,
             "domain": {"requirement_id": "left_domain", "source_quote": "左", "provider": "TEST", "parameters": {}}},
            {"requirement_id": "right_goal", "source_quote": "右", "goal_id": "right", "quantity": 1,
             "domain": {"requirement_id": "right_domain", "source_quote": "右", "provider": "TEST", "parameters": {}}},
        ],
        "constraints": [],
        "optimization": {
            "requirement_id": "max_score",
            "source_quote": "总分最高",
            "direction": "MAX",
            "scope_goal_ids": ["left", "right"],
            "attribute": "score",
            "function": "SUM",
        },
    }, strict=True)

    result = solve_finite_domain(spec, domains)

    assert result["search"]["complete"] is True
    assert result["search"]["feasible_count"] == 71 * 71
    assert result["search"]["feasible_count_exact"] is True
    assert result["solutions"][0]["optimization_value"] == 140
    assert result["optimality"]["status"] == "PROVEN_OPTIMAL"
    assert result["optimality"]["objective_value"] == 140
    assert constraint_plan_outcome({"artifact": result}, {}) == "SATISFIED"


def test_incomplete_optimization_only_reports_the_best_known_solution():
    candidates = tuple(
        CandidateValue(f"candidate-{index}", str(index), {"score": index})
        for index in range(3)
    )
    domains = {
        "left": DomainResolution("TEST", "fp", candidates, {}, ("score",)),
        "right": DomainResolution("TEST", "fp", candidates, {}, ("score",)),
    }
    spec = ConstraintPlanSpec.model_validate({
        "version": "constraint-plan/v1",
        "goals": [
            {"requirement_id": "left_goal", "source_quote": "左", "goal_id": "left", "quantity": 1,
             "domain": {"requirement_id": "left_domain", "source_quote": "左", "provider": "TEST", "parameters": {}}},
            {"requirement_id": "right_goal", "source_quote": "右", "goal_id": "right", "quantity": 1,
             "domain": {"requirement_id": "right_domain", "source_quote": "右", "provider": "TEST", "parameters": {}}},
        ],
        "constraints": [],
        "optimization": {
            "requirement_id": "max_score",
            "source_quote": "总分最高",
            "direction": "MAX",
            "scope_goal_ids": ["left", "right"],
            "attribute": "score",
            "function": "SUM",
        },
    }, strict=True)

    result = solve_finite_domain(spec, domains, max_search_nodes=4)

    assert result["feasibility"] == "FEASIBLE"
    assert result["search"]["complete"] is False
    assert result["search"]["feasible_count"] == 4
    assert result["search"]["feasible_count_exact"] is False
    assert result["optimality"]["status"] == "BEST_KNOWN"
    assert result["optimality"]["objective_value"] == result["solutions"][0]["optimization_value"]
    assert constraint_plan_outcome({"artifact": result}, {}) == "UNAVAILABLE_EVIDENCE"


def test_resource_capacity_is_enforced_across_overlapping_goal_domains():
    shared_left = CandidateValue(
        "left-shared",
        "共享设备",
        {"cost": 1},
        resource_claims=(ResourceClaim("device:shared", capacity=1),),
    )
    shared_right = CandidateValue(
        "right-shared",
        "共享设备的另一域表示",
        {"cost": 1},
        resource_claims=(ResourceClaim("device:shared", capacity=1),),
    )
    independent = CandidateValue(
        "right-independent",
        "独立设备",
        {"cost": 2},
        resource_claims=(ResourceClaim("device:independent", capacity=1),),
    )
    domains = {
        "left": DomainResolution("LEFT_PROVIDER", "left-fp", (shared_left,), {}, ("cost",)),
        "right": DomainResolution(
            "RIGHT_PROVIDER", "right-fp", (shared_right, independent), {}, ("cost",)
        ),
    }
    spec = ConstraintPlanSpec.model_validate({
        "version": "constraint-plan/v1",
        "goals": [
            {"requirement_id": "left_goal", "source_quote": "左", "goal_id": "left", "quantity": 1,
             "domain": {"requirement_id": "left_domain", "source_quote": "左", "provider": "TEST", "parameters": {}}},
            {"requirement_id": "right_goal", "source_quote": "右", "goal_id": "right", "quantity": 1,
             "domain": {"requirement_id": "right_domain", "source_quote": "右", "provider": "TEST", "parameters": {}}},
        ],
        "constraints": [],
        "optimization": None,
    }, strict=True)

    result = solve_finite_domain(spec, domains)

    assert result["feasibility"] == "FEASIBLE"
    assert result["search"]["feasible_count"] == 1
    selected = {
        candidate["candidate_id"]
        for selection in result["solutions"][0]["selections"]
        for candidate in selection["candidates"]
    }
    assert selected == {"left-shared", "right-independent"}
    assert all(check["satisfied"] for check in result["solutions"][0]["resource_checks"])
    assert next(
        check for check in result["solutions"][0]["resource_checks"]
        if check["resource_id"] == "device:shared"
    )["used"] == 1


def test_inconsistent_capacity_for_the_same_resource_is_rejected():
    domains = {
        "left": DomainResolution(
            "TEST", "fp", (
                CandidateValue("a", "甲", {}, resource_claims=(ResourceClaim("shared", capacity=1),)),
            ), {},
        ),
        "right": DomainResolution(
            "TEST", "fp", (
                CandidateValue("b", "乙", {}, resource_claims=(ResourceClaim("shared", capacity=2),)),
            ), {},
        ),
    }
    spec = ConstraintPlanSpec.model_validate({
        "version": "constraint-plan/v1",
        "goals": [
            {"requirement_id": "left_goal", "source_quote": "左", "goal_id": "left", "quantity": 1,
             "domain": {"requirement_id": "left_domain", "source_quote": "左", "provider": "TEST", "parameters": {}}},
            {"requirement_id": "right_goal", "source_quote": "右", "goal_id": "right", "quantity": 1,
             "domain": {"requirement_id": "right_domain", "source_quote": "右", "provider": "TEST", "parameters": {}}},
        ],
        "constraints": [],
        "optimization": None,
    }, strict=True)

    with pytest.raises(AgentError) as error:
        solve_finite_domain(spec, domains)

    assert error.value.code == "INCONSISTENT_RESOURCE_CAPACITY"


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

    session_id = f"constraint-{uuid.uuid4().hex}"
    result = run_conversation(message, session_id, request_id=f"request-{uuid.uuid4().hex}", store=RunStore())

    assert result["status"] == "SUCCEEDED"
    assert result["terminal_reason"] == "COMPLETED"
    assert result["trace"][0]["tool"] == "solve_constraint_plan"
    assert result["artifacts"][0]["artifact_type"] == "constraint_plan_result"
    assert result["artifacts"][0]["verification"]["status"] == "VERIFIED"
    assert result["plan"]["verification"]["all_goals_satisfied"] is True
    assert any(event["type"] == "global_goal_verified" for event in result["events"])
    state = RunStore().conversation(session_id)["state"]
    assert state["constraint_solution_set"]["solution_set_id"] == result["artifacts"][0]["solution_set_id"]
    assert state["constraint_solution_set"]["source_run_id"] == result["run_id"]
    assert state["constraint_solution_set"]["verification"]["global_goal"] is True
    assert model.index == 2


def test_verified_infeasible_plan_has_a_distinct_non_success_terminal(monkeypatch):
    message = "我只有1元预算，想买一个耳机一个键盘"
    spec = shared_budget_spec(budget=1)
    dialogue = {
        "reason": "共享预算需要约束规划",
        "commands": [],
        "objectives": [{
            "id": "bundle", "kind": "CONSTRAINT_PLAN", "source_quote": message,
            "constraint_plan": spec,
        }],
    }
    model = ScriptedModel(dialogue, approved_audit(spec, message))
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: model)

    result = orchestrator.run_orchestrator(message)

    assert result["status"] == "PARTIAL"
    assert result["terminal_reason"] == "NO_FEASIBLE_SOLUTION"
    assert result["plan"]["verification"]["all_goals_satisfied"] is False
    assert result["artifacts"][0]["verification"]["verdict"] == "PROVEN_INFEASIBLE"
    assert result["error"] is None
