from src.pg_progress import assess_action_progress, progress_snapshot
from src.runtime import BudgetLimits, RunRuntime
from src.task_plan import TaskPlan, TaskStep
from copy import deepcopy
import pytest


def state(*, verified=False):
    return {
        "goal_revision": 0,
        "binding_revision": 0,
        "goal": {"objectives": [{"id": "choice", "kind": "recommend"}]},
        "objective_ledger": [{
            "objective_id": "choice",
            "status": "VERIFIED" if verified else "PENDING",
            "evidence_refs": ["s1"] if verified else [],
        }],
        "feedback": None,
    }


def test_progress_requires_a_verifiable_state_change_not_just_a_completed_call():
    runtime = RunRuntime("progress", BudgetLimits())
    before = progress_snapshot(state(), TaskPlan([], []), runtime)
    empty = TaskStep("s1", "shopping", "search_products", {}, status="SUCCEEDED",
                     result={"products": []})
    after = progress_snapshot(state(), TaskPlan([], [empty]), runtime)
    result = assess_action_progress(before, after,
                                    {"action": "tool", "tool": "search_products"}, step=empty)
    assert result["progress_status"] == "NO_PROGRESS"
    assert result["signals"] == []


def test_progress_attributes_trusted_observation_and_verified_goal_separately():
    runtime = RunRuntime("progress", BudgetLimits())
    before = progress_snapshot(state(), TaskPlan([], []), runtime)
    observed = TaskStep("s1", "shopping", "search_products", {}, status="SUCCEEDED",
                        result={"products": [{"product_id": "P1"}]})
    after_observation = progress_snapshot(state(), TaskPlan([], [observed]), runtime)
    observed_result = assess_action_progress(before, after_observation,
        {"action": "tool", "tool": "search_products"}, step=observed)
    assert observed_result["progress_status"] == "PROGRESSED"
    assert observed_result["signals"] == ["TRUSTED_OBSERVATION_ADDED"]

    after_verified = progress_snapshot(state(verified=True), TaskPlan([], [observed]), runtime)
    verified_result = assess_action_progress(after_observation, after_verified,
        {"action": "recommend"}, step=observed)
    assert set(verified_result["signals"]) == {"OBJECTIVES_VERIFIED", "EVIDENCE_ADDED"}


def test_failed_and_deferred_actions_are_not_reported_as_progress():
    runtime = RunRuntime("progress", BudgetLimits())
    before = progress_snapshot(state(), TaskPlan([], []), runtime)
    failed = TaskStep("s1", "shopping", "search_products", {}, status="FAILED",
                      error={"code": "TOOL_TIMEOUT"})
    after = progress_snapshot(state(), TaskPlan([], [failed]), runtime)
    failure = assess_action_progress(before, after,
                                     {"action": "tool", "tool": "search_products"}, step=failed)
    deferred = assess_action_progress(before, before, {"action": "clarify"})
    assert failure["progress_status"] == "FAILED"
    assert deferred["progress_status"] == "DEFERRED"


def observed(step_id, *, products=None, tool="search_products", args=None, **metadata):
    return TaskStep(step_id, "shopping", tool, args or {}, status="SUCCEEDED", result={
        "products": products if products is not None else [{"product_id": "P1", "price": 100}],
        "data_source": "TEST_CATALOG", **metadata})


def assess(before_steps, after_steps, *, before_state=None, after_state=None):
    runtime = RunRuntime("information-gain", BudgetLimits())
    before = progress_snapshot(before_state or state(), TaskPlan([], before_steps), runtime)
    after = progress_snapshot(after_state or state(), TaskPlan([], after_steps), runtime)
    return assess_action_progress(before, after, {"action": "tool", "tool": after_steps[-1].tool},
                                  step=after_steps[-1])


@pytest.mark.parametrize("variation", ["step_id", "timestamp", "query", "reorder", "subset", "compare", "reused"])
def test_duplicate_observations_do_not_reward_new_execution_records(variation):
    products = [{"product_id": "P1", "price": 100}, {"product_id": "P2", "price": 200}]
    first = observed("s1", products=products, observation={"timestamp": "old", "reference": "run:1"})
    second = deepcopy(first)
    second.step_id = "s2"
    if variation == "timestamp":
        second.result["observation"] = {"timestamp": "new", "reference": "run:2", "digest": "new"}
    elif variation == "query":
        second.args = {"query": "another query"}
        second.result["filters"] = second.args
        second.result["total"] = 40
    elif variation == "reorder":
        second.result["products"].reverse()
    elif variation == "subset":
        second.result["products"] = products[:1]
    elif variation == "compare":
        second.tool = "compare_products"
        second.args = {"product_ids": ["P1", "P2"]}
    elif variation == "reused":
        second.status = "REUSED"
    outcome = assess([first], [first, second])
    assert outcome["execution_progress"] is True
    assert outcome["progress_status"] == "NO_PROGRESS"
    assert outcome["information_gain"]["status"] == "NO_NEW_INFORMATION"
    assert outcome["learning_eligible"] is False


@pytest.mark.parametrize("change", ["new_product", "price", "version", "source", "restore_price"])
def test_new_or_changed_trusted_information_is_still_progress(change):
    first = observed("s1")
    second = observed("s2")
    before = [first]
    if change == "new_product":
        second.result["products"].append({"product_id": "P2", "price": 200})
    elif change == "price":
        second.result["products"][0]["price"] = 120
    elif change == "version":
        second.result["source_version"] = "2"
    elif change == "source":
        second.result["data_source"] = "ANOTHER_CATALOG"
    else:
        before.append(observed("changed", products=[{"product_id": "P1", "price": 120}]))
    outcome = assess(before, [*before, second])
    assert outcome["information_gain"]["status"] == "NEW_INFORMATION"
    assert len(outcome["information_gain"]["changed_observation_scopes"]) == 1
    assert outcome["learning_eligible"] is True


def test_new_evidence_reference_to_same_information_is_not_rewarded():
    first, second = observed("s1"), observed("s2")
    old_state = state()
    old_state["objective_ledger"][0]["evidence_refs"] = ["s1"]
    new_state = deepcopy(old_state)
    new_state["objective_ledger"][0]["evidence_refs"].append("s2")
    outcome = assess([first], [first, second], before_state=old_state, after_state=new_state)
    assert outcome["signals"] == []
    assert outcome["information_gain"]["new_evidence_count"] == 0


def test_archive_reads_resolve_original_evidence_identity():
    first = observed("s1")
    read = TaskStep("s2", "shopping", "read_evidence", {"reference": "s1"}, status="SUCCEEDED",
                    result={"result": first.result, "reference": "s1"})
    old_state, new_state = state(), state()
    old_state["objective_ledger"][0]["evidence_refs"] = ["s1"]
    new_state["objective_ledger"][0]["evidence_refs"] = ["s2"]
    assert assess([first], [first, read], before_state=old_state, after_state=new_state)["signals"] == []


def test_goal_identity_and_actual_mapping_change_are_used_instead_of_counts():
    step = observed("s1")
    old_state, new_state = state(verified=True), state(verified=True)
    new_state["objective_ledger"][0]["objective_id"] = "another"
    new_state["binding_revision"] = 1
    outcome = assess([step], [step], before_state=old_state, after_state=new_state)
    assert "OBJECTIVES_VERIFIED" in outcome["signals"]
    assert "BINDING_REFINED" not in outcome["signals"]


@pytest.mark.parametrize("status", ["FAILED", "PENDING", "REUSED"])
def test_unsuccessful_or_reused_steps_do_not_refresh_observations(status):
    first = observed("s1")
    second = observed("s2", products=[{"product_id": "P1", "price": 200}])
    second.status = status
    assert assess([first], [first, second])["information_gain"]["changed_observation_scopes"] == []


def test_legacy_snapshot_remains_unknown_and_cannot_reward_learning():
    runtime = RunRuntime("legacy", BudgetLimits())
    after = progress_snapshot(state(), TaskPlan([], [observed("s1")]), runtime)
    outcome = assess_action_progress({"schema_version": 1}, after, {"action": "tool", "tool": "search_products"})
    assert outcome["progress_status"] == "UNKNOWN"
    assert outcome["learning_eligible"] is False


def test_snapshot_roundtrip_is_reproducible_and_contains_no_business_content():
    import json
    plan = TaskPlan([], [observed("s1")])
    runtime = RunRuntime("restore", BudgetLimits())
    first = progress_snapshot(state(), plan, runtime)
    restored_plan = TaskPlan([], [TaskStep(**item) for item in json.loads(json.dumps(plan.payload()))["steps"]])
    assert progress_snapshot(state(), restored_plan, runtime) == first
    assert "P1" not in json.dumps(first)


def test_non_product_observation_ignores_runtime_wrapper_but_retains_scope_and_version():
    first = TaskStep("s1", "order", "get_order_status", {"order_id": 1}, status="SUCCEEDED",
                     result={"order_id": 1, "status": "delivered", "observation": {"timestamp": "old"}})
    repeated = deepcopy(first)
    repeated.step_id = "s2"
    repeated.result["observation"]["timestamp"] = "new"
    assert assess([first], [first, repeated])["progress_status"] == "NO_PROGRESS"
    repeated.result["source_version"] = 2
    assert assess([first], [first, repeated])["progress_status"] == "PROGRESSED"


def test_restored_review_produces_one_idempotent_content_based_outcome():
    import json
    from pathlib import Path
    from src.context_engine import ContextEngine
    from src.pg_context import PGContextEngine
    from src.procedural_graph import ProceduralGraph

    runtime = RunRuntime("restored-review", BudgetLimits())
    first, second = observed("s1"), observed("s2")
    before = progress_snapshot(state(), TaskPlan([], [first]), runtime)
    saved_review = json.loads(json.dumps({"type": "pg_decision_reviewed", "review_id": "review",
        "decision_key": "decision", "progress_before": before, "matched_edge_ids": ["search-inspect"]}))
    runtime.events.append(saved_review)
    engine = PGContextEngine(ContextEngine("test"),
        ProceduralGraph.from_path(Path("research/graphs/pg-seed-v2.json")), mechanism_version="4")
    kwargs = dict(review_id="review", decision={"action": "tool", "tool": "search_products"}, step=second)
    outcome = engine.review_outcome(runtime, state(), TaskPlan([], [first, second]), **kwargs)
    repeated = engine.review_outcome(runtime, state(), TaskPlan([], [first, second]), **kwargs)
    assert repeated == outcome and outcome["progress_status"] == "NO_PROGRESS"
    assert len([event for event in runtime.events if event["type"] == "pg_action_outcome"]) == 1


def test_v6_records_one_prediction_comparison_from_real_outcome():
    import json
    from pathlib import Path
    from src.context_engine import ContextEngine
    from src.pg_context import PGContextEngine
    from src.procedural_graph import ProceduralGraph

    runtime = RunRuntime("prediction-outcome", BudgetLimits())
    before = progress_snapshot(state(), TaskPlan([], []), runtime)
    runtime.events.extend(json.loads(json.dumps([
        {"type": "pg_guidance", "decision_key": "decision", "preplay": {
            "plan_id": "plan-1", "selected_path": {"predictions": [{
                "action": "search_products", "effect_hypotheses": ["INFORMATION_GAIN"],
            }]},
        }},
        {"type": "pg_decision_reviewed", "review_id": "review", "decision_key": "decision",
         "contract_id": "contract", "progress_before": before, "matched_edge_ids": ["goal-search"]},
    ])))
    observed_step = observed("s1")
    engine = PGContextEngine(ContextEngine("test"),
        ProceduralGraph.from_path(Path("research/graphs/pg-seed-v2.json")), mechanism_version="6")
    kwargs = dict(review_id="review", decision={"action": "tool", "tool": "search_products"}, step=observed_step)
    outcome = engine.review_outcome(runtime, state(), TaskPlan([], [observed_step]), **kwargs)
    repeated = engine.review_outcome(runtime, state(), TaskPlan([], [observed_step]), **kwargs)
    comparisons = [event for event in runtime.events if event["type"] == "pg_prediction_outcome"]

    assert repeated == outcome and outcome["information_gain"]["status"] == "NEW_INFORMATION"
    assert len(comparisons) == 1
    assert comparisons[0]["status"] == "SUPPORT"
    assert comparisons[0]["preplay_plan_id"] == "plan-1"
    assert comparisons[0]["causal_claim"] is False
