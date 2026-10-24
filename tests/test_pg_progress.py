from src.pg_progress import assess_action_progress, progress_snapshot
from src.runtime import BudgetLimits, RunRuntime
from src.task_plan import TaskPlan, TaskStep


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
