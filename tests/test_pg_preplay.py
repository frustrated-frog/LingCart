from pathlib import Path

import pytest

from src.pg_preplay import compare_preplay_prediction, plan_preplay
from src.pg_retrieval import retrieve_state_conditioned
from src.procedural_graph import ProceduralGraph


GRAPH = Path("research/graphs/pg-seed-v2.json")


def trajectory(**changes):
    value = {
        "schema_version": 1,
        "goal_revision": 0,
        "binding_revision": 0,
        "goal_defined": True,
        "goal_objective_count": 1,
        "objectives": [{"objective_id": "choice", "kind": "recommend", "status": "PENDING",
                        "evidence_count": 0, "failure_codes": []}],
        "unresolved_objective_kinds": ["recommend"],
        "all_objectives_verified": False,
        "feedback_code": None,
        "recent_actions": [],
        "pending_steps": [],
        "budget": {"model_calls_remaining": 8, "tool_calls_remaining": 8},
        "edge_feedback": {},
    }
    value.update(changes)
    return value


def preplay(current=None, *, max_depth=3):
    graph = ProceduralGraph.from_path(GRAPH)
    state = current or trajectory()
    selection = retrieve_state_conditioned(graph, state, hops=2, max_edges=6)
    return plan_preplay(graph, selection, state, max_depth=max_depth, beam_width=4, max_paths=6)


def test_preplay_is_reproducible_bounded_and_never_marks_future_edges_executable():
    first = preplay()
    second = preplay()

    assert first == second
    assert first["status"] == "READY"
    assert first["selected_path"]["first_action"] == "search_products"
    assert first["selected_path"]["edge_ids"][0] == "goal-search"
    assert len(first["selected_path"]["edge_ids"]) <= 3
    assert first["selected_path"]["executable_prefix"] == 1
    assert first["selected_path"]["future_edges_require_revalidation"] == first["selected_path"]["edge_ids"][1:]
    assert first["selected_path"]["predictions"][0]["state_basis"] == "CURRENT_VERIFIED"
    assert all(row["state_basis"] == "PREDICTED_ACTION_ONLY"
               for row in first["selected_path"]["predictions"][1:])
    assert first["causal_claim"] is False and first["probabilistic_claim"] is False


def test_unknown_outcomes_remain_unknown_instead_of_becoming_execution_evidence():
    plan = preplay()
    future = plan["selected_path"]["predictions"][1:]
    assert future
    assert all(row["condition_status"] == "UNKNOWN" for row in future)
    assert all(row["effect_hypotheses"] for row in future)


def test_preplay_abstains_without_a_proven_first_transition():
    graph = ProceduralGraph.from_path(GRAPH)
    state = trajectory(unresolved_objective_kinds=["answer_faq"])
    selection = retrieve_state_conditioned(graph, state, hops=2, max_edges=6)
    result = plan_preplay(graph, selection, state)
    assert selection["abstained"] is True
    assert result["status"] == "ABSTAINED"
    assert result["selected_path"] is None and result["paths"] == []


def test_prediction_comparison_uses_real_information_gain_and_preserves_uncertainty():
    plan = preplay(max_depth=1)
    support = compare_preplay_prediction(plan, {
        "progress_status": "PROGRESSED",
        "progress_reason": "VERIFIABLE_STATE_CHANGE",
        "signals": ["TRUSTED_OBSERVATION_ADDED"],
        "information_gain": {"status": "NEW_INFORMATION", "newly_verified_objective_ids": []},
    }, actual_action="search_products")
    counter = compare_preplay_prediction(plan, {
        "progress_status": "NO_PROGRESS",
        "progress_reason": "NO_VERIFIABLE_STATE_CHANGE",
        "signals": [],
        "information_gain": {"status": "NO_NEW_INFORMATION", "newly_verified_objective_ids": []},
    }, actual_action="search_products")
    diverged = compare_preplay_prediction(plan, {}, actual_action="clarify")

    assert support["status"] == "SUPPORT" and support["causal_claim"] is False
    assert counter["status"] == "COUNTER"
    assert diverged["status"] == "DIVERGED" and diverged["action_match"] is False


@pytest.mark.parametrize("kwargs", [
    {"max_depth": 0}, {"max_depth": 5}, {"beam_width": 0}, {"beam_width": 9}, {"max_paths": 0},
])
def test_preplay_rejects_unbounded_search_configuration(kwargs):
    graph = ProceduralGraph.from_path(GRAPH)
    state = trajectory()
    selection = retrieve_state_conditioned(graph, state, hops=2, max_edges=6)
    with pytest.raises(ValueError):
        plan_preplay(graph, selection, state, **kwargs)
