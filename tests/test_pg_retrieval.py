from pathlib import Path

from src.pg_retrieval import build_trajectory_state, retrieve_state_aware, shadow_action_anchor
from src.procedural_graph import ProceduralGraph
from src.task_plan import TaskPlan, TaskStep


SEED = Path("research/graphs/pg-seed-v1.json")


def test_edge_feedback_excludes_legacy_progress_rewards():
    events = [
        {"type": "pg_action_outcome", "progress_status": "PROGRESSED", "matched_edge_ids": ["legacy"]},
        {"type": "pg_action_outcome", "schema_version": 2, "progress_status": "PROGRESSED",
         "learning_eligible": True, "matched_edge_ids": ["new"]},
        {"type": "pg_action_outcome", "schema_version": 2, "progress_status": "NO_PROGRESS",
         "learning_eligible": False, "matched_edge_ids": ["repeat"]},
    ]
    feedback = build_trajectory_state(goal_state(), TaskPlan([], []), events)["edge_feedback"]
    assert "legacy" not in feedback
    assert feedback["new"]["progressed"] == 1
    assert feedback["repeat"]["no_progress"] == 1


def graph():
    return ProceduralGraph.from_path(SEED)


def goal_state(*, recommend="PENDING", faq="VERIFIED"):
    return {
        "goal_revision": 0,
        "binding_revision": 0,
        "goal": {"objectives": [
            {"id": "recommendation", "kind": "recommend"},
            {"id": "policy", "kind": "answer_faq"},
        ]},
        "objective_ledger": [
            {"objective_id": "recommendation", "status": recommend, "evidence_refs": [], "failures": []},
            {"objective_id": "policy", "status": faq, "evidence_refs": ["step-2"], "failures": []},
        ],
        "feedback": None,
    }


def test_trajectory_projection_preserves_typed_goal_action_and_outcome_signals():
    plan = TaskPlan([], [
        TaskStep("step-1", "shopping", "search_products", {}, status="SUCCEEDED", result={"products": []}),
        TaskStep("step-2", "shopping", "answer_faq", {}, status="SUCCEEDED", result={"answer": "七天"}),
    ])
    events = [
        {"type": "adaptive_decision", "sequence": 10, "goal_revision": 0, "step_id": "step-1",
         "decision": {"action": "tool", "tool": "search_products"}},
        {"type": "adaptive_decision", "sequence": 20, "goal_revision": 0, "step_id": "step-2",
         "decision": {"action": "tool", "tool": "answer_faq"}},
        {"type": "adaptive_decision", "sequence": 30, "goal_revision": 1, "step_id": None,
         "decision": {"action": "clarify"}},
    ]
    trajectory = build_trajectory_state(goal_state(), plan, events)
    assert [item["action"] for item in trajectory["recent_actions"]] == ["search_products", "answer_faq"]
    assert trajectory["recent_actions"][0]["outcome"] == "EMPTY"
    assert trajectory["unresolved_objective_kinds"] == ["recommend"]
    assert trajectory["unfinished_plan_steps"] == []


def test_state_aware_localizer_uses_goal_lifecycle_before_any_action():
    missing = retrieve_state_aware(graph(), {
        "goal_defined": False, "recent_actions": [], "unresolved_objective_kinds": [],
        "pending_steps": [], "all_objectives_verified": False,
    })
    declared = retrieve_state_aware(graph(), {
        "goal_defined": True, "recent_actions": [], "unresolved_objective_kinds": ["recommend"],
        "pending_steps": [], "all_objectives_verified": False,
    })
    assert missing["anchor"] == "Start"
    assert declared["anchor"] == "set_goal"
    assert missing["fallback"] is False and declared["fallback"] is False


def test_unknown_latest_action_recovers_from_known_empty_search_without_full_graph_fallback():
    plan = TaskPlan([], [
        TaskStep("step-1", "shopping", "search_products", {}, status="SUCCEEDED", result={"products": []}),
        TaskStep("step-2", "faq", "answer_faq", {}, status="SUCCEEDED", result={"answer": "七天"}),
    ])
    events = [
        {"type": "adaptive_decision", "sequence": 10, "goal_revision": 0, "step_id": "step-1",
         "decision": {"action": "tool", "tool": "search_products"}},
        {"type": "adaptive_decision", "sequence": 20, "goal_revision": 0, "step_id": "step-2",
         "decision": {"action": "tool", "tool": "answer_faq"}},
    ]
    trajectory = build_trajectory_state(goal_state(), plan, events)
    selected = retrieve_state_aware(graph(), trajectory)
    shadow = shadow_action_anchor(graph(), trajectory)
    assert selected["anchor"] == "search_products"
    assert selected["status"] == "LOCALIZED"
    assert {edge["id"] for edge in selected["edges"]} == {"search-inspect", "inspect-recommend"}
    assert len(selected["nodes"]) < len(graph().nodes)
    assert shadow["anchor"] == "answer_faq" and shadow["fallback"] is True


def test_unrepresented_goal_and_unknown_action_abstain_instead_of_guessing_from_history():
    trajectory = {
        "goal_defined": True,
        "recent_actions": [
            {"action": "search_products", "outcome": "NON_EMPTY"},
            {"action": "answer_faq", "outcome": "SUCCESS"},
        ],
        "unresolved_objective_kinds": ["answer_faq"],
        "pending_steps": [],
        "all_objectives_verified": False,
    }
    selected = retrieve_state_aware(graph(), trajectory)
    assert selected["status"] == "ABSTAINED"
    assert selected["anchor"] is None
    assert selected["edges"] == [] and selected["fallback"] is False


def test_unrepresented_remaining_goal_abstains_even_when_latest_action_is_in_graph():
    selected = retrieve_state_aware(graph(), {
        "goal_defined": True,
        "recent_actions": [{"action": "recommend", "outcome": "SUCCESS"}],
        "unresolved_objective_kinds": ["answer_faq"],
        "pending_steps": [],
        "all_objectives_verified": False,
    })
    assert selected["status"] == "ABSTAINED" and selected["anchor"] is None


def test_goal_with_only_unrepresented_objective_abstains_before_first_action():
    selected = retrieve_state_aware(graph(), {
        "goal_defined": True,
        "recent_actions": [],
        "unresolved_objective_kinds": ["answer_faq"],
        "pending_steps": [],
        "all_objectives_verified": False,
    })
    assert selected["status"] == "ABSTAINED" and selected["edges"] == []


def test_verified_objectives_localize_completion_and_retrieval_is_bounded():
    selected = retrieve_state_aware(graph(), {
        "goal_defined": True,
        "recent_actions": [{"action": "recommend", "outcome": "SUCCESS"}],
        "unresolved_objective_kinds": [],
        "pending_steps": [],
        "all_objectives_verified": True,
    }, hops=2, max_edges=1)
    assert selected["anchor"] == "complete"
    assert [edge["id"] for edge in selected["edges"]] == ["complete-end"]
    assert len(selected["edges"]) <= 1

