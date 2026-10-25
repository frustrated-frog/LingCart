import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.pg_conditions import ConditionClause, EdgeCondition, evaluate_edge_condition
from src.pg_retrieval import retrieve_state_conditioned
from src.procedural_graph import ProceduralGraph


V1 = Path("research/graphs/pg-seed-v1.json")
V2 = Path("research/graphs/pg-seed-v2.json")


def graph():
    return ProceduralGraph.from_path(V2)


def trajectory(**changes):
    base = {
        "goal_defined": True,
        "all_objectives_verified": False,
        "objectives": [{"objective_id": "choice", "kind": "recommend", "status": "PENDING",
                        "evidence_refs": []}],
        "unresolved_objective_kinds": ["recommend"],
        "recent_actions": [],
        "pending_steps": [],
        "feedback_code": None,
        "budget": {"model_calls_remaining": 4, "tool_calls_remaining": 4},
        "edge_feedback": {},
    }
    return {**base, **changes}


def test_legacy_graph_identity_is_stable_and_typed_graph_is_separate():
    legacy = ProceduralGraph.from_path(V1)
    typed = graph()
    assert legacy.fingerprint == "1a809550753df75be2ecfb3b9ec7edce38f3bd3ac5acf90982728e3768615d81"
    assert legacy.schema_version == 1 and typed.schema_version == 2
    assert typed.fingerprint != legacy.fingerprint
    assert all(edge.applicability is not None for edge in typed.edges)


def test_condition_protocol_rejects_unknown_facts_code_and_invalid_types():
    with pytest.raises(ValidationError):
        ConditionClause.model_validate({"fact": "python.eval", "operator": "EQ", "value": True})
    with pytest.raises(ValidationError):
        ConditionClause(fact="goal.defined", operator="CONTAINS", value="x")
    with pytest.raises(ValidationError):
        EdgeCondition.model_validate({"all_of": [], "any_of": []})


def test_three_valued_evaluation_never_treats_missing_evidence_as_true():
    condition = EdgeCondition(all_of=(
        ConditionClause(fact="runtime.feedback_code", operator="EXISTS"),
    ))
    unknown = evaluate_edge_condition(condition, {})
    absent = evaluate_edge_condition(condition, {"feedback_code": None})
    present = evaluate_edge_condition(condition, {"feedback_code": "EMPTY_CATALOG_RESULT"})
    assert [unknown["status"], absent["status"], present["status"]] == ["UNKNOWN", "FALSE", "TRUE"]
    assert unknown["evidence_bindings"][0]["source"] == "trajectory_state.feedback_code"


@pytest.mark.parametrize(("changes", "anchor", "selected"), [
    ({"goal_defined": False, "objectives": [], "unresolved_objective_kinds": []}, "Start", ["start-goal"]),
    ({}, "set_goal", ["goal-search"]),
    ({"recent_actions": [{"action": "search_products", "outcome": "EMPTY"}]},
     "search_products", []),
    ({"recent_actions": [{"action": "search_products", "outcome": "NON_EMPTY"}]},
     "search_products", ["search-inspect"]),
    ({"recent_actions": [{"action": "inspect", "outcome": "SUCCESS"}]},
     "inspect", ["inspect-recommend"]),
    ({"all_objectives_verified": True, "objectives": [
        {"objective_id": "choice", "kind": "recommend", "status": "VERIFIED", "evidence_refs": ["s1"]}],
      "unresolved_objective_kinds": [], "recent_actions": [{"action": "recommend", "outcome": "SUCCESS"}]},
     "complete", ["complete-end"]),
])
def test_state_conditioned_retrieval_only_selects_currently_proven_edges(changes, anchor, selected):
    result = retrieve_state_conditioned(graph(), trajectory(**changes), hops=2, max_edges=6)
    assert result["anchor"] == anchor
    assert [edge["id"] for edge in result["edges"]] == selected
    assert result["applicability"]["selected_edge_ids"] == selected
    assert all(edge["applicability"]["status"] == "TRUE" for edge in result["edges"])
    assert all(path["currently_applicable_prefix"] == 1 for path in result["paths"])


def test_schema_versions_cannot_mix_typed_and_untyped_edges():
    data = json.loads(V2.read_text())
    data["edges"][0].pop("applicability")
    with pytest.raises(ValidationError, match="每条边"):
        ProceduralGraph.model_validate_json(json.dumps(data))
    legacy = json.loads(V1.read_text())
    legacy["edges"][0]["applicability"] = {
        "all_of": [{"fact": "goal.defined", "operator": "EQ", "value": False}],
        "any_of": [],
    }
    with pytest.raises(ValidationError, match="不能混入"):
        ProceduralGraph.model_validate_json(json.dumps(legacy))
