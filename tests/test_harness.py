from dataclasses import dataclass
from copy import deepcopy
import json

import pytest

from src.errors import AgentError
from src.harness.contracts import RunRequest
from src.harness.profile import EngineRef, HarnessProfile, ResearchOrigin
from src.harness.registry import (
    HarnessRegistry,
    HarnessRunner,
    get_registry,
    harness_execution_matches,
    resolve_claim_runner,
)
from src.runtime import BudgetLimits, RunRuntime, runtime_scope


@dataclass
class FakeEngine:
    engine_id: str = "fake"
    engine_version: str = "1"
    execution_mode: str = "auto"

    def execute(self, request: RunRequest):
        return {"reply": request.message, "status": "SUCCEEDED", "events": []}


def profile(profile_id="test-v1"):
    return HarnessProfile(
        schema_version=1,
        profile_id=profile_id,
        version="1.0.0",
        display_name="测试 Profile",
        description="只用于验证稳定 Harness 边界。",
        engine=EngineRef(id="orchestrator", version="1"),
        mechanisms=(),
        capabilities=("durable_runtime",),
        research=ResearchOrigin(status="baseline", source_type="native"),
    )


def test_built_in_profiles_are_versioned_and_reproducible():
    profiles = {item.profile_id: item for item in get_registry().profiles()}
    assert {"adaptive-v1", "baseline-v1"} <= set(profiles)
    assert profiles["adaptive-v1"].engine.id == "adaptive"
    assert profiles["baseline-v1"].engine.id == "orchestrator"
    assert all(len(item.fingerprint) == 64 for item in profiles.values())


def test_after_sales_experiment_profiles_change_only_the_planning_mechanism():
    from src.harness.registry import get_product_runner, get_runner

    baseline = get_runner("after-sales-plan-observer-v0")
    candidate = get_runner("after-sales-policy-execution-v1")
    left, right = baseline.provenance(), candidate.provenance()

    assert left["engine"] == right["engine"]
    assert [item for item in left["mechanisms"] if item["slot"] != "planning"] == [
        item for item in right["mechanisms"] if item["slot"] != "planning"
    ]
    assert left["mechanisms"][0]["configuration"]["case_execution_mode"] == "observe_only"
    assert right["mechanisms"][0]["configuration"]["case_execution_mode"] == "policy_aware"
    assert left["mechanisms"][0]["configuration"]["evidence_intake_mode"] == "model_default"
    assert right["mechanisms"][0]["configuration"]["evidence_intake_mode"] == "model_default"
    for profile_id in ("after-sales-plan-observer-v0", "after-sales-policy-execution-v1"):
        with pytest.raises(AgentError, match="仅限研究"):
            get_product_runner(profile_id)


def test_evidence_boundary_profile_changes_only_the_planning_mechanism():
    from src.harness.registry import get_product_runner, get_runner

    baseline = get_runner("after-sales-policy-execution-v1").provenance()
    candidate = get_runner("after-sales-evidence-boundary-v2").provenance()

    assert baseline["engine"] == candidate["engine"]
    assert [item for item in baseline["mechanisms"] if item["slot"] != "planning"] == [
        item for item in candidate["mechanisms"] if item["slot"] != "planning"
    ]
    baseline_planning = next(item for item in baseline["mechanisms"] if item["slot"] == "planning")
    candidate_planning = next(item for item in candidate["mechanisms"] if item["slot"] == "planning")
    assert baseline_planning["configuration"] == {
        "case_execution_mode": "policy_aware",
        "evidence_intake_mode": "model_default",
    }
    assert candidate_planning["configuration"] == {
        "case_execution_mode": "policy_aware",
        "evidence_intake_mode": "case_first",
    }
    with pytest.raises(AgentError, match="仅限研究"):
        get_product_runner("after-sales-evidence-boundary-v2")


def test_evidence_readiness_profile_is_a_research_only_planning_mechanism():
    from src.harness.registry import get_product_runner, get_runner

    boundary = get_runner("after-sales-evidence-boundary-v2").provenance()
    readiness = get_runner("after-sales-evidence-readiness-v3").provenance()

    assert boundary["engine"] == readiness["engine"]
    assert [item for item in boundary["mechanisms"] if item["slot"] != "planning"] == [
        item for item in readiness["mechanisms"] if item["slot"] != "planning"
    ]
    planning = next(item for item in readiness["mechanisms"] if item["slot"] == "planning")
    assert planning["id"] == "evidence-readiness-case-execution"
    assert planning["version"] == "3"
    assert planning["configuration"] == {
        "case_execution_mode": "policy_aware",
        "evidence_intake_mode": "readiness_ledger",
    }
    with pytest.raises(AgentError, match="仅限研究"):
        get_product_runner("after-sales-evidence-readiness-v3")


def test_two_profiles_bind_different_context_implementations_to_same_engine():
    from src.task_plan import TaskPlan, TaskStep

    baseline = get_registry().resolve("adaptive-v1")
    candidate = get_registry().resolve("adaptive-objective-context-v1")
    assert baseline.engine is candidate.engine
    plan = TaskPlan(["verified", "open"], [
        TaskStep("verified-old", "shopping", "search_products", {}, status="SUCCEEDED", result={
            "products": [{"product_id": "OLD", "name": "旧证据", "price": 1, "summary": "A" * 1200}],
            "observation": "A" * 500,
        }),
        TaskStep("latest-open", "shopping", "search_products", {}, status="SUCCEEDED", result={
            "products": [{"product_id": "NEW", "name": "新证据", "price": 2, "summary": "B" * 1200}],
            "observation": "B" * 500,
        }),
    ])
    state = {"request": "对比复合任务证据", "goal": None, "goal_revision": 0, "next_steps": None,
             "working_memory": None, "objective_ledger": [
                 {"objective_id": "verified", "status": "VERIFIED", "evidence_refs": ["verified-old"]},
                 {"objective_id": "open", "status": "PENDING", "evidence_refs": []},
             ]}
    left, left_manifest = baseline.build_context(state["request"]).decision_messages(
        "系统", dict(state), plan, [], max_chars=3500)
    right, right_manifest = candidate.build_context(state["request"]).decision_messages(
        "系统", dict(state), plan, [], max_chars=3500)
    left_observations = json.loads(left[1].content)["observations"]
    right_observations = json.loads(right[1].content)["observations"]
    left_full = [item["step_id"] for item in left_observations if not item["result"].get("archived")]
    right_full = [item["step_id"] for item in right_observations if not item["result"].get("archived")]

    assert left_manifest["selection_strategy"] == "recent-first"
    assert right_manifest["selection_strategy"] == "objective-first"
    assert left_full == ["latest-open"]
    assert right_full == ["verified-old"]
    assert baseline.provenance()["execution_sha256"] != candidate.provenance()["execution_sha256"]


def test_registry_rejects_unknown_profiles_and_engine_versions():
    registry = HarnessRegistry()
    with pytest.raises(ValueError, match="未注册引擎"):
        registry.register_profile(profile())
    with pytest.raises(AgentError) as error:
        registry.resolve("missing-v1")
    assert error.value.code == "HARNESS_PROFILE_UNKNOWN"


def test_runner_records_profile_before_engine_and_returns_provenance():
    selected = profile()
    runner = HarnessRunner(selected, FakeEngine())
    result = runner("保留真实请求")
    assert result["reply"] == "保留真实请求"
    assert result["events"][0]["type"] == "harness_profile_selected"
    assert result["harness"]["profile_sha256"] == selected.fingerprint


def test_one_run_cannot_switch_profiles():
    context = RunRuntime("run", BudgetLimits())
    with runtime_scope(context):
        HarnessRunner(profile("first-v1"), FakeEngine())("第一步")
        with pytest.raises(AgentError) as error:
            HarnessRunner(profile("second-v1"), FakeEngine())("第二步")
    assert error.value.code == "HARNESS_PROFILE_CONFLICT"


def test_resume_rejects_a_profile_whose_bound_implementation_changed():
    runner = get_registry().resolve("adaptive-v1")
    saved = runner.provenance()
    saved["engine"]["implementation_sha256"] = "0" * 64
    claim = {"snapshot": {"harness": saved}, "events": []}

    with pytest.raises(AgentError) as error:
        resolve_claim_runner(claim)

    assert error.value.code == "HARNESS_IMPLEMENTATION_MISMATCH"


def test_execution_identity_ignores_git_revision_but_not_source_fingerprints():
    current = get_registry().resolve("adaptive-v1").provenance()
    another_commit = deepcopy(current)
    another_commit["source_revision"] = "unrelated-documentation-commit"
    another_commit["execution_sha256"] = "legacy-hash-that-included-the-revision"

    assert harness_execution_matches(current, another_commit)

    another_commit["mechanisms"][0]["implementation_sha256"] = "changed-source"
    assert not harness_execution_matches(current, another_commit)

