from dataclasses import dataclass

import pytest

from src.errors import AgentError
from src.harness.contracts import RunRequest
from src.harness.profile import EngineRef, HarnessProfile, ResearchOrigin
from src.harness.registry import HarnessRegistry, HarnessRunner, get_registry
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


def test_two_profiles_bind_different_context_implementations_to_same_engine():
    from src.task_plan import TaskPlan

    baseline = get_registry().resolve("adaptive-v1")
    candidate = get_registry().resolve("adaptive-objective-context-v1")
    assert baseline.engine is candidate.engine
    baseline_context = baseline.build_context("推荐耳机")
    candidate_context = candidate.build_context("推荐耳机")
    state = {"request": "推荐耳机", "goal": None, "goal_revision": 0, "next_steps": None,
             "working_memory": None, "objective_ledger": []}
    left, left_manifest = baseline_context.decision_messages("系统", dict(state), TaskPlan([], []), [], max_chars=2000)
    right, right_manifest = candidate_context.decision_messages("系统", dict(state), TaskPlan([], []), [], max_chars=2000)
    assert left_manifest["selection_strategy"] == "recent-first"
    assert right_manifest["selection_strategy"] == "objective-first"
    assert left[1].content != right[1].content
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

