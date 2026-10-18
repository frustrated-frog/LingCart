import json

import pytest
from pydantic import ValidationError

from src.ace_context import ExperienceBullet, ExperiencePlaybook, PlaybookSource
from src.ace_learning import (
    CuratorOperation,
    Episode,
    Reflection,
    BulletTag,
    episodes_from_report,
    evolve_playbook,
)
from src.context_engine import ContextEngine
from src.harness.mechanisms import built_in_mechanisms
from src.harness.registry import get_registry
from src.task_plan import TaskPlan


def source():
    return PlaybookSource(
        method="ace_offline",
        training_report_path="output/dev.json",
        training_report_sha256="a" * 64,
        training_split="dev",
        sample_count=1,
        model_config_sha256="b" * 64,
        generated_at="2026-10-18T00:00:00+00:00",
    )


def playbook():
    return ExperiencePlaybook(
        playbook_id="lingcart-ace",
        version=1,
        parent_sha256="c" * 64,
        source=source(),
        bullets=(
            ExperienceBullet(
                bullet_id="ace-00001",
                section="evidence",
                content="推荐之前必须读取完整商品记录，不能只根据归档索引里的编号做判断。",
                keywords=("完整证据", "推荐"),
                source_reflections=("reflection-A01-r1",),
                source_cases=("A01",),
            ),
            ExperienceBullet(
                bullet_id="ace-00002",
                section="recovery",
                content="窄查询没有结果时应扩大检索范围，不能把局部空结果当作目录中不存在。",
                keywords=("空结果", "扩大检索"),
                source_reflections=("reflection-A02-r1",),
                source_cases=("A02",),
            ),
        ),
    )


def test_playbook_is_versioned_and_selection_saturates_before_ranking():
    current = playbook()
    selection = current.select("推荐耳机", max_bullets=8, max_chars=4000)

    assert selection.policy == "saturate"
    assert [item.bullet_id for item in selection.bullets] == ["ace-00001", "ace-00002"]
    assert len(current.fingerprint) == 64
    assert ExperiencePlaybook.model_validate_json(
        json.dumps(current.canonical_payload(), ensure_ascii=False)
    ) == current


def test_evolution_updates_counters_and_only_appends_nonduplicate_bullets():
    base = playbook()
    reflections = (
        Reflection(
            reflection_id="reflection-A11-r1",
            case_id="A11",
            repetition=1,
            passed=False,
            diagnosis="模型把一次空查询误判成目录没有候选。",
            lessons=("查询失败时根据失败类型选择恢复动作，不能直接结束任务。",),
            bullet_tags=(BulletTag(
                bullet_id="ace-00002", tag="helpful", evidence="该条目对应正确恢复方式"
            ),),
        ),
    )
    operations = (
        CuratorOperation(
            type="ADD",
            section="verification",
            content="完成复合请求前逐项目核对目标账本，缺少任何目标证据时继续执行而不是提前完成。",
            keywords=("复合请求", "目标账本"),
            reason="失败轨迹提前结束",
            reflection_refs=("reflection-A11-r1",),
        ),
        CuratorOperation(
            type="ADD",
            section="recovery",
            content=base.bullets[1].content,
            keywords=("空结果",),
            reason="重复建议",
            reflection_refs=("reflection-A11-r1",),
        ),
    )

    evolved, applied, rejected = evolve_playbook(base, reflections, operations, source=source())

    assert base.bullets[1].helpful == 0
    assert evolved.version == 2 and evolved.parent_sha256 == base.fingerprint
    assert evolved.bullets[1].helpful == 1
    assert evolved.bullets[-1].bullet_id == "ace-00003"
    assert len(applied) == len(rejected) == 1
    assert rejected[0]["rejection_code"] == "DUPLICATE_CONTENT"


def test_curator_rejects_business_identifiers():
    with pytest.raises(ValidationError, match="商品编号"):
        CuratorOperation(
            type="ADD",
            section="tool_use",
            content="以后直接推荐 P1107，因为它在这次任务中表现很好。",
            reason="错误地固化商品",
            reflection_refs=("reflection-A01-r1",),
        )


def test_ace_context_reserves_budget_and_exposes_provenance():
    from src.ace_context import AceContextEngine

    engine = AceContextEngine(ContextEngine("推荐耳机"), playbook(), max_bullets=8, max_chars=3000)
    state = {
        "request": "推荐耳机，遇到空结果继续查找",
        "goal": None,
        "goal_revision": 0,
        "next_steps": None,
        "feedback": None,
        "working_memory": None,
        "objective_ledger": [],
    }
    messages, manifest = engine.decision_messages(
        "系统规则", state, TaskPlan([], []), [], max_chars=5000
    )

    assert len(messages) == 3
    assert "候选执行策略" in messages[1].content
    assert manifest["experience_playbook_sha256"] == playbook().fingerprint
    assert manifest["experience_refs"] == ["ace-00001", "ace-00002"]
    assert engine.payload()["experience_playbook"]["version"] == 1


def test_ace_context_never_biases_independent_auditor():
    from src.ace_context import AceContextEngine

    engine = AceContextEngine(ContextEngine("推荐耳机"), playbook(), max_bullets=8, max_chars=3000)
    state = {
        "request": "推荐耳机",
        "goal": None,
        "goal_revision": 0,
        "next_steps": None,
        "feedback": {"audit_phase": "goal"},
        "working_memory": None,
        "objective_ledger": [],
    }
    messages, manifest = engine.decision_messages(
        "独立审查", state, TaskPlan([], []), [], max_chars=5000
    )

    assert len(messages) == 2
    assert manifest["experience_selection_policy"] == "independent_audit_bypass"
    assert manifest["experience_refs"] == []


def test_mechanism_configuration_is_path_scoped_and_source_bound():
    mechanism = next(item for item in built_in_mechanisms()
                     if item.mechanism_id == "ace-playbook-context")
    configured = mechanism.configure({
        "playbook_path": "research/playbooks/ace-empty-v1.json",
        "max_bullets": 16,
        "max_chars": 8000,
    })

    assert configured.provenance()["configuration"]["playbook_sha256"]
    assert "research/playbooks/ace-empty-v1.json" in configured.provenance()["source_files"]
    with pytest.raises(ValueError, match="research/playbooks"):
        mechanism.configure({"playbook_path": "../secret.json", "max_bullets": 16, "max_chars": 8000})


def test_existing_profiles_keep_configuration_out_of_their_canonical_identity():
    baseline = get_registry().resolve("adaptive-v1").profile
    assert all("configuration" not in item for item in baseline.canonical_payload()["mechanisms"])


def test_learning_report_rejects_test_leakage():
    report = {
        "execution_source": "live_provider",
        "status": "COMPLETED",
        "manifest": {"suite": "adaptive", "split": "test"},
        "samples": [{"case_id": "A11", "split": "test", "repetition": 1, "passed": True,
                     "turns": [], "model_attempts": 1, "tool_calls": 1}],
    }
    with pytest.raises(ValueError, match="dev split"):
        episodes_from_report(report)

