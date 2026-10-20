"""Offline ACE adaptation over frozen evaluation trajectories.

The production runtime remains read-only with respect to learned experience.
Generator output is the existing adaptive-agent trajectory; Reflector and
Curator run offline on the development split and emit a new immutable playbook.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Literal, Protocol

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.ace_context import (
    ExperienceBullet,
    ExperiencePlaybook,
    PlaybookSource,
    SECTIONS,
    normalized_content,
)
from src.agents.base import as_text
from src.llm import extract_usage, get_llm


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Episode(_StrictModel):
    case_id: str = Field(min_length=1, max_length=80)
    repetition: int = Field(ge=1)
    requests: tuple[str, ...] = Field(min_length=1, max_length=12)
    passed: bool
    failures: tuple[str, ...] = Field(default=(), max_length=40)
    trajectory: tuple[dict, ...] = Field(default=(), max_length=160)
    model_attempts: int = Field(ge=0)
    tool_calls: int = Field(ge=0)

    @property
    def identity(self) -> str:
        return f"{self.case_id}-r{self.repetition}"


class BulletTag(_StrictModel):
    bullet_id: str = Field(pattern=r"^ace-[0-9]{5}$")
    tag: Literal["helpful", "harmful", "neutral"]
    evidence: str = Field(min_length=1, max_length=500)


class ReflectionDraft(_StrictModel):
    diagnosis: str = Field(min_length=1, max_length=1600)
    lessons: tuple[str, ...] = Field(min_length=1, max_length=8)
    bullet_tags: tuple[BulletTag, ...] = Field(default=(), max_length=64)

    @field_validator("lessons")
    @classmethod
    def lessons_are_substantive(cls, values):
        if any(len(value.strip()) < 8 for value in values):
            raise ValueError("反思结论必须是可复用的完整策略")
        return values


class Reflection(_StrictModel):
    reflection_id: str = Field(pattern=r"^reflection-[a-zA-Z0-9_-]+-r[0-9]+$")
    case_id: str
    repetition: int
    passed: bool
    diagnosis: str
    lessons: tuple[str, ...]
    bullet_tags: tuple[BulletTag, ...] = ()


class CuratorOperation(_StrictModel):
    type: Literal["ADD"]
    section: Literal[
        "goal_parsing", "planning", "tool_use", "evidence",
        "verification", "recovery", "communication",
    ]
    content: str = Field(min_length=12, max_length=800)
    keywords: tuple[str, ...] = Field(default=(), max_length=16)
    reason: str = Field(min_length=1, max_length=500)
    reflection_refs: tuple[str, ...] = Field(min_length=1, max_length=40)

    @field_validator("content")
    @classmethod
    def content_is_reusable_strategy(cls, value):
        if re.search(r"(?<![A-Za-z0-9])P\d{4}S?(?![A-Za-z0-9])", value, re.I):
            raise ValueError("经验不能固化具体商品编号")
        if re.search(r"\b(?:order|订单)[-：:# ]*\d{3,}\b", value, re.I):
            raise ValueError("经验不能固化具体订单编号")
        if re.search(r"(?:放宽|放松|忽略|绕过|取消).{0,12}(?:预算|价格|硬约束|排除项|审批|安全)", value):
            raise ValueError("经验不能建议放宽硬约束、审批或安全边界")
        return value.strip()

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, values):
        normalized = tuple(value.strip().casefold() for value in values if value.strip())
        if len(normalized) != len(values) or len(set(normalized)) != len(normalized):
            raise ValueError("Curator 关键词必须非空且不能重复")
        return normalized


class CuratorDraft(_StrictModel):
    operations: tuple[CuratorOperation, ...] = Field(default=(), max_length=24)


class _RawCuratorDraft(_StrictModel):
    operations: tuple[dict, ...] = Field(default=(), max_length=24)


class Reflector(Protocol):
    def reflect(self, episode: Episode, playbook: ExperiencePlaybook) -> tuple[ReflectionDraft, dict]: ...


class Curator(Protocol):
    def curate(self, playbook: ExperiencePlaybook,
               reflections: tuple[Reflection, ...]) -> tuple[CuratorDraft, dict]: ...


_REFLECTOR_PROMPT = """你是 ACE Reflector。根据一次消费 Agent 的完整环境判分与压缩轨迹，诊断哪些策略真正帮助或伤害了结果。
环境判分是证据，不得把模型回复里的自我评价当真。passed=true 也要提炼可复用策略；失败时必须针对实际 failure/event，不得猜测隐藏原因。
Playbook 条目只是候选策略，不是业务事实、用户偏好或授权。只标注本次提供的 bullet_id；未实际相关的条目标 neutral。
输出严格 JSON：{diagnosis,lessons,bullet_tags:[{bullet_id,tag,evidence}]}。内容使用中文，lesson 必须可跨商品、订单和用户复用。"""


_CURATOR_PROMPT = """你是 ACE Curator。依据一组有来源编号的反思，对现有 Playbook 做 grow-and-refine 式增量更新。
只输出当前 Playbook 尚未覆盖、可跨任务复用的策略。禁止重写整本 Playbook，V1 只允许 ADD；有害条目由计数降权，不由模型静默删除。
禁止写入具体商品编号、订单编号、用户个人偏好、价格库存等易变业务事实；禁止放宽审批、安全、证据和工具边界。
每项必须引用支持它的 reflection_refs。输出严格 JSON：{operations:[{type:"ADD",section,content,keywords,reason,reflection_refs}]}。
合法 section：goal_parsing, planning, tool_use, evidence, verification, recovery, communication。内容使用中文。"""


class ModelReflector:
    def __init__(self, model=None):
        self.model = model or get_llm(purpose="ace_reflector")

    def reflect(self, episode: Episode, playbook: ExperiencePlaybook):
        selection = playbook.select("\n".join(episode.requests), max_bullets=32, max_chars=16000)
        payload = {
            "episode": episode.model_dump(mode="json"),
            "playbook": [bullet.model_dump(mode="json") for bullet in selection.bullets],
        }
        result, metadata = _json_call(self.model, _REFLECTOR_PROMPT, payload, ReflectionDraft)
        allowed = {bullet.bullet_id for bullet in selection.bullets}
        tagged = [tag.bullet_id for tag in result.bullet_tags]
        if len(tagged) != len(set(tagged)) or not set(tagged) <= allowed:
            raise ValueError("Reflector 引用了未暴露或重复的经验编号")
        return result, metadata


class ModelCurator:
    def __init__(self, model=None):
        self.model = model or get_llm(purpose="ace_curator")

    def curate(self, playbook: ExperiencePlaybook, reflections: tuple[Reflection, ...]):
        payload = {
            "playbook": playbook.canonical_payload(),
            "reflections": [reflection.model_dump(mode="json") for reflection in reflections],
        }
        raw, metadata = _json_call(self.model, _CURATOR_PROMPT, payload, _RawCuratorDraft)
        accepted, rejected = validate_curator_operations(raw.operations)
        result = CuratorDraft(operations=tuple(accepted))
        metadata["rejected_operations"] = rejected
        allowed = {reflection.reflection_id for reflection in reflections}
        if any(not set(operation.reflection_refs) <= allowed for operation in result.operations):
            raise ValueError("Curator 引用了不存在的反思证据")
        return result, metadata


def validate_curator_operations(operations):
    accepted, rejected = [], []
    for index, operation in enumerate(operations):
        try:
            accepted.append(CuratorOperation.model_validate_json(json.dumps(operation, ensure_ascii=False)))
        except Exception as error:
            rejected.append({
                "index": index,
                "operation": operation,
                "rejection_code": "POLICY_OR_SCHEMA_REJECTED",
                "message": str(error)[:800],
            })
    return tuple(accepted), rejected


def _json_call(model, system_prompt: str, payload: dict, schema):
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    prompt = system_prompt + "\nJSON Schema:\n" + json.dumps(
        schema.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
    bound = model.json_mode() if hasattr(model, "json_mode") else model
    response = bound.invoke([SystemMessage(content=prompt), HumanMessage(content=encoded)])
    parsed = schema.model_validate_json(as_text(response.content))
    return parsed, {
        "gateway": getattr(bound, "last_call", None),
        "usage": extract_usage(response),
    }


@dataclass(frozen=True)
class LearningResult:
    playbook: ExperiencePlaybook
    report: dict


class AceLearner:
    def __init__(self, reflector: Reflector, curator: Curator):
        self.reflector, self.curator = reflector, curator

    def learn(self, episodes: tuple[Episode, ...], base: ExperiencePlaybook, *, source: PlaybookSource):
        if not episodes:
            raise ValueError("ACE 学习至少需要一个 dev 样本")
        reflections, calls = [], []
        for episode in episodes:
            draft, metadata = self.reflector.reflect(episode, base)
            reflection = Reflection(
                reflection_id=f"reflection-{episode.identity}",
                case_id=episode.case_id,
                repetition=episode.repetition,
                passed=episode.passed,
                **draft.model_dump(),
            )
            reflections.append(reflection)
            calls.append({"role": "reflector", "episode": episode.identity, **metadata})
        reflection_tuple = tuple(reflections)
        curator_draft, metadata = self.curator.curate(base, reflection_tuple)
        calls.append({"role": "curator", **metadata})
        evolved, applied, rejected = evolve_playbook(base, reflection_tuple, curator_draft.operations, source=source)
        rejected = [*(metadata.get("rejected_operations") or []), *rejected]
        report = {
            "schema_version": 1,
            "method": "ace_offline",
            "paper_id": source.paper_id,
            "base_playbook_sha256": base.fingerprint,
            "playbook_sha256": evolved.fingerprint,
            "episodes": [episode.model_dump(mode="json") for episode in episodes],
            "reflections": [reflection.model_dump(mode="json") for reflection in reflection_tuple],
            "proposed_operations": [operation.model_dump(mode="json") for operation in curator_draft.operations],
            "applied_operations": applied,
            "rejected_operations": rejected,
            "model_calls": calls,
        }
        return LearningResult(evolved, report)


def evolve_playbook(base: ExperiencePlaybook, reflections: tuple[Reflection, ...],
                    operations: tuple[CuratorOperation, ...], *, source: PlaybookSource):
    known = {bullet.bullet_id: bullet for bullet in base.bullets}
    counters = {identity: [bullet.helpful, bullet.harmful] for identity, bullet in known.items()}
    for reflection in reflections:
        for tag in reflection.bullet_tags:
            if tag.bullet_id not in known:
                raise ValueError("反思标注了当前 Playbook 不存在的条目")
            if tag.tag == "helpful":
                counters[tag.bullet_id][0] += 1
            elif tag.tag == "harmful":
                counters[tag.bullet_id][1] += 1
    bullets = [bullet.model_copy(update={"helpful": counters[bullet.bullet_id][0],
                                         "harmful": counters[bullet.bullet_id][1]})
               for bullet in base.bullets]
    existing = {normalized_content(bullet.content) for bullet in bullets}
    reflections_by_id = {reflection.reflection_id: reflection for reflection in reflections}
    next_id = max([int(item.bullet_id.rsplit("-", 1)[1]) for item in bullets] + [0]) + 1
    applied, rejected = [], []
    for operation in operations:
        normalized = normalized_content(operation.content)
        if normalized in existing:
            rejected.append({**operation.model_dump(mode="json"), "rejection_code": "DUPLICATE_CONTENT"})
            continue
        if not set(operation.reflection_refs) <= set(reflections_by_id):
            rejected.append({**operation.model_dump(mode="json"), "rejection_code": "UNKNOWN_REFLECTION"})
            continue
        cases = tuple(sorted({reflections_by_id[identity].case_id for identity in operation.reflection_refs}))
        bullet = ExperienceBullet(
            bullet_id=f"ace-{next_id:05d}",
            section=operation.section,
            content=operation.content,
            keywords=operation.keywords,
            source_reflections=operation.reflection_refs,
            source_cases=cases,
        )
        bullets.append(bullet)
        existing.add(normalized)
        next_id += 1
        applied.append({"bullet_id": bullet.bullet_id, **operation.model_dump(mode="json")})
    result = ExperiencePlaybook(
        playbook_id=base.playbook_id,
        version=base.version + 1,
        parent_sha256=base.fingerprint,
        source=source,
        bullets=tuple(bullets),
    )
    return result, applied, rejected


def episodes_from_report(report: dict) -> tuple[Episode, ...]:
    manifest = report.get("manifest") or {}
    if report.get("execution_source") != "live_provider" or report.get("status") != "COMPLETED":
        raise ValueError("ACE 只能从已完成的真实模型报告学习")
    if manifest.get("suite") != "adaptive" or manifest.get("split") != "dev":
        raise ValueError("ACE 学习数据必须是 adaptive suite 的 dev split")
    samples = report.get("samples") or []
    if not samples or any(sample.get("split") != "dev" for sample in samples):
        raise ValueError("ACE 报告包含非 dev 样本或没有样本")
    episodes = []
    for sample in samples:
        failures, requests, trajectory = [], [], []
        for turn in sample.get("turns", []):
            requests.append(str(turn.get("message", "")))
            failures.extend(str(item) for item in turn.get("failures", []))
            result = turn.get("result") or {}
            for event in result.get("events", []):
                compact = _compact_event(event)
                if compact:
                    trajectory.append(compact)
        episodes.append(Episode(
            case_id=sample["case_id"],
            repetition=sample["repetition"],
            requests=tuple(requests),
            passed=sample["passed"],
            failures=tuple(failures),
            trajectory=tuple(trajectory[:160]),
            model_attempts=sample.get("model_attempts", 0),
            tool_calls=sample.get("tool_calls", 0),
        ))
    return tuple(episodes)


def _compact_event(event: dict) -> dict | None:
    event_type = event.get("type")
    if event_type not in {
        "adaptive_decision", "decision_rejected", "tool_result", "verification",
        "request_audited", "mechanism_applied", "evidence_context",
        "constraint_binding_proposed", "constraint_binding_unresolved",
        "constraint_binding_rejected", "constraint_bindings_refined", "decision_action_surface",
    }:
        return None
    compact = {"type": event_type}
    for key in ("code", "status", "tool", "mechanism_id", "reason", "verdict", "goal_revision",
                "phase", "source", "mode", "transaction_id", "resolution_id", "binding_revision"):
        if key in event:
            compact[key] = event[key]
    decision = event.get("decision")
    if isinstance(decision, dict):
        compact["decision"] = {key: decision.get(key) for key in
                               ("action", "tool", "reason", "objective_id") if decision.get(key) is not None}
    return compact


def source_for_report(path: Path, report: dict, *, model_manifest: dict | None = None,
                      generated_at: str | None = None) -> PlaybookSource:
    raw = path.read_bytes()
    model_config = model_manifest if model_manifest is not None else {
        "providers": report.get("providers"),
        "purpose_models": (report.get("manifest") or {}).get("purpose_models"),
    }
    encoded = json.dumps(model_config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return PlaybookSource(
        method="ace_offline",
        training_report_path=str(path),
        training_report_sha256=hashlib.sha256(raw).hexdigest(),
        training_split="dev",
        sample_count=len(report.get("samples") or []),
        model_config_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
        generated_at=generated_at or datetime.now(UTC).isoformat(),
    )

