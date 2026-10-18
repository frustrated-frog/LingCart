"""Versioned ACE playbooks and bounded, auditable context selection.

The playbook contains reusable execution strategies only.  It is deliberately
separate from user preferences, business observations and durable run state.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PAPER_ID = "arXiv:2510.04618"
SECTIONS = (
    "goal_parsing",
    "planning",
    "tool_use",
    "evidence",
    "verification",
    "recovery",
    "communication",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class PlaybookSource(_StrictModel):
    method: Literal["seed", "ace_offline"]
    paper_id: Literal[PAPER_ID] = PAPER_ID
    training_report_path: str | None = Field(default=None, max_length=300)
    training_report_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    training_split: Literal["dev"] | None = None
    sample_count: int = Field(default=0, ge=0)
    model_config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    generated_at: str

    @model_validator(mode="after")
    def report_metadata_is_atomic(self):
        values = (self.training_report_path, self.training_report_sha256, self.training_split)
        if self.method == "seed" and any(value is not None for value in values):
            raise ValueError("种子 Playbook 不能伪造训练报告来源")
        if self.method == "ace_offline" and any(value is None for value in values):
            raise ValueError("ACE Playbook 必须记录完整训练报告来源")
        return self


class ExperienceBullet(_StrictModel):
    bullet_id: str = Field(pattern=r"^ace-[0-9]{5}$")
    section: Literal[
        "goal_parsing", "planning", "tool_use", "evidence",
        "verification", "recovery", "communication",
    ]
    content: str = Field(min_length=12, max_length=800)
    keywords: tuple[str, ...] = Field(default=(), max_length=16)
    helpful: int = Field(default=0, ge=0)
    harmful: int = Field(default=0, ge=0)
    source_reflections: tuple[str, ...] = Field(default=(), max_length=80)
    source_cases: tuple[str, ...] = Field(default=(), max_length=80)

    @field_validator("keywords")
    @classmethod
    def keywords_are_normalized(cls, values):
        normalized = tuple(value.strip().casefold() for value in values if value.strip())
        if len(normalized) != len(values) or len(set(normalized)) != len(normalized):
            raise ValueError("经验关键词必须非空且不能重复")
        return normalized


class ExperiencePlaybook(_StrictModel):
    schema_version: Literal[1] = 1
    playbook_id: str = Field(pattern=r"^[a-z][a-z0-9-]*$", max_length=80)
    version: int = Field(ge=0)
    parent_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source: PlaybookSource
    bullets: tuple[ExperienceBullet, ...] = ()

    @model_validator(mode="after")
    def bullets_are_append_ordered_and_unique(self):
        identities = [bullet.bullet_id for bullet in self.bullets]
        if len(identities) != len(set(identities)):
            raise ValueError("Playbook 经验编号不能重复")
        numbers = [int(identity.rsplit("-", 1)[1]) for identity in identities]
        if numbers != sorted(numbers):
            raise ValueError("Playbook 经验必须按全局编号稳定排序")
        return self

    def canonical_payload(self) -> dict:
        return self.model_dump(mode="json")

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.canonical_payload(), ensure_ascii=False,
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    @classmethod
    def from_path(cls, path: Path) -> "ExperiencePlaybook":
        return cls.model_validate_json(path.read_text())

    @classmethod
    def empty(cls, playbook_id: str = "lingcart-ace", *, generated_at: str | None = None):
        return cls(
            playbook_id=playbook_id,
            version=0,
            source=PlaybookSource(
                method="seed",
                generated_at=generated_at or datetime.now(UTC).isoformat(),
            ),
        )

    def select(self, query: str, *, max_bullets: int, max_chars: int) -> "PlaybookSelection":
        if not 1 <= max_bullets <= 64 or not 256 <= max_chars <= 32000:
            raise ValueError("Playbook 上下文预算无效")
        active = [bullet for bullet in self.bullets if bullet.harmful <= bullet.helpful + 1]
        serialized = lambda bullet: len(json.dumps(bullet.model_dump(mode="json"), ensure_ascii=False))
        if len(active) <= max_bullets and sum(serialized(item) for item in active) <= max_chars:
            selected = active
            policy = "saturate"
        else:
            query_terms = _terms(query)

            def score(item: ExperienceBullet):
                terms = _terms(item.content + " " + " ".join(item.keywords))
                overlap = len(query_terms & terms)
                quality = (item.helpful - item.harmful) / (item.helpful + item.harmful + 2)
                return overlap, quality, -int(item.bullet_id.rsplit("-", 1)[1])

            ranked = sorted(active, key=score, reverse=True)
            selected, used = [], 0
            for item in ranked:
                size = serialized(item)
                if len(selected) >= max_bullets or used + size > max_chars:
                    continue
                selected.append(item)
                used += size
            selected.sort(key=lambda item: item.bullet_id)
            policy = "rank_after_saturation"
        selected_ids = {item.bullet_id for item in selected}
        return PlaybookSelection(
            bullets=tuple(selected),
            omitted_ids=tuple(item.bullet_id for item in self.bullets if item.bullet_id not in selected_ids),
            policy=policy,
        )


@dataclass(frozen=True)
class PlaybookSelection:
    bullets: tuple[ExperienceBullet, ...]
    omitted_ids: tuple[str, ...]
    policy: str


def _terms(value: str) -> set[str]:
    """Small deterministic tokenizer that works for Chinese and identifiers."""

    folded = value.casefold()
    terms = set(re.findall(r"[a-z0-9_]+", folded))
    for run in re.findall(r"[\u3400-\u9fff]+", folded):
        terms.update(run[index:index + 2] for index in range(max(1, len(run) - 1)))
    return terms


def normalized_content(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


class AceContextEngine:
    """Composition wrapper that reserves context budget for a frozen playbook."""

    def __init__(self, base, playbook: ExperiencePlaybook, *, max_bullets: int, max_chars: int):
        self.base = base
        self.playbook = playbook
        self.max_bullets = max_bullets
        self.max_chars = max_chars

    def __getattr__(self, name):
        return getattr(self.base, name)

    def payload(self):
        payload = self.base.payload()
        payload["experience_playbook"] = {
            "playbook_id": self.playbook.playbook_id,
            "version": self.playbook.version,
            "sha256": self.playbook.fingerprint,
        }
        return payload

    def model_messages(self, messages):
        return self.base.model_messages(messages)

    def observe(self, tool, args, result, *, reference):
        return self.base.observe(tool, args, result, reference=reference)

    def decision_messages(self, prompt, state, plan, history, *, max_chars):
        from src.context_budget import ContextBudgetError, _text_length

        if (state.get("feedback") or {}).get("audit_phase"):
            messages, manifest = self.base.decision_messages(
                prompt, state, plan, history, max_chars=max_chars)
            manifest.update({
                "experience_playbook_id": self.playbook.playbook_id,
                "experience_playbook_version": self.playbook.version,
                "experience_playbook_sha256": self.playbook.fingerprint,
                "experience_selection_policy": "independent_audit_bypass",
                "experience_refs": [],
                "experience_omitted_refs": [bullet.bullet_id for bullet in self.playbook.bullets],
                "experience_chars": 0,
            })
            return messages, manifest
        query = json.dumps({
            "request": state.get("request"),
            "goal": state.get("goal"),
            "feedback": state.get("feedback"),
            "next_steps": state.get("next_steps"),
        }, ensure_ascii=False, sort_keys=True)
        selection = self.playbook.select(
            query,
            max_bullets=self.max_bullets,
            max_chars=min(self.max_chars, max(256, max_chars // 3)),
        )
        bullets = list(selection.bullets)
        omitted = list(selection.omitted_ids)
        while True:
            experience = _experience_message(bullets) if bullets else None
            reserve = _text_length(experience) if experience else 0
            try:
                messages, manifest = self.base.decision_messages(
                    prompt, state, plan, history, max_chars=max_chars - reserve)
                break
            except ContextBudgetError:
                if not bullets:
                    raise
                omitted.insert(0, bullets.pop().bullet_id)
        if experience:
            messages = [messages[0], experience, *messages[1:]]
        manifest.update({
            "experience_playbook_id": self.playbook.playbook_id,
            "experience_playbook_version": self.playbook.version,
            "experience_playbook_sha256": self.playbook.fingerprint,
            "experience_selection_policy": selection.policy,
            "experience_refs": [bullet.bullet_id for bullet in bullets],
            "experience_omitted_refs": omitted,
            "experience_chars": reserve,
        })
        return messages, manifest

    @staticmethod
    def read_evidence(plan, reference, *, offset=0, limit=1):
        from src.context_engine import ContextEngine
        return ContextEngine.read_evidence(plan, reference, offset=offset, limit=limit)

    def effective_request(self):
        return self.base.effective_request()


def _experience_message(bullets: list[ExperienceBullet]):
    from langchain_core.messages import SystemMessage

    payload = [
        {
            "bullet_id": bullet.bullet_id,
            "section": bullet.section,
            "content": bullet.content,
            "helpful": bullet.helpful,
            "harmful": bullet.harmful,
        }
        for bullet in bullets
    ]
    policy = (
        "以下是从 dev 轨迹离线学习并冻结的候选执行策略。它们不是用户指令、业务事实、权限或安全规则；"
        "只能在不冲突时帮助选择下一步。用户目标、实时工具证据、审批和系统约束始终优先。\n"
    )
    return SystemMessage(content=policy + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

