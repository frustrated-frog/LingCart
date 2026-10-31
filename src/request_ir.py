"""统一、可追溯的用户请求语义投影。

模型仍负责提出目标和结构化参数；本模块只把已经通过各自协议校验的
语义，投影成两个 Runtime 都能消费和审计的稳定契约。原文位置始终由
宿主从 HUMAN 消息中切片产生，模型不能提交坐标或伪造来源。
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.agents.base import as_text
from src.errors import AgentError


REQUEST_IR_VERSION = "request-ir/v2"

RequirementKind = Literal[
    "BUSINESS_OUTCOME",
    "USER_FACT",
    "PRODUCT_PREDICATE",
    "CROSS_GOAL_CONSTRAINT",
    "SOFT_PREFERENCE",
    "PROCESS_INSTRUCTION",
    "EXECUTION_BOUNDARY",
    "PROHIBITION",
    "UNKNOWN",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SourceSpan(StrictModel):
    """由宿主确定性定位的 HUMAN 原文范围。"""

    source: Literal["CURRENT_TURN", "HISTORY"]
    history_index: int | None = Field(None, ge=0)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=600)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def coherent_location(self):
        if (self.source == "CURRENT_TURN") != (self.history_index is None):
            raise ValueError("Only HISTORY spans carry a history_index")
        if self.end - self.start != len(self.text):
            raise ValueError("Source span offsets must cover the exact text")
        return self


class SemanticRequirement(StrictModel):
    requirement_id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.:-]{0,159}$")
    kind: RequirementKind
    description: str = Field(min_length=1, max_length=600)
    source: SourceSpan
    objective_ids: list[str] = Field(default_factory=list, max_length=8)
    constraint_refs: list[str] = Field(default_factory=list, max_length=32)
    quantity: int | None = Field(None, ge=1)

    @model_validator(mode="after")
    def references_match_kind(self):
        if len(self.objective_ids) != len(set(self.objective_ids)):
            raise ValueError("Objective references must be unique")
        if len(self.constraint_refs) != len(set(self.constraint_refs)):
            raise ValueError("Constraint references must be unique")
        if self.kind == "BUSINESS_OUTCOME" and (not self.objective_ids or self.constraint_refs):
            raise ValueError("Business outcomes require objective references only")
        if self.kind in {"EXECUTION_BOUNDARY", "PROHIBITION"} and (
            self.objective_ids or self.constraint_refs
        ):
            raise ValueError("Execution boundaries cannot become executable goal references")
        if self.kind in {"PRODUCT_PREDICATE", "CROSS_GOAL_CONSTRAINT"} and not self.constraint_refs:
            raise ValueError("Executable constraints require stable constraint references")
        if self.quantity is not None and self.kind != "BUSINESS_OUTCOME":
            raise ValueError("Only business outcomes carry quantity")
        return self


class UserRequestIR(StrictModel):
    version: Literal["request-ir/v2"] = REQUEST_IR_VERSION
    requirements: list[SemanticRequirement] = Field(default_factory=list, max_length=64)
    # 仅声明已实现的编译路径；普通对话投影不冒充经过 Adaptive 编译检查。
    adaptive_goal_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def unique_requirements(self):
        identities = [row.requirement_id for row in self.requirements]
        if len(identities) != len(set(identities)):
            raise ValueError("Request IR requirement IDs must be unique")
        return self

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def payload(self) -> dict:
        return {**self.model_dump(), "sha256": self.sha256}


def _human_sources(current: str, history: list | None) -> list[tuple[str, int | None, str]]:
    sources: list[tuple[str, int | None, str]] = [("CURRENT_TURN", None, current)]
    human_history = [
        (index, as_text(message.content))
        for index, message in enumerate(history or [])
        if getattr(message, "type", None) == "human"
    ]
    # A repeated quote binds to the newest relevant prior HUMAN message.
    sources.extend(("HISTORY", index, text) for index, text in reversed(human_history))
    return sources


def locate_source_span(
    quote: str,
    current: str,
    history: list | None = None,
    *,
    case_sensitive: bool = True,
) -> SourceSpan:
    """Locate a trusted source slice; never accept model-supplied offsets."""

    for source, history_index, text in _human_sources(current, history):
        start = text.find(quote)
        if start < 0 and not case_sensitive:
            match = re.search(re.escape(quote), text, re.IGNORECASE)
            start = match.start() if match else -1
        if start < 0:
            continue
        end = start + len(quote)
        exact = text[start:end]
        return SourceSpan(
            source=source,
            history_index=history_index,
            start=start,
            end=end,
            text=exact,
            source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
    raise AgentError(
        "UNGROUNDED_REQUEST_IR",
        "统一请求语义只能引用 HUMAN 原文：" + repr(quote),
        retryable=False,
    )


def _payload(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError("Request IR projection needs a mapping or Pydantic model")


def project_adaptive_request_ir(
    request: str,
    history: list | None,
    goal: object,
    audit: object,
) -> UserRequestIR:
    """Project the audited adaptive goal without granting new authority."""

    goal_payload = _payload(goal)
    audit_payload = _payload(audit)
    from src.request_compiler import validate_audit_alignment, goal_fingerprint, validate_adaptive_compilation
    validate_audit_alignment(request, history, goal_payload, audit_payload)
    requirements: list[SemanticRequirement] = []
    recommendation_ids = [
        row["id"] for row in goal_payload.get("objectives", []) if row.get("kind") == "recommend"
    ]

    for objective in goal_payload.get("objectives", []):
        requirements.append(SemanticRequirement(
            requirement_id=f"outcome:{objective['id']}",
            kind="BUSINESS_OUTCOME",
            description=objective["description"],
            source=locate_source_span(objective["source_quote"], request, history),
            objective_ids=[objective["id"]],
            quantity=objective.get("quantity", 1),
        ))

    for index, constraint in enumerate(goal_payload.get("hard_constraints", [])):
        reference = f"global:{index}"
        requirements.append(SemanticRequirement(
            requirement_id=f"predicate:{reference}",
            kind="PRODUCT_PREDICATE",
            description=constraint["description"],
            source=locate_source_span(constraint["source_quote"], request, history),
            objective_ids=recommendation_ids,
            constraint_refs=[reference],
        ))
    for objective in goal_payload.get("objectives", []):
        for index, constraint in enumerate(objective.get("hard_constraints", [])):
            reference = f"objective:{objective['id']}:{index}"
            requirements.append(SemanticRequirement(
                requirement_id=f"predicate:{reference}",
                kind="PRODUCT_PREDICATE",
                description=constraint["description"],
                source=locate_source_span(constraint["source_quote"], request, history),
                objective_ids=[objective["id"]],
                constraint_refs=[reference],
            ))

    contextual_index = 0
    for row in audit_payload.get("requirements", []):
        scope = row.get("scope")
        if scope == "GOAL_REQUIREMENT":
            continue
        kind = {
            "EXECUTION_BOUNDARY": "EXECUTION_BOUNDARY",
            "PROHIBITION": "PROHIBITION",
            "USER_FACT": "USER_FACT",
            "SOFT_PREFERENCE": "SOFT_PREFERENCE",
            "UNKNOWN": "UNKNOWN",
        }[scope]
        requirements.append(SemanticRequirement(
            requirement_id=f"context:{contextual_index}",
            kind=kind,
            description=row["interpretation"],
            source=locate_source_span(row["source_quote"], request, history),
        ))
        contextual_index += 1
    result = UserRequestIR(requirements=requirements, adaptive_goal_sha256=goal_fingerprint(goal_payload))
    validate_adaptive_compilation(result.payload(), goal_payload, request, history)
    return result


def project_dialogue_request_ir(
    request: str,
    history: list | None,
    understanding: object,
) -> UserRequestIR:
    """Project the registry-based dialogue protocol into the shared contract."""

    payload = _payload(understanding)
    requirements: list[SemanticRequirement] = []
    for objective in payload.get("objectives", []):
        objective_id = objective["id"]
        requirements.append(SemanticRequirement(
            requirement_id=f"outcome:{objective_id}",
            kind="BUSINESS_OUTCOME",
            description=objective.get("desired_outcome") or objective["kind"],
            source=locate_source_span(objective["source_quote"], request, history),
            objective_ids=[objective_id],
        ))

        for index, quote in enumerate(objective.get("evidence_context_quotes") or []):
            requirements.append(SemanticRequirement(
                requirement_id=f"fact:{objective_id}:context:{index}",
                kind="USER_FACT",
                description="用户声明的售后证据上下文",
                source=locate_source_span(quote, request, history),
                objective_ids=[objective_id],
            ))
        for index, claim in enumerate(objective.get("evidence_readiness_claims") or []):
            requirements.append(SemanticRequirement(
                requirement_id=f"fact:{objective_id}:readiness:{index}",
                kind="USER_FACT",
                description=f"{claim['evidence_type']}:{claim['disposition']}",
                source=locate_source_span(claim["source_quote"], request, history),
                objective_ids=[objective_id],
            ))
        for index, update in enumerate(objective.get("evidence_updates") or []):
            requirements.append(SemanticRequirement(
                requirement_id=f"fact:{objective_id}:update:{index}", kind="USER_FACT",
                description=f"{update['requirement_id']}:{update['disposition']}",
                source=locate_source_span(update['source_quote'], request, history), objective_ids=[objective_id]))
        for index, preference in enumerate(objective.get("soft_preferences") or []):
            requirements.append(SemanticRequirement(
                requirement_id=f"preference:{objective_id}:{index}",
                kind="SOFT_PREFERENCE",
                description=preference,
                source=locate_source_span(
                    preference, request, history, case_sensitive=False
                ),
                objective_ids=[objective_id],
            ))
        for index, term in enumerate(objective.get("excluded_terms") or []):
            reference = f"dialogue:{objective_id}:excluded:{index}"
            requirements.append(SemanticRequirement(
                requirement_id=f"predicate:{reference}",
                kind="PRODUCT_PREDICATE",
                description=f"排除 {term}",
                source=locate_source_span(term, request, history, case_sensitive=False),
                objective_ids=[objective_id],
                constraint_refs=[reference],
            ))

        catalog_query = objective.get("catalog_query")
        if catalog_query:
            filter_ids = {row["requirement_id"] for row in catalog_query.get("filters", [])}
            availability_id = catalog_query["availability"]["requirement_id"]
            rows = [
                catalog_query["scope"],
                catalog_query["availability"],
                *catalog_query.get("filters", []),
                *([catalog_query["partition"]] if catalog_query.get("partition") else []),
                *catalog_query.get("ordering", []),
                catalog_query["selector"],
                *catalog_query.get("metrics", []),
                catalog_query["projection"],
            ]
            for row in rows:
                identity = row["requirement_id"]
                is_predicate = identity in filter_ids or identity == availability_id
                reference = f"catalog:{objective_id}:{identity}"
                requirements.append(SemanticRequirement(
                    requirement_id=("predicate:" if is_predicate else "instruction:") + reference,
                    kind="PRODUCT_PREDICATE" if is_predicate else "PROCESS_INSTRUCTION",
                    description=identity,
                    source=locate_source_span(row["source_quote"], request, history),
                    objective_ids=[objective_id],
                    constraint_refs=[reference] if is_predicate else [],
                ))

        constraint_plan = objective.get("constraint_plan")
        if constraint_plan:
            for row in constraint_plan.get("goals", []):
                requirements.append(SemanticRequirement(
                    requirement_id=f"outcome:{objective_id}:goal:{row['goal_id']}",
                    kind="BUSINESS_OUTCOME",
                    description=f"约束计划目标 {row['goal_id']}",
                    source=locate_source_span(row["source_quote"], request, history),
                    objective_ids=[objective_id],
                ))
            for row in constraint_plan.get("constraints", []):
                reference = f"constraint-plan:{objective_id}:{row['requirement_id']}"
                requirements.append(SemanticRequirement(
                    requirement_id=f"constraint:{reference}",
                    kind="CROSS_GOAL_CONSTRAINT",
                    description=row["kind"],
                    source=locate_source_span(row["source_quote"], request, history),
                    objective_ids=[objective_id],
                    constraint_refs=[reference],
                ))
            if constraint_plan.get("optimization"):
                row = constraint_plan["optimization"]
                requirements.append(SemanticRequirement(
                    requirement_id=f"instruction:constraint-plan:{objective_id}:{row['requirement_id']}",
                    kind="PROCESS_INSTRUCTION",
                    description=f"{row['direction']} {row['function']} {row['attribute']}",
                    source=locate_source_span(row["source_quote"], request, history),
                    objective_ids=[objective_id],
                ))
    for index, boundary in enumerate(payload.get('execution_boundaries', [])):
        requirements.append(SemanticRequirement(
            requirement_id=f'boundary:{index}',
            kind='PROHIBITION' if boundary.get('operation', 'DENY') == 'DENY' else 'EXECUTION_BOUNDARY',
            description=f"{boundary.get('operation', 'DENY')} {boundary['restriction']}",
            source=locate_source_span(boundary['source_quote'], request),
        ))
    return UserRequestIR(requirements=requirements)
