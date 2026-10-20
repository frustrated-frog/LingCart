"""Versioned data contracts shared by LivePlan monitoring and intervention."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


SignalCode = Literal[
    "ACTION_OSCILLATION",
    "PHASE_STAGNATION",
    "EVIDENCE_STAGNATION",
    "VERIFICATION_LOOP",
    "BUDGET_RISK",
]
AdviceAction = Literal[
    "CHANGE_APPROACH",
    "INSPECT_EVIDENCE",
    "REASSESS_CONSTRAINTS",
    "VERIFY_GOAL",
    "CLARIFY",
    "FINISH_HONESTLY",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ProgressVector(StrictModel):
    goal_revision: int = Field(ge=0)
    verified_objectives: int = Field(ge=0)
    pending_objectives: int = Field(ge=0)
    binding_revision: int = Field(ge=0)
    distinct_evidence: int = Field(ge=0)
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    remaining_model_calls: int = Field(ge=0)
    remaining_tool_calls: int = Field(ge=0)

    def semantic_key(self) -> tuple:
        """Budget burn is observable but is not itself semantic progress."""

        return (
            self.goal_revision,
            self.verified_objectives,
            self.pending_objectives,
            self.binding_revision,
            self.distinct_evidence,
            self.evidence_sha256,
        )


class TrajectoryAction(StrictModel):
    sequence: int = Field(ge=1)
    action: str = Field(min_length=1, max_length=80)
    phase: str = Field(min_length=1, max_length=40)
    signature: str = Field(pattern=r"^[a-f0-9]{64}$")
    step_id: str | None = Field(default=None, max_length=80)
    status: str | None = Field(default=None, max_length=80)


class DriftSignal(StrictModel):
    code: SignalCode
    severity: Literal["warning", "blocking"]
    evidence_sequences: list[int] = Field(min_length=1, max_length=16)
    detail: str = Field(min_length=1, max_length=500)


class MonitorAssessment(StrictModel):
    evaluation_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    basis_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    basis_sequence: int = Field(ge=1)
    action_count: int = Field(ge=1)
    current_phase: str = Field(min_length=1, max_length=40)
    latest_action_signature: str = Field(pattern=r"^[a-f0-9]{64}$")
    progress: ProgressVector
    signals: list[DriftSignal] = Field(min_length=1, max_length=5)


class CorrectiveAdvice(StrictModel):
    recommended_action: AdviceAction
    target_objective_ids: list[str] = Field(default_factory=list, max_length=8)
    reason_code: SignalCode
    guidance: str = Field(min_length=1, max_length=480)
    confidence: float = Field(ge=0, le=1)


class Intervention(StrictModel):
    intervention_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    drift_id: str = Field(pattern=r"^[a-f0-9]{24}$")
    mode: Literal["ADVISE_NEXT_STEP", "CONSTRAIN_NEXT_ACTION"]
    advice: CorrectiveAdvice
    blocked_action_signature: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    block_count: int = Field(default=0, ge=0)
    max_blocks: int = Field(default=0, ge=0, le=4)

    @model_validator(mode="after")
    def blocking_mode_has_a_target(self):
        if self.mode == "CONSTRAIN_NEXT_ACTION" and not self.blocked_action_signature:
            raise ValueError("Blocking intervention requires an action signature")
        if self.mode == "ADVISE_NEXT_STEP" and self.blocked_action_signature is not None:
            raise ValueError("Advisory intervention cannot block an action")
        return self


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)


def fingerprint(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def short_id(value) -> str:
    return fingerprint(value)[:24]
