"""Strict contracts for semantic evaluation datasets, stages and gate receipts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ValueAssertion(StrictFrozenModel):
    path: str = Field(min_length=1, max_length=160)
    operator: Literal["eq", "present", "contains_all", "set_eq"] = "eq"
    value: Any = None
    origin: Literal["latest_turn", "history", "runtime", "tool_result"] | None = None

    @model_validator(mode="after")
    def value_matches_operator(self):
        if self.operator == "present" and self.value is not None:
            raise ValueError("present assertion must not carry a value")
        if self.operator != "present" and self.value is None:
            raise ValueError(f"{self.operator} assertion requires a value")
        if self.operator in {"contains_all", "set_eq"} and not isinstance(self.value, list):
            raise ValueError(f"{self.operator} assertion requires a JSON array")
        return self


class OperationOracle(StrictFrozenModel):
    operation: str = Field(min_length=1, max_length=100)
    args: tuple[ValueAssertion, ...] = ()
    result: tuple[ValueAssertion, ...] = ()
    min_calls: int = Field(default=1, ge=0, le=20)
    max_calls: int = Field(default=1, ge=0, le=20)
    accepted_statuses: tuple[str, ...] = ("SUCCEEDED", "REUSED")

    @model_validator(mode="after")
    def valid_cardinality(self):
        if self.max_calls < self.min_calls:
            raise ValueError("operation max_calls must be >= min_calls")
        return self


class GoalOracle(StrictFrozenModel):
    oracle_goal_id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,79}$")
    objective_kind: str = Field(min_length=1, max_length=100)
    objective: tuple[ValueAssertion, ...] = ()
    capability_id: str | None = Field(default=None, min_length=1, max_length=160)
    required_operations: tuple[OperationOracle, ...] = ()
    outcome: str | None = Field(default=None, min_length=1, max_length=80)


class CommandOracle(StrictFrozenModel):
    required_all: tuple[str, ...] = ()
    required_any: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    allow_extra: bool = False

    @model_validator(mode="after")
    def disjoint_commands(self):
        required = set(self.required_all) | set(self.required_any)
        overlap = required & set(self.forbidden)
        if overlap:
            raise ValueError(f"commands cannot be both required and forbidden: {sorted(overlap)}")
        return self


class TerminalOracle(StrictFrozenModel):
    statuses: tuple[str, ...] = Field(min_length=1)
    reason: str | None = Field(default=None, min_length=1, max_length=100)
    all_goals_satisfied: bool | None = None


class SemanticOracle(StrictFrozenModel):
    goal_set: Literal["exact", "contains"] = "exact"
    goals: tuple[GoalOracle, ...] = ()
    commands: CommandOracle = CommandOracle()
    forbidden_operations: tuple[str, ...] = ()
    terminal: TerminalOracle

    @model_validator(mode="after")
    def unique_goal_contracts(self):
        ids = [goal.oracle_goal_id for goal in self.goals]
        if len(ids) != len(set(ids)):
            raise ValueError("oracle_goal_id values must be unique")
        signatures = [
            fingerprint({
                "kind": goal.objective_kind,
                "objective": [item.model_dump(mode="json") for item in goal.objective],
            })
            for goal in self.goals
        ]
        if len(signatures) != len(set(signatures)):
            raise ValueError("same-kind goals must contain enough assertions to be distinguishable")
        required_operations = {
            operation.operation for goal in self.goals for operation in goal.required_operations
        }
        overlap = required_operations & set(self.forbidden_operations)
        if overlap:
            raise ValueError(f"operations cannot be both required and forbidden: {sorted(overlap)}")
        return self


class ExecutionPolicy(StrictFrozenModel):
    semantic_path: Literal["required", "allowed", "forbidden"] = "required"
    side_effects: Literal["forbidden", "allowed"] = "forbidden"
    trace_integrity: Literal["required", "best_effort"] = "required"


class HistoryTurn(StrictFrozenModel):
    role: Literal["human", "assistant"]
    content: str = Field(min_length=1, max_length=12000)


class SemanticCase(StrictFrozenModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,99}$")
    split: Literal["smoke", "acceptance"] = "acceptance"
    risk_group: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=12000)
    history: tuple[HistoryTurn, ...] = ()
    execution_policy: ExecutionPolicy = ExecutionPolicy()
    oracle: SemanticOracle


class SemanticDataset(StrictFrozenModel):
    schema_version: Literal[2]
    dataset_id: str = Field(min_length=1, max_length=100)
    dataset_version: str = Field(min_length=1, max_length=40)
    cases: tuple[SemanticCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self):
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("semantic case IDs must be unique")
        return self


class StageExecution(StrictFrozenModel):
    mode: Literal["auto", "adaptive"]
    profile_id: str = Field(min_length=1, max_length=100)


class StageBudget(StrictFrozenModel):
    max_executions: int = Field(ge=1, le=10000)
    max_model_calls_per_run: int = Field(ge=1, le=100)


class StageSelection(StrictFrozenModel):
    case_ids: tuple[str, ...] = ()
    all_cases: bool = False

    @model_validator(mode="after")
    def exactly_one_selector(self):
        if bool(self.case_ids) == self.all_cases:
            raise ValueError("stage selection must use case_ids or all_cases")
        return self


class AcceptanceStage(StrictFrozenModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,79}$")
    gate_type: Literal["smoke", "acceptance"]
    claim_scope: Literal["selected_cases_only", "frozen_stage"]
    selection: StageSelection
    executions: tuple[StageExecution, ...] = Field(min_length=1)
    repeats: int = Field(ge=1, le=20)
    budget: StageBudget
    requires: tuple[str, ...] = ()

    @model_validator(mode="after")
    def gate_scope_matches(self):
        expected = "selected_cases_only" if self.gate_type == "smoke" else "frozen_stage"
        if self.claim_scope != expected:
            raise ValueError(f"{self.gate_type} stage must use claim_scope={expected}")
        modes = [item.mode for item in self.executions]
        if len(modes) != len(set(modes)):
            raise ValueError("stage execution modes must be unique")
        return self


class DatasetReference(StrictFrozenModel):
    path: str = Field(min_length=1, max_length=300)
    dataset_id: str = Field(min_length=1, max_length=100)
    dataset_version: str = Field(min_length=1, max_length=40)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    case_count: int = Field(ge=1)


class OracleProtocol(StrictFrozenModel):
    id: Literal["semantic-goal-oracle"]
    version: int = Field(ge=1)


class AcceptanceSpec(StrictFrozenModel):
    schema_version: Literal[2]
    experiment_id: str = Field(min_length=1, max_length=100)
    dataset: DatasetReference
    oracle_protocol: OracleProtocol
    stages: tuple[AcceptanceStage, ...] = Field(min_length=1)
    progression_rule: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def valid_stage_graph(self):
        ids = [stage.id for stage in self.stages]
        if len(ids) != len(set(ids)):
            raise ValueError("acceptance stage IDs must be unique")
        known: set[str] = set()
        for stage in self.stages:
            missing = set(stage.requires) - known
            if missing:
                raise ValueError(f"stage {stage.id} requires later or unknown stages: {sorted(missing)}")
            known.add(stage.id)
        return self

    def stage(self, stage_id: str) -> AcceptanceStage:
        for stage in self.stages:
            if stage.id == stage_id:
                return stage
        raise ValueError(f"未知语义验收阶段：{stage_id}")


class ExperimentMetricRule(StrictFrozenModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_.]*$", min_length=1, max_length=100)
    objective: Literal["maximize", "minimize"]
    max_regression: float = Field(default=0, ge=0)
    min_improvement: float = Field(default=0, ge=0)


class SemanticExperimentSpec(StrictFrozenModel):
    """Frozen admission policy for a cross-Profile semantic experiment."""

    schema_version: Literal[1]
    experiment_id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,99}$")
    baseline_profiles: tuple[str, ...] = Field(min_length=1)
    candidate_profiles: tuple[str, ...] = Field(min_length=1)
    primary_metrics: tuple[ExperimentMetricRule, ...] = Field(min_length=1)
    guard_metrics: tuple[ExperimentMetricRule, ...] = ()

    @model_validator(mode="after")
    def valid_comparison(self):
        if set(self.baseline_profiles) == set(self.candidate_profiles):
            raise ValueError("候选实验必须改变至少一个 Profile")
        if len(self.baseline_profiles) != len(set(self.baseline_profiles)):
            raise ValueError("baseline_profiles 不能重复")
        if len(self.candidate_profiles) != len(set(self.candidate_profiles)):
            raise ValueError("candidate_profiles 不能重复")
        rules = (*self.primary_metrics, *self.guard_metrics)
        names = [rule.name for rule in rules]
        if len(names) != len(set(names)):
            raise ValueError("实验指标名称不能重复")
        if not any(rule.min_improvement > 0 for rule in self.primary_metrics):
            raise ValueError("至少一个主指标必须声明正向最小改善，才能形成收益结论")
        return self


class GateReceipt(StrictFrozenModel):
    schema_version: Literal[1] = 1
    gate_type: Literal[
        "SMOKE", "ACCEPTANCE", "EXPERIMENT_COMPATIBILITY", "EXPERIMENT_EFFECT", "EXPERIMENT",
    ]
    stage_id: str
    claim_scope: Literal["selected_cases_only", "frozen_stage", "paired_experiment"]
    checked_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    passed: bool
    failure_codes: tuple[str, ...] = ()
    failures: tuple[dict, ...] = ()
    evidence: dict = Field(default_factory=dict)


def load_dataset_payload(payload: dict) -> SemanticDataset:
    # Strict tuple fields still accept JSON arrays, which is the persisted
    # contract representation.  Re-encoding also rejects NaN/Infinity here.
    return SemanticDataset.model_validate_json(canonical_json(payload))


def load_acceptance_spec_payload(payload: dict) -> AcceptanceSpec:
    return AcceptanceSpec.model_validate_json(canonical_json(payload))
