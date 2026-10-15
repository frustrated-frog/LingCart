"""Machine-checkable contract for one baseline/candidate Harness experiment."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChangeDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    kind: Literal["mechanism", "engine", "prompt", "runtime"]
    slot: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9._-]*$")
    implementation_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9._-]*$")
    paths: tuple[str, ...] = ()

    @model_validator(mode="after")
    def target_matches_kind(self):
        if self.kind == "mechanism" and (not self.slot or not self.implementation_id):
            raise ValueError("机制变更必须声明 slot 和 implementation_id")
        if self.kind == "engine" and (self.slot is not None or not self.implementation_id):
            raise ValueError("引擎变更只声明 implementation_id")
        if self.kind in {"prompt", "runtime"} and (self.slot is not None or self.implementation_id is not None):
            raise ValueError("Prompt 或 Runtime 变更通过 paths 声明范围")
        if not self.paths:
            raise ValueError("每项实验变更必须声明允许修改的文件")
        if len(self.paths) != len(set(self.paths)):
            raise ValueError("同一变更不能重复声明文件")
        if any(path.startswith("/") or "\\" in path or PurePosixPath(path).as_posix() != path
               or ".." in path.split("/") or not path for path in self.paths):
            raise ValueError("实验路径必须是仓库内的规范相对路径")
        return self


class MetricRule(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_.]*$", min_length=1, max_length=80)
    objective: Literal["maximize", "minimize"]
    max_regression: float = Field(default=0, ge=0)
    min_improvement: float = Field(default=0, ge=0)


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1]
    experiment_id: str = Field(pattern=r"^[a-z][a-z0-9-]*$", min_length=1, max_length=100)
    baseline_profile: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    candidate_profile: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    dataset: str = Field(pattern=r"^eval/[a-z][a-z0-9_-]*\.json$")
    suite: Literal["support", "adaptive"]
    repeats: int = Field(ge=1, le=10)
    tuning_split: Literal["dev"] = "dev"
    evaluation_split: Literal["test"] = "test"
    changes: tuple[ChangeDeclaration, ...] = Field(min_length=1)
    primary_metrics: tuple[MetricRule, ...] = Field(min_length=1)
    guard_metrics: tuple[MetricRule, ...] = Field(min_length=1)
    compatibility_review: str | None = Field(default=None, min_length=1, max_length=240)

    @model_validator(mode="after")
    def contract_is_complete(self):
        if self.baseline_profile == self.candidate_profile:
            raise ValueError("候选 Profile 必须不同于基线")
        if self.suite == "adaptive" and not self.dataset.endswith("adaptive_tasks.json"):
            raise ValueError("自主实验必须使用独立 adaptive_tasks 任务集")
        names = [rule.name for rule in (*self.primary_metrics, *self.guard_metrics)]
        if len(names) != len(set(names)):
            raise ValueError("同一指标不能重复声明")
        targets = [(change.kind, change.slot, change.implementation_id) for change in self.changes]
        if len(targets) != len(set(targets)):
            raise ValueError("同一实验变更目标不能重复声明")
        guards = {rule.name for rule in self.guard_metrics}
        required = {"pass_at_1", "all_repeats_success_rate", "safety_violations", "protocol_violations"}
        if not required <= guards:
            raise ValueError("保护指标必须包含正确率、稳定性、安全与协议违规")
        expected_objectives = {
            "pass_at_1": "maximize",
            "all_repeats_success_rate": "maximize",
            "safety_violations": "minimize",
            "protocol_violations": "minimize",
        }
        if any(rule.objective != expected_objectives.get(rule.name, rule.objective)
               or (rule.name in required and rule.max_regression != 0)
               for rule in self.guard_metrics):
            raise ValueError("必选保护指标的方向必须正确且不允许退化")
        cost_metrics = {"tokens.total_tokens", "model_attempts", "tool_calls", "latency_ms.mean",
                        "latency_ms.p50", "latency_ms.p95", "monetary_cost"}
        cost_rules = [rule for rule in (*self.primary_metrics, *self.guard_metrics) if rule.name in cost_metrics]
        if not cost_rules or any(rule.objective != "minimize" for rule in cost_rules):
            raise ValueError("实验必须声明至少一项最小化的成本或延迟指标")
        if any(change.kind == "runtime" for change in self.changes) and not self.compatibility_review:
            raise ValueError("Runtime 实验必须提供独立兼容性审查编号")
        return self

    @classmethod
    def from_path(cls, path):
        return cls.model_validate_json(path.read_text())

