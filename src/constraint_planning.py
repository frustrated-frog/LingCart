"""可扩展的有限域约束规划协议、求解器与独立全局核验。

模型只声明目标图和约束 IR。候选事实由注册的数据域 Provider 解析，
可行性与最终目标满足性由确定性求解器和核验器负责。
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import hashlib
from itertools import combinations_with_replacement
import json
import math
import re
from typing import Annotated, Callable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.catalog_query import (
    CatalogQuerySpec,
    compile_catalog_query,
    query_catalog,
    validate_catalog_query_grounding,
    verify_catalog_result,
)
from src.errors import AgentError


CONSTRAINT_PLAN = "CONSTRAINT_PLAN"
CONSTRAINT_PLAN_VERSION = "constraint-plan/v1"
CONSTRAINT_PLAN_OPERATION = "solve_constraint_plan"
MAX_DOMAIN_CANDIDATES = 100
MAX_GOAL_OPTIONS = 2_000
MAX_SEARCH_NODES = 100_000
MAX_STORED_FEASIBLE = 5_000
MAX_DELIVERED_SOLUTIONS = 3

Scalar = str | int | float
ComparisonOperator = Literal["EQ", "NE", "LT", "LTE", "GT", "GTE"]
Reducer = Literal["FIRST", "COUNT", "SUM", "MIN", "MAX", "AVG"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Requirement(StrictModel):
    requirement_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    source_quote: str = Field(min_length=1, max_length=300)


class CandidateDomainSpec(Requirement):
    provider: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")
    parameters: dict[str, object] = Field(default_factory=dict, max_length=8)


class GoalNode(Requirement):
    goal_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    quantity: int = Field(1, ge=1, le=4)
    domain: CandidateDomainSpec


class AggregateConstraint(Requirement):
    kind: Literal["AGGREGATE"]
    scope_goal_ids: list[str] = Field(min_length=1, max_length=8)
    attribute: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]{0,79}$")
    function: Literal["SUM", "MIN", "MAX", "AVG", "COUNT"]
    operator: ComparisonOperator
    value: Scalar


class GoalOperand(StrictModel):
    goal_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    attribute: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]{0,79}$")
    reducer: Reducer = "FIRST"


class ComparisonConstraint(Requirement):
    kind: Literal["COMPARISON"]
    left: GoalOperand
    operator: ComparisonOperator
    right: GoalOperand


class AllDifferentConstraint(Requirement):
    kind: Literal["ALL_DIFFERENT"]
    scope_goal_ids: list[str] = Field(min_length=2, max_length=8)
    attribute: str = Field("candidate_id", pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]{0,79}$")


HardConstraint = Annotated[
    AggregateConstraint | ComparisonConstraint | AllDifferentConstraint,
    Field(discriminator="kind"),
]


class OptimizationDirective(Requirement):
    direction: Literal["MIN", "MAX"]
    scope_goal_ids: list[str] = Field(min_length=1, max_length=8)
    attribute: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]{0,79}$")
    function: Literal["SUM", "MIN", "MAX", "AVG", "COUNT"]


class ConstraintPlanSpec(StrictModel):
    version: Literal["constraint-plan/v1"]
    goals: list[GoalNode] = Field(min_length=2, max_length=8)
    constraints: list[HardConstraint] = Field(default_factory=list, max_length=16)
    optimization: OptimizationDirective | None = None

    @model_validator(mode="after")
    def coherent_graph(self):
        goal_ids = [goal.goal_id for goal in self.goals]
        if len(goal_ids) != len(set(goal_ids)):
            raise ValueError("Goal IDs must be unique")
        requirement_ids = [goal.requirement_id for goal in self.goals]
        requirement_ids += [constraint.requirement_id for constraint in self.constraints]
        if self.optimization:
            requirement_ids.append(self.optimization.requirement_id)
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("Top-level constraint-plan requirement IDs must be unique")
        known = set(goal_ids)
        for constraint in self.constraints:
            if isinstance(constraint, ComparisonConstraint):
                referenced = {constraint.left.goal_id, constraint.right.goal_id}
            else:
                referenced = set(constraint.scope_goal_ids)
            if not referenced <= known:
                raise ValueError("Constraint references an unknown goal")
            if isinstance(constraint, AggregateConstraint) and constraint.function == "COUNT":
                if constraint.attribute != "candidate_id":
                    raise ValueError("COUNT must use candidate_id")
            if isinstance(constraint, AggregateConstraint) and constraint.function != "COUNT":
                if isinstance(constraint.value, str):
                    raise ValueError("Numeric aggregates require a numeric comparison value")
        if self.optimization:
            if not set(self.optimization.scope_goal_ids) <= known:
                raise ValueError("Optimization references an unknown goal")
            if self.optimization.function == "COUNT" and self.optimization.attribute != "candidate_id":
                raise ValueError("COUNT optimization must use candidate_id")
        return self


@dataclass(frozen=True)
class ResourceClaim:
    """A candidate's consumption of a globally identified finite resource."""

    resource_id: str
    capacity: int
    units: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.resource_id, str) or not self.resource_id.strip():
            raise ValueError("Resource ID must not be empty")
        if not isinstance(self.capacity, int) or isinstance(self.capacity, bool) or self.capacity < 0:
            raise ValueError("Resource capacity must be non-negative")
        if not isinstance(self.units, int) or isinstance(self.units, bool) or self.units < 1:
            raise ValueError("Resource units must be positive")


@dataclass(frozen=True)
class CandidateValue:
    candidate_id: str
    label: str
    attributes: Mapping[str, Scalar]
    evidence: Mapping[str, object] = field(default_factory=dict)
    # Kept as a provider-compatibility bridge. New providers should emit a
    # globally stable ResourceClaim so overlapping domains can share capacity.
    max_quantity: int | None = None
    resource_claims: tuple[ResourceClaim, ...] = ()

    def __post_init__(self) -> None:
        resource_ids = [claim.resource_id for claim in self.resource_claims]
        if len(resource_ids) != len(set(resource_ids)):
            raise ValueError("Candidate resource claims must have unique IDs")
        if (
            self.max_quantity is not None
            and (
                not isinstance(self.max_quantity, int)
                or isinstance(self.max_quantity, bool)
                or self.max_quantity < 0
            )
        ):
            raise ValueError("Candidate max quantity must be non-negative")


@dataclass(frozen=True)
class DomainResolution:
    provider: str
    provider_fingerprint: str
    candidates: tuple[CandidateValue, ...]
    evidence: Mapping[str, object]
    attribute_schema: tuple[str, ...] = ()


ParameterValidator = Callable[[Mapping[str, object]], Mapping[str, object]]
DomainResolver = Callable[[Mapping[str, object]], DomainResolution]
GroundingValidator = Callable[[Mapping[str, object], list[str]], None]
RequirementExtractor = Callable[[Mapping[str, object]], list[tuple[str, str, str]]]


@dataclass(frozen=True)
class DomainProviderRegistration:
    provider_id: str
    description: str
    parameter_schema: Mapping[str, object]
    validate_parameters: ParameterValidator
    resolve: DomainResolver
    validate_grounding: GroundingValidator
    attribute_schema: tuple[str, ...] = ()
    extract_requirements: RequirementExtractor = lambda _parameters: []

    @property
    def fingerprint(self) -> str:
        payload = {
            "provider_id": self.provider_id,
            "description": self.description,
            "parameter_schema": self.parameter_schema,
            "validator": _callable_identity(self.validate_parameters),
            "resolver": _callable_identity(self.resolve),
            "grounding": _callable_identity(self.validate_grounding),
            "attribute_schema": self.attribute_schema,
            "requirements": _callable_identity(self.extract_requirements),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class DomainProviderRegistry:
    """数据域适配注册表；求解器本身不包含商品或类目知识。"""

    def __init__(self, registrations: tuple[DomainProviderRegistration, ...] = ()):
        self._registrations: dict[str, DomainProviderRegistration] = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: DomainProviderRegistration) -> None:
        if registration.provider_id in self._registrations:
            raise ValueError(f"Duplicated domain provider: {registration.provider_id}")
        self._registrations[registration.provider_id] = registration

    def registration(self, provider_id: str) -> DomainProviderRegistration:
        try:
            return self._registrations[provider_id]
        except KeyError as exc:
            raise AgentError(
                "UNKNOWN_DOMAIN_PROVIDER",
                f"约束目标引用了未注册数据域：{provider_id}",
                retryable=False,
            ) from exc

    def validate(self, domain: CandidateDomainSpec) -> Mapping[str, object]:
        registration = self.registration(domain.provider)
        try:
            return registration.validate_parameters(domain.parameters)
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError("INVALID_DOMAIN_PARAMETERS", "候选数据域参数无效", retryable=False) from exc

    def resolve(self, domain: CandidateDomainSpec) -> DomainResolution:
        registration = self.registration(domain.provider)
        parameters = self.validate(domain)
        resolution = registration.resolve(parameters)
        if resolution.provider != domain.provider or resolution.provider_fingerprint != registration.fingerprint:
            raise AgentError("DOMAIN_PROVIDER_EVIDENCE_MISMATCH", "数据域返回的身份与注册表不一致", retryable=False)
        identities = [candidate.candidate_id for candidate in resolution.candidates]
        if len(identities) != len(set(identities)):
            raise AgentError("INVALID_DOMAIN_RESULT", "数据域返回了重复候选身份", retryable=False)
        if len(identities) > MAX_DOMAIN_CANDIDATES:
            raise AgentError("DOMAIN_CANDIDATE_BUDGET_EXHAUSTED", "候选数据域规模超过安全上限", retryable=False)
        return resolution

    def prompt_catalog(self) -> list[dict]:
        return [
            {
                "provider": item.provider_id,
                "description": item.description,
                "parameters": item.parameter_schema,
                "attributes": list(item.attribute_schema),
                "fingerprint": item.fingerprint,
            }
            for item in self._registrations.values()
        ]

    @property
    def version(self) -> str:
        payload = [(key, value.fingerprint) for key, value in sorted(self._registrations.items())]
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _callable_identity(value: object) -> str:
    value = getattr(value, "function", value)
    return f"{getattr(value, '__module__', type(value).__module__)}:{getattr(value, '__qualname__', type(value).__qualname__)}"


def _catalog_parameters(parameters: Mapping[str, object]) -> Mapping[str, object]:
    if set(parameters) != {"query"}:
        raise AgentError("INVALID_DOMAIN_PARAMETERS", "CATALOG 数据域只接受 query 参数", retryable=False)
    spec = CatalogQuerySpec.model_validate(parameters["query"], strict=True)
    if spec.selector.mode != "ALL" or spec.metrics or spec.partition is not None:
        raise AgentError(
            "INVALID_DOMAIN_QUERY",
            "约束规划的候选数据域必须使用无损 ALL 查询，不能聚合、分页或分组",
            retryable=False,
        )
    return {"query": spec.model_dump(mode="json")}


def _catalog_grounding(parameters: Mapping[str, object], sources: list[str]) -> None:
    normalized = _catalog_parameters(parameters)
    validate_catalog_query_grounding(CatalogQuerySpec.model_validate(normalized["query"], strict=True), sources)


def _catalog_requirements(parameters: Mapping[str, object]) -> list[tuple[str, str, str]]:
    normalized = _catalog_parameters(parameters)
    query = CatalogQuerySpec.model_validate(normalized["query"], strict=True)
    return [
        (requirement.requirement_id, requirement.source_quote, f"query.requirements.{index}")
        for index, requirement in enumerate(query.requirements())
    ]


def _catalog_resolution(parameters: Mapping[str, object]) -> DomainResolution:
    normalized = _catalog_parameters(parameters)
    arguments = compile_catalog_query(normalized["query"])
    result = query_catalog(**arguments)
    verify_catalog_result(result, arguments)
    artifact = result["artifact"]
    candidates = tuple(
        CandidateValue(
            candidate_id=row["product_id"],
            label=row["name"],
            attributes={
                "candidate_id": row["product_id"],
                "name": row["name"],
                "category": row["category"],
                "price": row["price"],
                "stock": row["stock"],
                "summary": row["summary"],
                **{
                    key: value for key, value in (row.get("values") or {}).items()
                    if isinstance(value, (str, int, float)) and not isinstance(value, bool)
                },
            },
            evidence={
                "product_id": row["product_id"],
                "name": row["name"],
                "category": row["category"],
                "price": row["price"],
                "currency": row.get("currency", "CNY"),
                "summary": row["summary"],
                "stock": row["stock"],
                "specs": row.get("specs") or {},
                "data_source": row.get("data_source", "DEMO_CATALOG"),
            },
            resource_claims=(ResourceClaim(
                resource_id=f"catalog.product:{row['product_id']}",
                capacity=int(row["stock"]),
            ),),
        )
        for row in artifact["products"]
    )
    registration = get_domain_provider_registry().registration("CATALOG")
    return DomainResolution(
        provider="CATALOG",
        provider_fingerprint=registration.fingerprint,
        candidates=candidates,
        evidence={
            "result_set_id": artifact["result_set_id"],
            "catalog_snapshot": artifact["catalog_snapshot"],
            "catalog_data_fingerprint": artifact["catalog_data_fingerprint"],
            "result_data_fingerprint": artifact["result_data_fingerprint"],
            "query_spec": artifact["query_spec"],
            "matched_count": artifact["matched_count"],
            "verification": artifact["verification"],
            "data_source": artifact["data_source"],
        },
        attribute_schema=(
            "candidate_id", "name", "category", "price", "stock", "summary",
            *tuple(sorted({key for row in artifact["products"] for key in (row.get("values") or {})})),
        ),
    )


_CATALOG_PROVIDER = DomainProviderRegistration(
    provider_id="CATALOG",
    description="从真实商品目录解析一个完整、已核验的有限候选域；parameters 只能包含 query",
    parameter_schema={
        "query": "CatalogQuerySpec；selector.mode 必须为 ALL，不能包含 metrics 或 partition",
    },
    validate_parameters=_catalog_parameters,
    resolve=_catalog_resolution,
    validate_grounding=_catalog_grounding,
    attribute_schema=("candidate_id", "name", "category", "price", "stock", "summary"),
    extract_requirements=_catalog_requirements,
)
_DEFAULT_DOMAIN_REGISTRY = DomainProviderRegistry((_CATALOG_PROVIDER,))
_ACTIVE_DOMAIN_REGISTRY: ContextVar[DomainProviderRegistry | None] = ContextVar(
    "constraint_domain_registry", default=None
)


def get_domain_provider_registry() -> DomainProviderRegistry:
    return _ACTIVE_DOMAIN_REGISTRY.get() or _DEFAULT_DOMAIN_REGISTRY


@contextmanager
def use_domain_provider_registry(registry: DomainProviderRegistry):
    token = _ACTIVE_DOMAIN_REGISTRY.set(registry)
    try:
        yield registry
    finally:
        _ACTIVE_DOMAIN_REGISTRY.reset(token)


def constraint_provider_prompt_catalog() -> list[dict]:
    return get_domain_provider_registry().prompt_catalog()


def _number_in_sources(value: int | float, sources: list[str]) -> bool:
    variants = {str(value), f"{value:g}"}
    return any(
        re.search(r"(?<![\d.])" + re.escape(candidate) + r"(?![\d.])", source)
        for source in sources for candidate in variants
    )


def validate_constraint_plan_grounding(spec: ConstraintPlanSpec, sources: list[str]) -> None:
    registry = get_domain_provider_registry()
    for goal in spec.goals:
        if not any(goal.source_quote in source for source in sources):
            raise AgentError("UNGROUNDED_CONSTRAINT_REQUIREMENT", f"目标 {goal.requirement_id} 必须引用用户原文", retryable=False)
        if not any(goal.domain.source_quote in source for source in sources):
            raise AgentError("UNGROUNDED_CONSTRAINT_REQUIREMENT", f"数据域 {goal.domain.requirement_id} 必须引用用户原文", retryable=False)
        if goal.quantity != 1 and not _number_in_sources(goal.quantity, sources):
            raise AgentError("UNGROUNDED_CONSTRAINT_VALUE", "目标数量必须来自用户原文", retryable=False)
        registry.registration(goal.domain.provider).validate_grounding(goal.domain.parameters, sources)
    for constraint in spec.constraints:
        if not any(constraint.source_quote in source for source in sources):
            raise AgentError("UNGROUNDED_CONSTRAINT_REQUIREMENT", f"约束 {constraint.requirement_id} 必须引用用户原文", retryable=False)
        if isinstance(constraint, AggregateConstraint) and isinstance(constraint.value, (int, float)):
            if not _number_in_sources(constraint.value, sources):
                raise AgentError("UNGROUNDED_CONSTRAINT_VALUE", "约束值必须来自用户原文", retryable=False)
    if spec.optimization and not any(spec.optimization.source_quote in source for source in sources):
        raise AgentError("UNGROUNDED_CONSTRAINT_REQUIREMENT", "优化目标必须引用用户原文", retryable=False)


def compilation_coverage(spec: ConstraintPlanSpec) -> list[dict]:
    rows: list[dict] = []
    for goal_index, goal in enumerate(spec.goals):
        rows.append(_coverage_row(f"goal:{goal.goal_id}:{goal.requirement_id}", goal.source_quote,
                                  f"spec.goals.{goal_index}"))
        rows.append(_coverage_row(f"domain:{goal.goal_id}:{goal.domain.requirement_id}", goal.domain.source_quote,
                                  f"spec.goals.{goal_index}.domain"))
        registry = get_domain_provider_registry()
        parameters = registry.validate(goal.domain)
        registration = registry.registration(goal.domain.provider)
        for requirement_id, source_quote, relative_path in registration.extract_requirements(parameters):
            rows.append(_coverage_row(
                f"domain:{goal.goal_id}:{requirement_id}",
                source_quote,
                f"spec.goals.{goal_index}.domain.parameters.{relative_path}",
            ))
    for index, constraint in enumerate(spec.constraints):
        rows.append(_coverage_row(f"constraint:{constraint.requirement_id}", constraint.source_quote,
                                  f"spec.constraints.{index}"))
    if spec.optimization:
        rows.append(_coverage_row(f"optimization:{spec.optimization.requirement_id}",
                                  spec.optimization.source_quote, "spec.optimization"))
    identities = [row["requirement_id"] for row in rows]
    if len(identities) != len(set(identities)):
        raise AgentError("INCOMPLETE_COMPILATION_COVERAGE", "约束规划要求身份不唯一", retryable=False)
    return rows


def _coverage_row(requirement_id: str, source_quote: str, path: str) -> dict:
    return {
        "requirement_id": requirement_id,
        "source_quote": source_quote,
        "operation": CONSTRAINT_PLAN_OPERATION,
        "argument_path": path,
        "status": "COMPILED",
    }


def compile_constraint_plan(spec_payload: object) -> dict:
    spec = ConstraintPlanSpec.model_validate(spec_payload, strict=True)
    for goal in spec.goals:
        get_domain_provider_registry().validate(goal.domain)
    coverage = compilation_coverage(spec)
    from src.runtime import current_runtime

    context = current_runtime()
    if context:
        context.record("compilation_coverage", operation=CONSTRAINT_PLAN_OPERATION, coverage=coverage)
    return {
        "spec": spec.model_dump(mode="json"),
        "compilation_coverage": coverage,
        "provider_registry_version": get_domain_provider_registry().version,
    }


def _decimal(value: object) -> Decimal | None:
    if type(value) not in {int, float} or not math.isfinite(value):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _compare(actual: object, operator: ComparisonOperator, expected: object) -> bool:
    actual_number, expected_number = _decimal(actual), _decimal(expected)
    if actual_number is not None and expected_number is not None:
        left, right = actual_number, expected_number
    elif isinstance(actual, str) and isinstance(expected, str):
        left, right = actual, expected
    else:
        return False
    return {
        "EQ": left == right,
        "NE": left != right,
        "LT": left < right,
        "LTE": left <= right,
        "GT": left > right,
        "GTE": left >= right,
    }[operator]


def _reduce(candidates: list[CandidateValue], attribute: str, reducer: Reducer) -> Scalar | None:
    values = [candidate.candidate_id if attribute == "candidate_id" else candidate.attributes.get(attribute)
              for candidate in candidates]
    if reducer == "COUNT":
        return len(values)
    if not values or any(value is None for value in values):
        return None
    if reducer == "FIRST":
        return values[0] if len(values) == 1 else None
    numbers = [_decimal(value) for value in values]
    if any(value is None for value in numbers):
        return None
    if reducer == "SUM":
        result = sum(numbers, Decimal(0))
    elif reducer == "MIN":
        result = min(numbers)
    elif reducer == "MAX":
        result = max(numbers)
    else:
        result = sum(numbers, Decimal(0)) / len(numbers)
    return int(result) if result == result.to_integral_value() else float(result)


def _scoped_candidates(assignments: Mapping[str, tuple[CandidateValue, ...]], goal_ids: list[str]) -> list[CandidateValue]:
    return [candidate for goal_id in goal_ids for candidate in assignments.get(goal_id, ())]


def _constraint_result(constraint: HardConstraint,
                       assignments: Mapping[str, tuple[CandidateValue, ...]]) -> tuple[bool, dict]:
    if isinstance(constraint, AggregateConstraint):
        observed = _reduce(_scoped_candidates(assignments, constraint.scope_goal_ids),
                           constraint.attribute, constraint.function)
        satisfied = observed is not None and _compare(observed, constraint.operator, constraint.value)
        return satisfied, {
            "requirement_id": constraint.requirement_id,
            "kind": constraint.kind,
            "observed": observed,
            "operator": constraint.operator,
            "expected": constraint.value,
            "satisfied": satisfied,
        }
    if isinstance(constraint, ComparisonConstraint):
        left = _reduce(list(assignments.get(constraint.left.goal_id, ())),
                       constraint.left.attribute, constraint.left.reducer)
        right = _reduce(list(assignments.get(constraint.right.goal_id, ())),
                        constraint.right.attribute, constraint.right.reducer)
        satisfied = left is not None and right is not None and _compare(left, constraint.operator, right)
        return satisfied, {
            "requirement_id": constraint.requirement_id,
            "kind": constraint.kind,
            "left": left,
            "operator": constraint.operator,
            "right": right,
            "satisfied": satisfied,
        }
    candidates = _scoped_candidates(assignments, constraint.scope_goal_ids)
    values = [candidate.candidate_id if constraint.attribute == "candidate_id"
              else candidate.attributes.get(constraint.attribute) for candidate in candidates]
    satisfied = all(value is not None for value in values) and len(values) == len(set(values))
    return satisfied, {
        "requirement_id": constraint.requirement_id,
        "kind": constraint.kind,
        "attribute": constraint.attribute,
        "observed": values,
        "satisfied": satisfied,
    }


def _candidate_resource_claims(candidate: CandidateValue, provider: str) -> tuple[ResourceClaim, ...]:
    if candidate.resource_claims:
        return candidate.resource_claims
    if candidate.max_quantity is not None:
        return (ResourceClaim(
            resource_id=f"{provider}:{candidate.candidate_id}",
            capacity=candidate.max_quantity,
        ),)
    return ()


def _resource_capacities(domains: Mapping[str, DomainResolution]) -> dict[str, int]:
    capacities: dict[str, int] = {}
    for resolution in domains.values():
        for candidate in resolution.candidates:
            for claim in _candidate_resource_claims(candidate, resolution.provider):
                previous = capacities.setdefault(claim.resource_id, claim.capacity)
                if previous != claim.capacity:
                    raise AgentError(
                        "INCONSISTENT_RESOURCE_CAPACITY",
                        f"资源 {claim.resource_id} 在不同候选域中声明了不一致的容量",
                        retryable=False,
                    )
    return capacities


def _resource_capacity_checks(
    assignments: Mapping[str, tuple[CandidateValue, ...]],
    domains: Mapping[str, DomainResolution],
    capacities: Mapping[str, int],
) -> tuple[bool, list[dict]]:
    usage: dict[str, dict] = {
        resource_id: {"used": 0, "consumers": []}
        for resource_id in capacities
    }
    for goal_id, candidates in assignments.items():
        resolution = domains[goal_id]
        for candidate in candidates:
            for claim in _candidate_resource_claims(candidate, resolution.provider):
                row = usage[claim.resource_id]
                row["used"] += claim.units
                row["consumers"].append({
                    "goal_id": goal_id,
                    "candidate_id": candidate.candidate_id,
                    "units": claim.units,
                })
    checks = [
        {
            "resource_id": resource_id,
            "capacity": capacity,
            "used": usage[resource_id]["used"],
            "remaining": capacity - usage[resource_id]["used"],
            "satisfied": usage[resource_id]["used"] <= capacity,
            "consumers": usage[resource_id]["consumers"],
        }
        for resource_id, capacity in sorted(capacities.items())
        if usage[resource_id]["used"]
    ]
    return all(check["satisfied"] for check in checks), checks


def _goal_options(
    goal: GoalNode,
    resolution: DomainResolution,
    capacities: Mapping[str, int],
) -> tuple[tuple[CandidateValue, ...], ...]:
    options = []
    for choice in combinations_with_replacement(resolution.candidates, goal.quantity):
        within_capacity, _checks = _resource_capacity_checks(
            {goal.goal_id: choice},
            {goal.goal_id: resolution},
            capacities,
        )
        if not within_capacity:
            continue
        options.append(choice)
        if len(options) > MAX_GOAL_OPTIONS:
            raise AgentError("GOAL_OPTION_BUDGET_EXHAUSTED", "单个目标的可选组合超过安全上限", retryable=False)
    return tuple(options)


def _solution_signature(assignments: Mapping[str, tuple[CandidateValue, ...]], goals: list[GoalNode]) -> tuple[str, ...]:
    return tuple(",".join(candidate.candidate_id for candidate in assignments[goal.goal_id]) for goal in goals)


def _optimization_value(spec: ConstraintPlanSpec,
                        assignments: Mapping[str, tuple[CandidateValue, ...]]) -> Scalar | None:
    if spec.optimization is None:
        return None
    return _reduce(_scoped_candidates(assignments, spec.optimization.scope_goal_ids),
                   spec.optimization.attribute, spec.optimization.function)


def _optimization_order_key(
    spec: ConstraintPlanSpec,
    assignments: Mapping[str, tuple[CandidateValue, ...]],
) -> tuple[Decimal, tuple[str, ...]]:
    """Return a stable best-first key for an explicit numeric objective."""

    value = _optimization_value(spec, assignments)
    numeric = _decimal(value)
    if numeric is None:
        raise AgentError(
            "UNRESOLVED_OPTIMIZATION_ATTRIBUTE",
            "优化目标引用的属性不能从候选数据域中完整解析",
            retryable=False,
        )
    rank = -numeric if spec.optimization and spec.optimization.direction == "MAX" else numeric
    return rank, _solution_signature(assignments, spec.goals)


def _serialize_solution(spec: ConstraintPlanSpec,
                        assignments: Mapping[str, tuple[CandidateValue, ...]],
                        domains: Mapping[str, DomainResolution],
                        capacities: Mapping[str, int]) -> dict:
    constraint_checks = [_constraint_result(constraint, assignments)[1] for constraint in spec.constraints]
    _resources_satisfied, resource_checks = _resource_capacity_checks(assignments, domains, capacities)
    selections = []
    for goal in spec.goals:
        candidates = assignments[goal.goal_id]
        selections.append({
            "goal_id": goal.goal_id,
            "source_quote": goal.source_quote,
            "quantity": goal.quantity,
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "label": candidate.label,
                    "attributes": dict(candidate.attributes),
                    "evidence": dict(candidate.evidence),
                    "resource_claims": [
                        {
                            "resource_id": claim.resource_id,
                            "capacity": claim.capacity,
                            "units": claim.units,
                        }
                        for claim in _candidate_resource_claims(candidate, domains[goal.goal_id].provider)
                    ],
                }
                for candidate in candidates
            ],
        })
    identity = {
        "selections": selections,
        "constraints": constraint_checks,
        "resource_capacity": resource_checks,
    }
    return {
        "solution_id": "cps_" + hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:20],
        "selections": selections,
        "constraint_checks": constraint_checks,
        "resource_checks": resource_checks,
        "optimization_value": _optimization_value(spec, assignments),
    }


def _choose_solutions(
    spec: ConstraintPlanSpec,
    feasible: list[Mapping[str, tuple[CandidateValue, ...]]],
    domains: Mapping[str, DomainResolution],
    capacities: Mapping[str, int],
) -> list[dict]:
    if not feasible:
        return []
    if spec.optimization:
        ordered = sorted(feasible, key=lambda row: _optimization_order_key(spec, row))
        return [
            _serialize_solution(spec, row, domains, capacities)
            for row in ordered[:MAX_DELIVERED_SOLUTIONS]
        ]
    remaining = sorted(feasible, key=lambda row: _solution_signature(row, spec.goals))
    selected = [remaining.pop(0)]
    while remaining and len(selected) < MAX_DELIVERED_SOLUTIONS:
        selected_signatures = [_solution_signature(row, spec.goals) for row in selected]

        def distance(row):
            signature = _solution_signature(row, spec.goals)
            nearest = min(sum(left != right for left, right in zip(signature, existing))
                          for existing in selected_signatures)
            return nearest, signature

        chosen = max(remaining, key=distance)
        selected.append(chosen)
        remaining.remove(chosen)
    return [_serialize_solution(spec, row, domains, capacities) for row in selected]


def solve_finite_domain(
    spec: ConstraintPlanSpec,
    domains: Mapping[str, DomainResolution],
    *,
    max_search_nodes: int = MAX_SEARCH_NODES,
) -> dict:
    """Solve a generic finite-domain CSP without knowing what candidates represent."""

    if max_search_nodes < 1:
        raise AgentError("INVALID_SEARCH_BUDGET", "约束搜索预算必须大于零", retryable=False)
    goal_domains: dict[str, DomainResolution] = {}
    for goal in spec.goals:
        resolution = domains.get(goal.goal_id)
        if resolution is None:
            raise AgentError("MISSING_GOAL_DOMAIN", f"目标 {goal.goal_id} 缺少候选数据域", retryable=False)
        goal_domains[goal.goal_id] = resolution
    resource_capacities = _resource_capacities(goal_domains)
    options: dict[str, tuple[tuple[CandidateValue, ...], ...]] = {}
    for goal in spec.goals:
        resolution = goal_domains[goal.goal_id]
        options[goal.goal_id] = _goal_options(goal, resolution, resource_capacities)
    _validate_constraint_attributes(spec, goal_domains)
    feasible: list[Mapping[str, tuple[CandidateValue, ...]]] = []
    feasible_count = 0
    nodes_examined = 0
    exhausted = False

    def visit(index: int, assignments: dict[str, tuple[CandidateValue, ...]]) -> None:
        nonlocal feasible_count, nodes_examined, exhausted
        if exhausted:
            return
        if index == len(spec.goals):
            if nodes_examined >= max_search_nodes:
                exhausted = True
                return
            nodes_examined += 1
            resources_satisfied, _resource_checks = _resource_capacity_checks(
                assignments, goal_domains, resource_capacities
            )
            if resources_satisfied and all(
                _constraint_result(constraint, assignments)[0] for constraint in spec.constraints
            ):
                feasible_count += 1
                feasible.append(dict(assignments))
                if spec.optimization:
                    # Retain the true best K seen so far instead of the first K.
                    # A completed traversal can therefore prove optimality without
                    # retaining every feasible assignment in memory.
                    feasible.sort(key=lambda row: _optimization_order_key(spec, row))
                    del feasible[MAX_DELIVERED_SOLUTIONS:]
                elif len(feasible) > MAX_STORED_FEASIBLE:
                    feasible.pop()
            return
        goal = spec.goals[index]
        for choice in options[goal.goal_id]:
            assignments[goal.goal_id] = choice
            visit(index + 1, assignments)
            assignments.pop(goal.goal_id, None)
            if exhausted:
                return

    visit(0, {})
    if exhausted and not feasible:
        raise AgentError(
            "CONSTRAINT_SEARCH_INCONCLUSIVE",
            "约束搜索在找到可验证方案前达到安全上限，不能据此宣称无解",
            retryable=False,
        )
    solutions = _choose_solutions(spec, feasible, goal_domains, resource_capacities)
    if spec.optimization is None:
        optimality = {"status": "NOT_REQUESTED", "objective_value": None, "proof": None}
    elif not solutions:
        optimality = {
            "status": "NO_FEASIBLE_SOLUTION",
            "objective_value": None,
            "proof": "EXHAUSTIVE_SEARCH",
        }
    else:
        optimality = {
            "status": "PROVEN_OPTIMAL" if not exhausted else "BEST_KNOWN",
            "objective_value": solutions[0]["optimization_value"],
            "proof": "EXHAUSTIVE_SEARCH" if not exhausted else "BOUNDED_SEARCH",
        }
    return {
        "feasibility": "FEASIBLE" if solutions else "INFEASIBLE",
        "solutions": solutions,
        "search": {
            "complete": not exhausted,
            "nodes_examined": nodes_examined,
            "feasible_count": feasible_count,
            "feasible_count_exact": not exhausted,
            "retained_feasible_count": len(feasible),
            "stored_solution_limit_reached": (
                spec.optimization is None and feasible_count > MAX_STORED_FEASIBLE
            ),
            "max_search_nodes": max_search_nodes,
        },
        "optimality": optimality,
        "selection_policy": (
            "EXPLICIT_OPTIMIZATION" if spec.optimization else "STABLE_FARTHEST_FIRST"
        ),
    }


def _validate_constraint_attributes(spec: ConstraintPlanSpec,
                                    domains: Mapping[str, DomainResolution]) -> None:
    def supports(goal_id: str, attribute: str) -> bool:
        resolution = domains[goal_id]
        available = set(resolution.attribute_schema)
        if not available:
            available = {
                key for candidate in resolution.candidates
                for key in ("candidate_id", *candidate.attributes.keys())
            }
        return attribute in available

    checks: list[tuple[list[str], str]] = []
    for constraint in spec.constraints:
        if isinstance(constraint, AggregateConstraint | AllDifferentConstraint):
            checks.append((constraint.scope_goal_ids, constraint.attribute))
        else:
            checks.extend([
                ([constraint.left.goal_id], constraint.left.attribute),
                ([constraint.right.goal_id], constraint.right.attribute),
            ])
    if spec.optimization:
        checks.append((spec.optimization.scope_goal_ids, spec.optimization.attribute))
    invalid = [f"{goal_id}.{attribute}" for goal_ids, attribute in checks for goal_id in goal_ids
               if not supports(goal_id, attribute)]
    if invalid:
        raise AgentError(
            "UNKNOWN_CONSTRAINT_ATTRIBUTE",
            "约束引用了数据域未声明的属性：" + "、".join(sorted(set(invalid))),
            retryable=False,
        )


def _execute_spec(spec: ConstraintPlanSpec) -> tuple[dict, dict[str, DomainResolution]]:
    registry = get_domain_provider_registry()
    domains = {goal.goal_id: registry.resolve(goal.domain) for goal in spec.goals}
    return solve_finite_domain(spec, domains), domains


def solve_constraint_plan(spec: dict, compilation_coverage: list[dict],
                          provider_registry_version: str) -> dict:
    parsed = ConstraintPlanSpec.model_validate(spec, strict=True)
    expected_coverage = globals()["compilation_coverage"](parsed)
    registry = get_domain_provider_registry()
    if compilation_coverage != expected_coverage:
        raise AgentError("INCOMPLETE_COMPILATION_COVERAGE", "约束规划目标没有被完整编译", retryable=False)
    if provider_registry_version != registry.version:
        raise AgentError("DOMAIN_PROVIDER_REGISTRY_CHANGED", "数据域注册表在编译后发生变化", retryable=False)
    solved, domains = _execute_spec(parsed)
    domain_evidence = [
        {
            "goal_id": goal.goal_id,
            "provider": domains[goal.goal_id].provider,
            "provider_fingerprint": domains[goal.goal_id].provider_fingerprint,
            "candidate_count": len(domains[goal.goal_id].candidates),
            "evidence": dict(domains[goal.goal_id].evidence),
        }
        for goal in parsed.goals
    ]
    identity = {
        "version": CONSTRAINT_PLAN_VERSION,
        "spec": parsed.model_dump(mode="json"),
        "coverage": expected_coverage,
        "provider_registry_version": registry.version,
        "domains": domain_evidence,
        "solved": solved,
    }
    artifact = {
        "artifact_type": "constraint_plan_result",
        "version": CONSTRAINT_PLAN_VERSION,
        "solution_set_id": "cpr_" + hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:24],
        "plan_spec": parsed.model_dump(mode="json"),
        "compilation_coverage": expected_coverage,
        "provider_registry_version": registry.version,
        "domains": domain_evidence,
        **solved,
        "generated_at": datetime.now(UTC).isoformat(),
        "verification": {"status": "PENDING"},
    }
    return {"artifact": artifact, "feasibility": solved["feasibility"], "solutions": solved["solutions"]}


def _artifact_core(artifact: Mapping[str, object]) -> dict:
    return {key: value for key, value in artifact.items() if key not in {"generated_at", "verification"}}


def verify_constraint_plan_result(result: object, arguments: Mapping[str, object]) -> dict:
    artifact = result.get("artifact") if isinstance(result, dict) else None
    if not isinstance(artifact, dict):
        raise AgentError("INVALID_TOOL_RESULT", "约束求解没有返回结构化结果产物", retryable=False)
    artifact["verification"] = {"status": "PENDING"}
    try:
        required = {
            "artifact_type", "version", "solution_set_id", "plan_spec", "compilation_coverage",
            "provider_registry_version", "domains", "feasibility", "solutions", "search",
            "optimality", "selection_policy", "generated_at",
        }
        if not required <= artifact.keys() or artifact.get("artifact_type") != "constraint_plan_result":
            raise AgentError("INVALID_TOOL_RESULT", "约束结果缺少必要证据", retryable=False)
        parsed = ConstraintPlanSpec.model_validate(arguments["spec"], strict=True)
        compiled = compilation_coverage(parsed)
        if arguments.get("compilation_coverage") != compiled or artifact.get("compilation_coverage") != compiled:
            raise AgentError("INCOMPLETE_COMPILATION_COVERAGE", "约束结果的编译覆盖证明不一致", retryable=False)
        if arguments.get("provider_registry_version") != get_domain_provider_registry().version:
            raise AgentError("DOMAIN_PROVIDER_REGISTRY_CHANGED", "核验时数据域注册表身份不一致", retryable=False)
        expected = solve_constraint_plan(
            parsed.model_dump(mode="json"), compiled, get_domain_provider_registry().version
        )["artifact"]
        if _artifact_core(artifact) != _artifact_core(expected):
            raise AgentError("CONSTRAINT_RESULT_EVIDENCE_MISMATCH", "约束结果与权威数据重算不一致", retryable=False)
        if artifact["feasibility"] == "INFEASIBLE" and not artifact["search"].get("complete"):
            raise AgentError("CONSTRAINT_SEARCH_INCONCLUSIVE", "未完成搜索不能宣称约束无解", retryable=False)
        checked = [row["requirement_id"] for row in compiled]
        best_known_only = artifact["optimality"].get("status") == "BEST_KNOWN"
        artifact["verification"] = {
            "status": "VERIFIED",
            "verdict": (
                "BEST_KNOWN" if best_known_only
                else "SATISFIED" if artifact["feasibility"] == "FEASIBLE"
                else "PROVEN_INFEASIBLE"
            ),
            "checked_requirements": checked,
            "global_goal": not best_known_only,
        }
        return artifact["verification"]
    except AgentError as exc:
        artifact["verification"] = {
            "status": "REJECTED",
            "error": {"code": exc.code, "message": str(exc), "retryable": exc.retryable},
        }
        raise


def constraint_plan_outcome(result: object, _arguments: Mapping[str, object]) -> str:
    artifact = result.get("artifact") if isinstance(result, dict) else None
    if not isinstance(artifact, dict) or artifact.get("feasibility") != "FEASIBLE":
        return "UNAVAILABLE_EVIDENCE"
    return (
        "UNAVAILABLE_EVIDENCE"
        if (artifact.get("optimality") or {}).get("status") == "BEST_KNOWN"
        else "SATISFIED"
    )


def render_constraint_plan_result(result: object, _arguments: Mapping[str, object]) -> str:
    artifact = result["artifact"]
    if artifact["feasibility"] == "INFEASIBLE":
        return "已完整检查当前真实候选域，没有找到同时满足全部硬约束的方案；系统没有放宽条件或用单项结果替代。"
    lines = [f"已找到 {len(artifact['solutions'])} 组通过全部约束核验的可行方案："]
    for index, solution in enumerate(artifact["solutions"], start=1):
        parts = []
        for selection in solution["selections"]:
            labels = []
            for candidate in selection["candidates"]:
                price = candidate["attributes"].get("price")
                labels.append(f"{candidate['label']}（¥{price:.2f}）" if isinstance(price, (int, float)) else candidate["label"])
            parts.append(f"{selection['source_quote']}：{' × '.join(labels)}")
        checks = [check for check in solution["constraint_checks"] if check.get("kind") == "AGGREGATE"]
        aggregate = f"；约束值 {checks[0]['observed']}" if checks else ""
        lines.append(f"- 方案 {index}：{'；'.join(parts)}{aggregate}")
    if artifact["selection_policy"] == "STABLE_FARTHEST_FIRST":
        lines.append("这些方案按确定性的差异覆盖策略选取，不代表系统已经知道你的个人偏好。")
    elif artifact["optimality"]["status"] == "BEST_KNOWN":
        lines.append("搜索已达到安全上限，以上是当前已知最优方案，尚未证明全局最优。")
    elif artifact["optimality"]["status"] == "PROVEN_OPTIMAL":
        lines.append("搜索已完整结束，首个方案的显式优化目标已证明达到全局最优。")
    return "\n".join(lines)
