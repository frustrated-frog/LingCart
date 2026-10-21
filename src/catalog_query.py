"""受限、可验证的商品目录查询协议。

模型只能声明语义 AST；字段绑定、结果集引用、执行和核验都由 Stable Kernel
完成。这个模块刻意不接受 SQL、Python 表达式或任意 JSONPath。
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import UTC, datetime
from enum import Enum
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from src.catalog_concepts import CatalogConceptRef, CatalogConceptError, get_catalog_concept_registry
from src.errors import AgentError


CATALOG_QUERY_VERSION = "catalog-query/v1"
CATALOG_QUERY_OPERATION = "query_catalog"

CoreField = Literal["product_id", "name", "category", "price", "stock", "summary"]
FilterOperator = Literal["EQ", "NE", "CONTAINS", "NOT_CONTAINS", "LT", "LTE", "GT", "GTE", "IN"]
SortDirection = Literal["ASC", "DESC"]
SelectorMode = Literal["ALL", "TOP_K", "PAGE", "ARGMIN", "ARGMAX"]
MetricFunction = Literal["COUNT", "MIN", "MAX", "AVG", "SUM"]
ScopeKind = Literal["CATALOG", "PREVIOUS_RESULT_SET"]
ResultSetView = Literal["DELIVERED", "POPULATION"]
AvailabilityScope = Literal["ALL", "AVAILABLE"]
Scalar = str | int | float


class PredicateTruth(str, Enum):
    """目录谓词的三值结果；缺失证据不能证明否定条件成立。"""

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Requirement(StrictModel):
    requirement_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    source_quote: str = Field(min_length=1, max_length=300)


class CatalogFieldRef(StrictModel):
    """对查询可见的逻辑字段；物理 specs 路径不会由模型直接提供。"""

    core: CoreField | None = None
    concept_id: str | None = Field(None, pattern=r"^[a-z][a-z0-9_.-]{1,79}$")

    @model_validator(mode="after")
    def exactly_one_field(self):
        if (self.core is None) == (self.concept_id is None):
            raise ValueError("Catalog field must declare exactly one of core or concept_id")
        if self.concept_id is not None:
            get_catalog_concept_registry().definition(self.concept_id)
        return self

    @property
    def identity(self) -> str:
        return self.core or str(self.concept_id)


class CatalogScope(Requirement):
    kind: ScopeKind
    result_set_view: ResultSetView = "DELIVERED"

    @model_validator(mode="after")
    def valid_result_set_view(self):
        if self.kind == "CATALOG" and self.result_set_view != "DELIVERED":
            raise ValueError("CATALOG scope cannot select a previous result-set view")
        return self


class CatalogAvailability(Requirement):
    scope: AvailabilityScope


class CatalogFilter(Requirement):
    field: CatalogFieldRef | None = None
    concept: CatalogConceptRef | None = None
    operator: FilterOperator | None = None
    value: Scalar | list[Scalar] | None = None

    @model_validator(mode="after")
    def valid_filter(self):
        if (self.field is None) == (self.concept is None):
            raise ValueError("Filter must declare exactly one of field or concept")
        if self.concept is not None:
            if self.operator is not None or self.value is not None:
                raise ValueError("Concept filter owns its operator and canonical value")
            return self
        if self.operator is None or self.value is None:
            raise ValueError("Core field filter requires operator and value")
        if self.operator == "IN" and not isinstance(self.value, list):
            raise ValueError("IN requires an array value")
        if self.operator != "IN" and isinstance(self.value, list):
            raise ValueError("Only IN accepts an array value")
        return self


class CatalogOrdering(Requirement):
    field: CatalogFieldRef
    direction: SortDirection


class CatalogPartition(Requirement):
    field: CatalogFieldRef


class CatalogSelector(Requirement):
    mode: SelectorMode
    field: CatalogFieldRef | None = None
    limit: int | None = Field(None, ge=1, le=500)
    page: int | None = Field(None, ge=1, le=10000)
    page_size: int | None = Field(None, ge=1, le=200)
    include_ties: bool = False

    @model_validator(mode="after")
    def selector_shape(self):
        needs_field = self.mode in {"ARGMIN", "ARGMAX"}
        if needs_field != (self.field is not None):
            raise ValueError("ARGMIN/ARGMAX require a field; other selectors do not accept one")
        if self.mode == "TOP_K" and self.limit is None:
            raise ValueError("TOP_K requires limit")
        if self.mode != "TOP_K" and self.limit is not None:
            raise ValueError("Only TOP_K accepts limit")
        if self.mode == "PAGE" and (self.page is None or self.page_size is None):
            raise ValueError("PAGE requires page and page_size")
        if self.mode != "PAGE" and (self.page is not None or self.page_size is not None):
            raise ValueError("Only PAGE accepts page fields")
        if self.mode not in {"ARGMIN", "ARGMAX"} and self.include_ties:
            raise ValueError("Only ARGMIN/ARGMAX may include ties")
        return self


class CatalogMetric(Requirement):
    function: MetricFunction
    field: CatalogFieldRef | None = None
    alias: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")

    @model_validator(mode="after")
    def metric_shape(self):
        if self.function == "COUNT" and self.field is not None:
            raise ValueError("COUNT does not accept a field")
        if self.function != "COUNT" and self.field is None:
            raise ValueError(f"{self.function} requires a field")
        return self


class CatalogProjection(Requirement):
    fields: list[CatalogFieldRef] = Field(min_length=1, max_length=12)


class CatalogQuerySpec(StrictModel):
    version: Literal["catalog-query/v1"]
    scope: CatalogScope
    availability: CatalogAvailability
    filters: list[CatalogFilter] = Field(default_factory=list, max_length=16)
    partition: CatalogPartition | None = None
    ordering: list[CatalogOrdering] = Field(default_factory=list, max_length=6)
    selector: CatalogSelector
    metrics: list[CatalogMetric] = Field(default_factory=list, max_length=8)
    projection: CatalogProjection

    @model_validator(mode="after")
    def stable_requirements(self):
        rows = self.requirements()
        identities = [row.requirement_id for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError("Catalog requirement IDs must be unique")
        aliases = [metric.alias for metric in self.metrics]
        if len(aliases) != len(set(aliases)):
            raise ValueError("Catalog metric aliases must be unique")
        field_ids = [field.identity for field in self.projection.fields]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("Projection fields must be unique")
        if self.selector.mode in {"ARGMIN", "ARGMAX", "TOP_K", "PAGE"} and self.partition is not None:
            raise ValueError("Partitioned limited selection is ambiguous; use ALL or a grouped metric")
        if self.selector.mode in {"TOP_K", "PAGE"} and not self.ordering:
            raise ValueError("TOP_K and PAGE require deterministic ordering")
        if self.metrics and self.selector.mode != "ALL":
            raise ValueError("Metrics require ALL so their population is explicit")
        return self

    def requirements(self) -> list[Requirement]:
        return [
            self.scope,
            self.availability,
            *self.filters,
            *([self.partition] if self.partition else []),
            *self.ordering,
            self.selector,
            *self.metrics,
            self.projection,
        ]


def constrain_catalog_query_schema(schema: dict) -> dict:
    """Apply the existing concept registry constraints to a generated schema."""

    return get_catalog_concept_registry().constrain_schema(schema)


def validate_catalog_query_grounding(spec: CatalogQuerySpec, sources: list[str]) -> None:
    folded_sources = [source.casefold() for source in sources]
    for requirement in spec.requirements():
        if not any(requirement.source_quote in source for source in sources):
            raise AgentError(
                "UNGROUNDED_CATALOG_REQUIREMENT",
                f"目录要求 {requirement.requirement_id} 必须引用用户原文",
                retryable=False,
            )
    for predicate in spec.filters:
        if predicate.concept is not None:
            continue
        values = predicate.value if isinstance(predicate.value, list) else [predicate.value]
        for value in values:
            if value is None:
                continue
            if isinstance(value, str):
                grounded = any(value.casefold() in source for source in folded_sources)
            else:
                variants = {str(value), f"{value:g}"}
                grounded = any(
                    re.search(r"(?<![\d.])" + re.escape(candidate) + r"(?![\d.])", source)
                    for source in folded_sources for candidate in variants
                )
            if not grounded:
                raise AgentError(
                    "UNGROUNDED_CATALOG_VALUE",
                    f"目录筛选值 {value!r} 必须来自用户或对话历史",
                    retryable=False,
                )


def compilation_coverage(spec: CatalogQuerySpec) -> list[dict]:
    """Produce a total, auditable mapping from semantic requirements to AST paths."""

    paths: list[tuple[Requirement, str]] = [(spec.scope, "spec.scope"), (spec.availability, "spec.availability")]
    paths.extend((item, f"spec.filters.{index}") for index, item in enumerate(spec.filters))
    if spec.partition:
        paths.append((spec.partition, "spec.partition"))
    paths.extend((item, f"spec.ordering.{index}") for index, item in enumerate(spec.ordering))
    paths.append((spec.selector, "spec.selector"))
    paths.extend((item, f"spec.metrics.{index}") for index, item in enumerate(spec.metrics))
    paths.append((spec.projection, "spec.projection"))
    coverage = [
        {
            "requirement_id": item.requirement_id,
            "source_quote": item.source_quote,
            "operation": CATALOG_QUERY_OPERATION,
            "argument_path": path,
            "status": "COMPILED",
        }
        for item, path in paths
    ]
    expected = {item.requirement_id for item in spec.requirements()}
    actual = {item["requirement_id"] for item in coverage}
    if expected != actual or len(coverage) != len(expected):
        raise AgentError("INCOMPLETE_COMPILATION_COVERAGE", "目录目标没有被无损编译", retryable=False)
    return coverage


build_compilation_coverage = compilation_coverage


def previous_result_set() -> dict | None:
    from src.runtime import current_runtime

    runtime = current_runtime()
    if runtime is None:
        return None
    conversation = runtime.snapshot.get("conversation") or {}
    result_set = conversation.get("catalog_result_set")
    return dict(result_set) if isinstance(result_set, dict) else None


def compile_catalog_query(spec_payload: object) -> dict:
    spec = CatalogQuerySpec.model_validate(spec_payload, strict=True)
    base = previous_result_set() if spec.scope.kind == "PREVIOUS_RESULT_SET" else None
    if spec.scope.kind == "PREVIOUS_RESULT_SET" and not base:
        raise AgentError(
            "MISSING_PREVIOUS_RESULT_SET",
            "当前会话没有可引用的上一组商品结果，请先查询或明确商品范围。",
            retryable=False,
        )
    return {
        "spec": spec.model_dump(mode="json"),
        "base_result_set": base,
        "compilation_coverage": compilation_coverage(spec),
    }


def _path_value(product: dict, path: str) -> object:
    value: object = product
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _field_value(product: dict, field: CatalogFieldRef) -> object:
    if field.core is not None:
        return product.get(field.core)
    registry = get_catalog_concept_registry()
    definition = registry.definition(str(field.concept_id))
    raw = _path_value(product, definition.field)
    if raw is None:
        return None
    try:
        return registry.decode_observation(str(field.concept_id), raw)
    except CatalogConceptError:
        return None


def _compare(actual: object, operator: str, expected: object) -> PredicateTruth:
    if actual is None:
        return PredicateTruth.UNKNOWN
    if operator in {"CONTAINS", "NOT_CONTAINS"}:
        found = str(expected).casefold() in str(actual).casefold()
        matched = found if operator == "CONTAINS" else not found
        return PredicateTruth.TRUE if matched else PredicateTruth.FALSE
    if operator == "IN":
        return PredicateTruth.TRUE if actual in expected else PredicateTruth.FALSE
    try:
        if operator == "EQ":
            matched = actual == expected
        if operator == "NE":
            matched = actual != expected
        if operator == "LT":
            matched = actual < expected
        if operator == "LTE":
            matched = actual <= expected
        if operator == "GT":
            matched = actual > expected
        if operator == "GTE":
            matched = actual >= expected
        if operator not in {"EQ", "NE", "LT", "LTE", "GT", "GTE"}:
            return PredicateTruth.UNKNOWN
    except TypeError:
        return PredicateTruth.UNKNOWN
    return PredicateTruth.TRUE if matched else PredicateTruth.FALSE


def _resolved_filter(predicate: CatalogFilter, products: list[dict]) -> tuple[CatalogFieldRef, str, object]:
    if predicate.concept is None:
        return predicate.field, str(predicate.operator), predicate.value
    registry = get_catalog_concept_registry()
    definition = registry.validate_reference(predicate.concept)
    observations = tuple(
        value for product in products
        if (value := _path_value(product, definition.field)) is not None
        and not isinstance(value, (dict, list, bool))
    )
    try:
        resolved = registry.resolve(predicate.concept, observations)
    except CatalogConceptError as exc:
        raise AgentError("CATALOG_CONCEPT_UNRESOLVED", str(exc), retryable=False) from exc
    operator_map = {
        "eq": "EQ", "ne": "NE", "contains": "CONTAINS", "not_contains": "NOT_CONTAINS",
        "lte": "LTE", "gte": "GTE",
    }
    return CatalogFieldRef(concept_id=predicate.concept.concept_id), operator_map[resolved.op], resolved.value


def _expected_filter_payload(predicate: CatalogFilter) -> dict:
    if predicate.concept is None:
        field, operator, value = predicate.field, str(predicate.operator), predicate.value
    else:
        encoded = get_catalog_concept_registry().encode(predicate.concept)
        operator_map = {
            "eq": "EQ", "ne": "NE", "contains": "CONTAINS", "not_contains": "NOT_CONTAINS",
            "lte": "LTE", "gte": "GTE",
        }
        field = CatalogFieldRef(concept_id=predicate.concept.concept_id)
        operator, value = operator_map[encoded.op], encoded.value
    return {"requirement_id": predicate.requirement_id, "field": field.model_dump(mode="json"),
            "operator": operator, "value": value}


def _sort_key(value: object) -> tuple[int, object]:
    if value is None:
        return (1, "")
    if isinstance(value, str):
        return (0, value.casefold())
    return (0, value)


def _ordered(products: list[dict], ordering: list[CatalogOrdering]) -> list[dict]:
    rows = list(products)
    # Stable sorting from the least-significant key preserves declared precedence.
    for order in reversed(ordering):
        present = [product for product in rows if _field_value(product, order.field) is not None]
        missing = [product for product in rows if _field_value(product, order.field) is None]
        present.sort(key=lambda product: _sort_key(_field_value(product, order.field)),
                     reverse=order.direction == "DESC")
        rows = present + missing
    return rows


def _partitioned(products: list[dict], partition: CatalogPartition | None) -> list[dict]:
    if partition is None:
        return products
    buckets: dict[str, list[dict]] = {}
    for product in products:
        value = _field_value(product, partition.field)
        key = str(value if value is not None else "未分类")
        buckets.setdefault(key, []).append(product)
    return [product for key in sorted(buckets, key=str.casefold) for product in buckets[key]]


def _select(products: list[dict], selector: CatalogSelector) -> list[dict]:
    if selector.mode == "ALL":
        return products
    if selector.mode == "TOP_K":
        return products[: selector.limit]
    if selector.mode == "PAGE":
        start = (selector.page - 1) * selector.page_size
        return products[start:start + selector.page_size]
    values = [(product, _field_value(product, selector.field)) for product in products]
    values = [(product, value) for product, value in values if value is not None]
    if not values:
        return []
    target = (min if selector.mode == "ARGMIN" else max)(value for _, value in values)
    matches = [product for product, value in values if value == target]
    return matches if selector.include_ties else matches[:1]


def _metrics(products: list[dict], metrics: list[CatalogMetric]) -> dict:
    values: dict[str, object] = {}
    for metric in metrics:
        if metric.function == "COUNT":
            values[metric.alias] = len(products)
            continue
        observed = [_field_value(product, metric.field) for product in products]
        observed = [value for value in observed if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if not observed:
            values[metric.alias] = None
        elif metric.function == "MIN":
            values[metric.alias] = min(observed)
        elif metric.function == "MAX":
            values[metric.alias] = max(observed)
        elif metric.function == "SUM":
            values[metric.alias] = sum(observed)
        elif metric.function == "AVG":
            values[metric.alias] = sum(observed) / len(observed)
    return values


def _groups(products: list[dict], partition: CatalogPartition | None) -> list[dict]:
    if partition is None:
        return []
    grouped: OrderedDict[str, list[str]] = OrderedDict()
    for product in products:
        value = _field_value(product, partition.field)
        label = str(value if value is not None else "未分类")
        grouped.setdefault(label, []).append(product["product_id"])
    return [{"key": key, "product_ids": ids, "count": len(ids)} for key, ids in grouped.items()]


def _projection_rows(products: list[dict], projection: CatalogProjection) -> list[dict]:
    rows = []
    for product in products:
        values = {field.identity: _field_value(product, field) for field in projection.fields}
        # Product identity and display label are always carried as evidence, even
        # when the user asked for a narrow projection.
        rows.append({
            "product_id": product["product_id"],
            "name": product["name"],
            "category": product["category"],
            "price": product["price"],
            "currency": product.get("currency", "CNY"),
            "summary": product["summary"],
            "stock": product["stock"],
            "specs": product.get("specs") or {},
            "data_source": product.get("data_source", "DEMO_CATALOG"),
            "values": values,
        })
    return rows


def _delivery_evidence(spec: CatalogQuerySpec, matched_count: int, delivered_count: int) -> dict:
    population_complete = delivered_count == matched_count
    has_more = False
    next_page = None
    if spec.selector.mode == "PAGE":
        end = (spec.selector.page - 1) * spec.selector.page_size + delivered_count
        has_more = end < matched_count
        next_page = spec.selector.page + 1 if has_more else None
    elif spec.selector.mode == "TOP_K":
        has_more = delivered_count < matched_count
    return {
        "mode": spec.selector.mode,
        "requested_complete": spec.selector.mode == "ALL",
        # complete means the declared selector was fully evaluated, while
        # population_complete says whether every matching row was delivered.
        "complete": True,
        "selection_complete": True,
        "population_complete": population_complete,
        "has_more": has_more,
        "page": spec.selector.page if spec.selector.mode == "PAGE" else None,
        "page_size": spec.selector.page_size if spec.selector.mode == "PAGE" else None,
        "next_page": next_page,
        "matched_count": matched_count,
        "delivered_count": delivered_count,
    }


def _catalog_record(product: dict) -> dict:
    """Normalize an authoritative row for hashing and frozen result-set reuse."""

    return {
        "product_id": product["product_id"],
        "name": product["name"],
        "category": product["category"],
        "price": product["price"],
        "currency": product.get("currency", "CNY"),
        "summary": product["summary"],
        "stock": product["stock"],
        "specs": product.get("specs") or {},
        "data_source": product.get("data_source", "DEMO_CATALOG"),
    }


def _catalog_data_fingerprint(products: list[dict]) -> str:
    payload = [_catalog_record(product) for product in sorted(products, key=lambda row: row["product_id"])]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_persisted_result_set(base_result_set: dict, view: ResultSetView = "DELIVERED") -> tuple[dict, list[dict]]:
    """Resolve a compact conversation pointer to the immutable prior artifact."""

    result_set_id = base_result_set.get("result_set_id")
    source_run_id = base_result_set.get("source_run_id")
    artifact = None
    if source_run_id:
        from src.persistence import RunStore

        stored = RunStore().run(str(source_run_id))
        result = (stored or {}).get("result") or {}
        artifact = next(
            (
                candidate for candidate in result.get("artifacts", [])
                if isinstance(candidate, dict) and candidate.get("result_set_id") == result_set_id
            ),
            None,
        )
    # Explicit inline artifacts remain useful for isolated executor tests, but
    # production conversation state stores only a run/result-set pointer.
    if artifact is None and isinstance(base_result_set.get("products"), list):
        artifact = base_result_set
    if not isinstance(artifact, dict):
        raise AgentError(
            "CATALOG_RESULT_SET_UNAVAILABLE",
            "上一组商品结果已经不可用，无法安全继承其范围",
            retryable=False,
        )
    delivered_rows = artifact.get("products")
    delivered_ids = artifact.get("member_ids")
    population_rows = artifact.get("population_products")
    population_ids = artifact.get("population_ids")
    # Results written before population snapshots existed remain usable for
    # delivered-set references.  They can represent a population only when the
    # delivered members were already the complete population.
    if population_rows is None and population_ids == delivered_ids:
        population_rows = delivered_rows
    rows = population_rows if view == "POPULATION" else delivered_rows
    member_ids = population_ids if view == "POPULATION" else delivered_ids
    if (
        not isinstance(rows, list)
        or not isinstance(member_ids, list)
        or member_ids != [row.get("product_id") for row in rows]
    ):
        raise AgentError(
            "CATALOG_RESULT_SET_CORRUPT",
            f"上一组商品结果的 {view.lower()} 成员证据不一致",
            retryable=False,
        )
    compact_ids = base_result_set.get("population_ids" if view == "POPULATION" else "member_ids")
    if compact_ids is not None and compact_ids != member_ids:
        raise AgentError(
            "CATALOG_RESULT_SET_CORRUPT",
            "上一组商品结果与会话引用的成员不一致",
            retryable=False,
        )
    products = [_catalog_record(row) for row in rows]
    fingerprint = _catalog_data_fingerprint(products)
    fingerprint_key = "population_data_fingerprint" if view == "POPULATION" else "result_data_fingerprint"
    expected = base_result_set.get(fingerprint_key) or artifact.get(fingerprint_key)
    if expected and fingerprint != expected:
        raise AgentError(
            "CATALOG_RESULT_SET_CORRUPT",
            f"上一组商品结果的 {view.lower()} 数据指纹不一致",
            retryable=False,
        )
    return artifact, products


def query_catalog(spec: dict, base_result_set: dict | None = None,
                  compilation_coverage: list[dict] | None = None) -> dict:
    """Execute one constrained catalog AST against the real local repository."""

    from src import tools

    parsed = CatalogQuerySpec.model_validate(spec, strict=True)
    expected_coverage = build_compilation_coverage(parsed)
    if compilation_coverage != expected_coverage:
        return {"error": "目录查询的编译覆盖证明无效"}
    snapshot = tools.get_catalog_metadata_provider().get_snapshot()
    with tools._session() as session:
        records = session.scalars(select(tools.Product).order_by(tools.Product.product_id)).all()
        products = [tools._product_dict(product) for product in records]
    if parsed.scope.kind == "PREVIOUS_RESULT_SET":
        if not base_result_set or not isinstance(base_result_set.get("member_ids"), list):
            raise AgentError(
                "MISSING_PREVIOUS_RESULT_SET",
                "缺少可验证的上一组商品结果",
                retryable=False,
            )
        _, products = _load_persisted_result_set(base_result_set, parsed.scope.result_set_view)
    source_data_fingerprint = _catalog_data_fingerprint(products)
    if parsed.availability.scope == "AVAILABLE":
        products = [product for product in products if product["stock"] > 0]
    resolved_filters = []
    for predicate in parsed.filters:
        field, operator, expected = _resolved_filter(predicate, products)
        resolved_filters.append({
            "requirement_id": predicate.requirement_id,
            "field": field.model_dump(mode="json"),
            "operator": operator,
            "value": expected,
        })
        products = [
            product for product in products
            if _compare(_field_value(product, field), operator, expected) is PredicateTruth.TRUE
        ]
    ordered = _partitioned(_ordered(products, parsed.ordering), parsed.partition)
    selected = _select(ordered, parsed.selector)
    rows = _projection_rows(selected, parsed.projection)
    population_products = [_catalog_record(product) for product in ordered]
    population_ids = [product["product_id"] for product in population_products]
    member_ids = [product["product_id"] for product in selected]
    result_identity = {
        "snapshot": snapshot.fingerprint,
        "catalog_data": source_data_fingerprint,
        "spec": parsed.model_dump(mode="json"),
        "base": (base_result_set or {}).get("result_set_id"),
        "base_view": parsed.scope.result_set_view if parsed.scope.kind == "PREVIOUS_RESULT_SET" else None,
        "population": population_ids,
        "members": member_ids,
    }
    encoded = json.dumps(result_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result_set_id = "crs_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    artifact = {
        "artifact_type": "catalog_query_result",
        "version": CATALOG_QUERY_VERSION,
        "result_set_id": result_set_id,
        "source_result_set_id": (base_result_set or {}).get("result_set_id"),
        "source_result_set_view": (
            parsed.scope.result_set_view if parsed.scope.kind == "PREVIOUS_RESULT_SET" else None
        ),
        "catalog_snapshot": snapshot.audit_identity(),
        "catalog_data_fingerprint": source_data_fingerprint,
        "population_data_fingerprint": _catalog_data_fingerprint(population_products),
        "result_data_fingerprint": _catalog_data_fingerprint(selected),
        "query_spec": parsed.model_dump(mode="json"),
        "compilation_coverage": expected_coverage,
        "resolved_filters": resolved_filters,
        "matched_count": len(products),
        "delivered_count": len(rows),
        "population_ids": population_ids,
        "population_products": population_products,
        "selected_ids": member_ids,
        "delivered_ids": member_ids,
        "member_ids": member_ids,
        "products": rows,
        "groups": _groups(rows, parsed.partition),
        "metrics": _metrics(products, parsed.metrics),
        "delivery": _delivery_evidence(parsed, len(products), len(rows)),
        "generated_at": datetime.now(UTC).isoformat(),
        "data_source": "DEMO_CATALOG",
    }
    return {"artifact": artifact, "products": rows, "total": len(products),
            "data_source": "DEMO_CATALOG"}


def validate_catalog_operation(result: object, arguments: dict) -> None:
    if not isinstance(result, dict) or not isinstance(result.get("artifact"), dict):
        raise AgentError("INVALID_TOOL_RESULT", "目录查询没有返回结构化结果产物", retryable=False)
    artifact = result["artifact"]
    required = {
        "artifact_type", "version", "result_set_id", "catalog_snapshot", "catalog_data_fingerprint",
        "population_data_fingerprint", "result_data_fingerprint", "query_spec",
        "compilation_coverage", "resolved_filters", "matched_count", "delivered_count", "population_ids",
        "population_products", "selected_ids", "delivered_ids", "member_ids",
        "products", "groups", "metrics", "delivery", "data_source",
    }
    if not required <= artifact.keys() or artifact.get("artifact_type") != "catalog_query_result":
        raise AgentError("INVALID_TOOL_RESULT", "目录结果产物缺少必要证据", retryable=False)
    spec = CatalogQuerySpec.model_validate(arguments["spec"], strict=True)
    if artifact.get("query_spec") != spec.model_dump(mode="json"):
        raise AgentError("CATALOG_QUERY_EVIDENCE_MISMATCH", "结果产物与执行查询不一致", retryable=False)
    coverage = build_compilation_coverage(spec)
    if arguments.get("compilation_coverage") != coverage or artifact.get("compilation_coverage") != coverage:
        raise AgentError("INCOMPLETE_COMPILATION_COVERAGE", "目录目标编译覆盖不完整", retryable=False)
    if artifact.get("resolved_filters") != [_expected_filter_payload(item) for item in spec.filters]:
        raise AgentError("CATALOG_FILTER_EVIDENCE_MISMATCH", "目录筛选绑定与语义目标不一致", retryable=False)
    rows = artifact.get("products")
    ids = artifact.get("member_ids")
    population_ids = artifact.get("population_ids")
    population_rows = artifact.get("population_products")
    selected_ids = artifact.get("selected_ids")
    delivered_ids = artifact.get("delivered_ids")
    if (not isinstance(rows, list) or not isinstance(ids, list) or not isinstance(population_ids, list)
            or not isinstance(population_rows, list)
            or ids != [row.get("product_id") for row in rows]):
        raise AgentError("INVALID_CATALOG_RESULT_SET", "目录结果成员与证据行不一致", retryable=False)
    if (len(ids) != len(set(ids)) or len(population_ids) != len(set(population_ids))
            or population_ids != [row.get("product_id") for row in population_rows]
            or selected_ids != ids or delivered_ids != ids
            or not set(ids) <= set(population_ids)
            or artifact.get("matched_count") != len(population_ids)
            or artifact.get("delivered_count") != len(rows)):
        raise AgentError("INVALID_CATALOG_RESULT_SET", "目录结果包含重复或错误计数", retryable=False)
    if artifact.get("result_data_fingerprint") != _catalog_data_fingerprint(rows):
        raise AgentError("CATALOG_RESULT_SET_CORRUPT", "目录结果行与数据指纹不一致", retryable=False)
    if artifact.get("population_data_fingerprint") != _catalog_data_fingerprint(population_rows):
        raise AgentError("CATALOG_RESULT_SET_CORRUPT", "目录总体行与数据指纹不一致", retryable=False)


def _authoritative_population(spec: CatalogQuerySpec, arguments: dict, resolved_filters: list[dict]) -> list[dict]:
    """Independently reconstruct the declared population from authoritative rows."""

    from src import tools

    with tools._session() as session:
        records = session.scalars(select(tools.Product).order_by(tools.Product.product_id)).all()
        products = [tools._product_dict(product) for product in records]
    if spec.scope.kind == "PREVIOUS_RESULT_SET":
        _, products = _load_persisted_result_set(
            arguments.get("base_result_set") or {}, spec.scope.result_set_view
        )
    fingerprint_key = (
        "population_data_fingerprint" if spec.scope.result_set_view == "POPULATION"
        else "result_data_fingerprint"
    )
    expected_fingerprint = (arguments.get("base_result_set") or {}).get(fingerprint_key)
    current_fingerprint = _catalog_data_fingerprint(products)
    if expected_fingerprint and current_fingerprint != expected_fingerprint:
        raise AgentError(
            "CATALOG_RESULT_SET_CORRUPT",
            "上一组商品结果的数据指纹与执行输入不一致",
            retryable=False,
        )
    if spec.availability.scope == "AVAILABLE":
        products = [product for product in products if product["stock"] > 0]
    for item in resolved_filters:
        field = CatalogFieldRef.model_validate(item["field"], strict=True)
        products = [product for product in products
                    if _compare(_field_value(product, field), item["operator"], item["value"])
                    is PredicateTruth.TRUE]
    return _partitioned(_ordered(products, spec.ordering), spec.partition)


def verify_catalog_goal(result: dict, arguments: dict) -> dict:
    """Re-evaluate the declared goal over returned evidence, independent of rendering."""

    artifact = result["artifact"]
    spec = CatalogQuerySpec.model_validate(arguments["spec"], strict=True)
    rows = artifact["products"]
    failures: list[str] = []
    resolved = artifact.get("resolved_filters") or []
    if [item.get("requirement_id") for item in resolved] != [item.requirement_id for item in spec.filters]:
        failures.extend(item.requirement_id for item in spec.filters)
    for item in resolved:
        try:
            field = CatalogFieldRef.model_validate(item["field"], strict=True)
            operator, expected = item["operator"], item["value"]
        except (KeyError, TypeError, ValueError):
            failures.append(str(item.get("requirement_id") or "filter"))
            continue
        if any(_compare(_field_value(row, field), operator, expected) is not PredicateTruth.TRUE for row in rows):
            failures.append(str(item["requirement_id"]))
    population = _authoritative_population(spec, arguments, resolved)
    source_products = population
    if spec.availability.scope == "AVAILABLE" or spec.filters:
        # The artifact fingerprint binds the unfiltered source population. The
        # authoritative source is reconstructed separately to avoid accepting
        # a result-derived fingerprint.
        if spec.scope.kind == "PREVIOUS_RESULT_SET":
            _, source_products = _load_persisted_result_set(
                arguments.get("base_result_set") or {}, spec.scope.result_set_view
            )
        else:
            from src import tools
            with tools._session() as session:
                records = session.scalars(select(tools.Product).order_by(tools.Product.product_id)).all()
                source_products = [tools._product_dict(product) for product in records]
    if artifact.get("catalog_data_fingerprint") != _catalog_data_fingerprint(source_products):
        failures.append(spec.scope.requirement_id)
    expected_population_ids = [product["product_id"] for product in population]
    expected_population_rows = [_catalog_record(product) for product in population]
    if (
        artifact.get("population_ids") != expected_population_ids
        or artifact.get("population_products") != expected_population_rows
    ):
        failures.extend([spec.scope.requirement_id, spec.availability.requirement_id,
                         *(item.requirement_id for item in spec.filters),
                         *(item.requirement_id for item in spec.ordering)])
    expected_selected = _select(population, spec.selector)
    expected_rows = _projection_rows(expected_selected, spec.projection)
    if rows != expected_rows:
        failures.extend([spec.selector.requirement_id, spec.projection.requirement_id])
    expected_groups = _groups(expected_rows, spec.partition)
    if artifact.get("groups") != expected_groups:
        failures.append(spec.partition.requirement_id if spec.partition else spec.selector.requirement_id)
    expected_metrics = _metrics(population, spec.metrics)
    if spec.metrics and artifact.get("metrics") != expected_metrics:
        failures.extend(metric.requirement_id for metric in spec.metrics)
    verdict = "SATISFIED" if not failures else "FAILED"
    return {"verdict": verdict, "failed_requirements": list(dict.fromkeys(failures)),
            "checked_requirements": [row.requirement_id for row in spec.requirements()]}


def verify_catalog_delivery(result: dict, arguments: dict) -> dict:
    artifact = result["artifact"]
    spec = CatalogQuerySpec.model_validate(arguments["spec"], strict=True)
    delivery = artifact.get("delivery") or {}
    failures = []
    if delivery.get("delivered_count") != len(artifact.get("products") or []):
        failures.append("DELIVERED_COUNT_MISMATCH")
    expected = _delivery_evidence(spec, artifact.get("matched_count"), artifact.get("delivered_count"))
    if delivery != expected:
        failures.append("DELIVERY_EVIDENCE_MISMATCH")
    if spec.selector.mode == "ALL" and (
        not delivery.get("population_complete")
        or artifact.get("matched_count") != artifact.get("delivered_count")
    ):
        failures.append("COMPLETE_RESULT_TRUNCATED")
    return {"verdict": "SATISFIED" if not failures else "FAILED", "failures": failures,
            "mode": spec.selector.mode, "complete": bool(delivery.get("complete")),
            "population_complete": bool(delivery.get("population_complete")),
            "has_more": bool(delivery.get("has_more")), "next_page": delivery.get("next_page")}


def verify_catalog_result(result: object, arguments: dict) -> tuple[dict, dict]:
    artifact = result.get("artifact") if isinstance(result, dict) else None
    if isinstance(artifact, dict):
        artifact["verification"] = {"status": "PENDING"}
    try:
        validate_catalog_operation(result, arguments)
        goal = verify_catalog_goal(result, arguments)
        delivery = verify_catalog_delivery(result, arguments)
        if delivery["verdict"] != "SATISFIED":
            raise AgentError("CATALOG_DELIVERY_INCOMPLETE", "目录结果没有按请求完整交付", retryable=False)
        if goal["verdict"] != "SATISFIED":
            raise AgentError("CATALOG_GOAL_NOT_SATISFIED", "目录结果没有满足全部语义目标", retryable=False)
    except AgentError as exc:
        if isinstance(artifact, dict):
            artifact["verification"] = {
                "status": "REJECTED",
                "error": {"code": exc.code, "message": str(exc), "retryable": exc.retryable},
            }
        raise
    artifact["verification"] = {"status": "VERIFIED", "goal": goal, "delivery": delivery}
    return goal, delivery


def render_catalog_result(result: dict) -> str:
    artifact = result["artifact"]
    spec = CatalogQuerySpec.model_validate(artifact["query_spec"], strict=True)
    rows = artifact["products"]
    if not rows and not artifact["metrics"]:
        return "当前真实目录中没有符合这些条件的商品。"
    labels = {
        "product_id": "商品编号", "name": "商品名称", "category": "类目",
        "price": "价格", "stock": "库存", "summary": "商品摘要",
    }

    def field_label(field: CatalogFieldRef | None) -> str:
        if field is None:
            return "商品"
        if field.core is not None:
            return labels[field.core]
        return get_catalog_concept_registry().definition(str(field.concept_id)).description

    def display_value(field: CatalogFieldRef | None, value: object, row: dict | None = None) -> str:
        if field and field.core == "price" and isinstance(value, (int, float)):
            return f"¥{value:.2f}"
        if field and field.core == "stock" and isinstance(value, (int, float)):
            return f"{value:g} 件"
        return str(value if value is not None else "无可用值")

    if spec.selector.mode in {"ARGMIN", "ARGMAX"} and rows:
        row = rows[0]
        label = "最低值" if spec.selector.mode == "ARGMIN" else "最高值"
        scope = "上一结果范围" if spec.scope.kind == "PREVIOUS_RESULT_SET" else "当前目录范围"
        field = spec.selector.field
        value = _field_value(row, field)
        return (
            f"在{scope}内，{field_label(field)}{label}对应 {row['name']}（{row['product_id']}），"
            f"{field_label(field)}为 {display_value(field, value, row)}。"
        )
    if artifact["metrics"]:
        metric_labels = {"COUNT": "数量", "MIN": "最低", "MAX": "最高", "AVG": "平均", "SUM": "合计"}
        values = []
        for metric in spec.metrics:
            label = metric_labels[metric.function]
            if metric.field is not None:
                label += field_label(metric.field)
            values.append(f"{label}：{display_value(metric.field, artifact['metrics'].get(metric.alias))}")
        metrics = "，".join(values)
        return f"已完成目录聚合：{metrics}。"
    partition_label = labels.get(spec.partition.field.identity, spec.partition.field.identity) if spec.partition else ""
    partition = f"，按{partition_label}分为 {len(artifact['groups'])} 组" if spec.partition else ""
    ordering = "，并按指定顺序排列" if spec.ordering else ""
    completeness = "完整结果已附在下方结构化目录中" if spec.selector.mode == "ALL" else "所选结果已附在下方结构化目录中"
    return f"已从真实目录得到 {artifact['delivered_count']} 款商品{partition}{ordering}；{completeness}。"
