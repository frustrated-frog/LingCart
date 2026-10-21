"""受限、可验证的商品目录查询协议。

模型只能声明语义 AST；字段绑定、结果集引用、执行和核验都由 Stable Kernel
完成。这个模块刻意不接受 SQL、Python 表达式或任意 JSONPath。
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import UTC, datetime
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
AvailabilityScope = Literal["ALL", "AVAILABLE"]
Scalar = str | int | float


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


def _compare(actual: object, operator: str, expected: object) -> bool:
    if operator in {"CONTAINS", "NOT_CONTAINS"}:
        found = str(expected).casefold() in str(actual).casefold()
        return found if operator == "CONTAINS" else not found
    if operator == "IN":
        return actual in expected
    try:
        if operator == "EQ":
            return actual == expected
        if operator == "NE":
            return actual != expected
        if operator == "LT":
            return actual < expected
        if operator == "LTE":
            return actual <= expected
        if operator == "GT":
            return actual > expected
        if operator == "GTE":
            return actual >= expected
    except TypeError:
        return False
    return False


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
            return {"error": "缺少可验证的上一组商品结果"}
        allowed = set(base_result_set["member_ids"])
        products = [product for product in products if product["product_id"] in allowed]
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
        products = [product for product in products if _compare(_field_value(product, field), operator, expected)]
    ordered = _partitioned(_ordered(products, parsed.ordering), parsed.partition)
    selected = _select(ordered, parsed.selector)
    rows = _projection_rows(selected, parsed.projection)
    member_ids = [product["product_id"] for product in selected]
    result_identity = {
        "snapshot": snapshot.fingerprint,
        "spec": parsed.model_dump(mode="json"),
        "base": (base_result_set or {}).get("result_set_id"),
        "members": member_ids,
    }
    encoded = json.dumps(result_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result_set_id = "crs_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]
    requested_complete = parsed.selector.mode == "ALL"
    artifact = {
        "artifact_type": "catalog_query_result",
        "version": CATALOG_QUERY_VERSION,
        "result_set_id": result_set_id,
        "source_result_set_id": (base_result_set or {}).get("result_set_id"),
        "catalog_snapshot": snapshot.audit_identity(),
        "query_spec": parsed.model_dump(mode="json"),
        "compilation_coverage": expected_coverage,
        "resolved_filters": resolved_filters,
        "matched_count": len(products),
        "delivered_count": len(rows),
        "member_ids": member_ids,
        "products": rows,
        "groups": _groups(rows, parsed.partition),
        "metrics": _metrics(products, parsed.metrics),
        "delivery": {
            "mode": parsed.selector.mode,
            "requested_complete": requested_complete,
            "complete": (not requested_complete) or len(rows) == len(products),
            "matched_count": len(products),
            "delivered_count": len(rows),
        },
        "generated_at": datetime.now(UTC).isoformat(),
        "data_source": "DEMO_CATALOG",
    }
    return {"artifact": artifact, "products": rows, "total": len(products),
            "data_source": "DEMO_CATALOG"}


def _field_values(rows: list[dict], field: CatalogFieldRef) -> list[object]:
    return [_field_value(row, field) for row in rows]


def validate_catalog_operation(result: object, arguments: dict) -> None:
    if not isinstance(result, dict) or not isinstance(result.get("artifact"), dict):
        raise AgentError("INVALID_TOOL_RESULT", "目录查询没有返回结构化结果产物", retryable=False)
    artifact = result["artifact"]
    required = {
        "artifact_type", "version", "result_set_id", "catalog_snapshot", "query_spec",
        "compilation_coverage", "resolved_filters", "matched_count", "delivered_count", "member_ids",
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
    if not isinstance(rows, list) or not isinstance(ids, list) or ids != [row.get("product_id") for row in rows]:
        raise AgentError("INVALID_CATALOG_RESULT_SET", "目录结果成员与证据行不一致", retryable=False)
    if len(ids) != len(set(ids)) or artifact.get("delivered_count") != len(rows):
        raise AgentError("INVALID_CATALOG_RESULT_SET", "目录结果包含重复或错误计数", retryable=False)


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
        if any(not _compare(_field_value(row, field), operator, expected) for row in rows):
            failures.append(str(item["requirement_id"]))
    if spec.partition:
        for group in artifact.get("groups") or []:
            allowed = set(group.get("product_ids") or [])
            group_rows = [row for row in rows if row.get("product_id") in allowed]
            ordered = _ordered(group_rows, spec.ordering)
            if [row["product_id"] for row in ordered] != [row["product_id"] for row in group_rows]:
                failures.extend(item.requirement_id for item in spec.ordering)
                break
    else:
        ordered = _ordered(rows, spec.ordering)
        if [row["product_id"] for row in ordered] != [row["product_id"] for row in rows]:
            failures.extend(item.requirement_id for item in spec.ordering)
    if spec.partition:
        groups = artifact.get("groups") or []
        grouped_ids = [identity for group in groups for identity in group.get("product_ids", [])]
        if grouped_ids != [row["product_id"] for row in rows]:
            failures.append(spec.partition.requirement_id)
    if spec.selector.mode in {"ARGMIN", "ARGMAX"} and rows:
        values = _field_values(rows, spec.selector.field)
        if any(value is None for value in values) or len(set(values)) != 1:
            failures.append(spec.selector.requirement_id)
    expected_metrics = _metrics(rows, spec.metrics)
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
    if spec.selector.mode == "ALL" and (
        not delivery.get("complete")
        or artifact.get("matched_count") != artifact.get("delivered_count")
    ):
        failures.append("COMPLETE_RESULT_TRUNCATED")
    return {"verdict": "SATISFIED" if not failures else "FAILED", "failures": failures,
            "mode": spec.selector.mode, "complete": bool(delivery.get("complete"))}


def verify_catalog_result(result: object, arguments: dict) -> tuple[dict, dict]:
    validate_catalog_operation(result, arguments)
    goal = verify_catalog_goal(result, arguments)
    delivery = verify_catalog_delivery(result, arguments)
    if delivery["verdict"] != "SATISFIED":
        raise AgentError("CATALOG_DELIVERY_INCOMPLETE", "目录结果没有按请求完整交付", retryable=False)
    if goal["verdict"] != "SATISFIED":
        raise AgentError("CATALOG_GOAL_NOT_SATISFIED", "目录结果没有满足全部语义目标", retryable=False)
    return goal, delivery


def render_catalog_result(result: dict) -> str:
    artifact = result["artifact"]
    spec = CatalogQuerySpec.model_validate(artifact["query_spec"], strict=True)
    rows = artifact["products"]
    if not rows and not artifact["metrics"]:
        return "当前真实目录中没有符合这些条件的商品。"
    if spec.selector.mode in {"ARGMIN", "ARGMAX"} and rows:
        row = rows[0]
        label = "最低值" if spec.selector.mode == "ARGMIN" else "最高值"
        return f"在上一结果范围内，{label}对应 {row['name']}（{row['product_id']}），价格 ¥{row['price']:.2f}。"
    if artifact["metrics"] and not rows:
        metrics = "，".join(f"{key}：{value}" for key, value in artifact["metrics"].items())
        return f"已完成目录聚合：{metrics}。"
    partition = f"，按 {spec.partition.field.identity} 分为 {len(artifact['groups'])} 组" if spec.partition else ""
    ordering = "，并按指定顺序排列" if spec.ordering else ""
    completeness = "完整结果已附在下方结构化目录中" if spec.selector.mode == "ALL" else "所选结果已附在下方结构化目录中"
    return f"已从真实目录得到 {artifact['delivered_count']} 款商品{partition}{ordering}；{completeness}。"
