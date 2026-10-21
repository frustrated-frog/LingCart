"""Runtime capability contracts for semantic request routing.

The dialogue model describes outcomes with stable protocol fields.  Business
coverage lives in this registry, so adding a capability does not require a new
branch in the dialogue parser or orchestrator.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Callable, Mapping

from src import tools
from src.catalog_metadata import CATALOG_FACETS_DEPENDENCY, CatalogMetadataProvider, CatalogMetadataSnapshot
from src.catalog_query import CATALOG_QUERY_OPERATION, compile_catalog_query, query_catalog
from src.errors import AgentError
from src.intake import (
    FAQ,
    CATALOG_QUERY,
    ORDER_DETAILS,
    ORDER_STATUS,
    PAYMENT_PROCESS,
    PAYMENT_STATUS,
    PRODUCT_COMPARE,
    PRODUCT_DISCOVERY,
    PRODUCT_SEARCH,
    REFUND_ELIGIBILITY,
    REFUND_REQUEST,
    SHIPMENT_TRACKING,
)


GENERAL_ANSWER = "GENERAL_ANSWER"
SYSTEM_TIME = "SYSTEM_TIME"
CALCULATE = "CALCULATE"
MODEL_ANSWER_OPERATION = "__model_answer__"


@dataclass(frozen=True)
class CompiledOperation:
    """One bounded operation emitted by a registered capability compiler."""

    ref: str
    specialist: str
    operation: str
    arguments: dict
    depends_on: tuple[str, ...] = ()


Compiler = Callable[[Mapping[str, object], str], tuple[CompiledOperation, ...]]
Normalizer = Callable[[dict, tuple[str, ...]], tuple[dict, dict[str, dict]]]
ResultValidator = Callable[[object, Mapping[str, object]], bool]
ResultRenderer = Callable[[object, Mapping[str, object]], str]
OutcomeMapper = Callable[[object, Mapping[str, object]], str]


@dataclass(frozen=True)
class OperationContract:
    """Execution and observation contract for a dynamically registered tool."""

    handler: object
    required_result_fields: tuple[str, ...] = ()
    result_validator: ResultValidator | None = None
    renderer: ResultRenderer | None = None
    outcome_mapper: OutcomeMapper | None = None

    def validate_result(self, result: object, arguments: Mapping[str, object]) -> bool:
        fields_ok = isinstance(result, dict) and set(self.required_result_fields) <= result.keys()
        if not fields_ok:
            return False
        try:
            return self.result_validator is None or bool(self.result_validator(result, arguments))
        except Exception:
            return False

    def render_result(self, result: object, arguments: Mapping[str, object]) -> str:
        try:
            rendered = self.renderer(result, arguments) if self.renderer else str(result)
        except Exception as exc:
            raise AgentError("CAPABILITY_RENDER_FAILED", "注册能力无法安全渲染已验证结果", retryable=False) from exc
        if not isinstance(rendered, str) or not rendered.strip():
            raise AgentError("CAPABILITY_RENDER_FAILED", "注册能力返回了空或无效文本", retryable=False)
        return rendered

    def outcome(self, result: object, arguments: Mapping[str, object]) -> str:
        try:
            outcome = self.outcome_mapper(result, arguments) if self.outcome_mapper else "SATISFIED"
        except Exception as exc:
            raise AgentError("INVALID_CAPABILITY_OUTCOME", "注册能力无法判定目标结果", retryable=False) from exc
        allowed = {"SATISFIED", "UNAVAILABLE_EVIDENCE", "WAITING_APPROVAL", "FAILED"}
        if outcome not in allowed:
            raise AgentError("INVALID_CAPABILITY_OUTCOME", f"注册能力返回未知目标状态：{outcome}", retryable=False)
        return outcome


@dataclass(frozen=True)
class CapabilitySpec:
    """Description, validation rules and executor binding for one capability."""

    capability_id: str
    objective_kind: str
    description: str
    target_type: str
    evidence_requirement: str
    required_fields: tuple[str, ...] = ()
    accepted_fields: tuple[str, ...] = ()
    min_items: Mapping[str, int] = field(default_factory=dict)
    max_items: Mapping[str, int] = field(default_factory=dict)
    required_parameters: tuple[str, ...] = ()
    parameter_schema: Mapping[str, str] = field(default_factory=dict)
    normalizer_version: str | None = None
    ungrounded_optional_defaults: Mapping[str, object] = field(default_factory=dict)
    alternative_evidence_requirements: tuple[str, ...] = ()
    supports_under_specified_goal: bool = False
    side_effect: str = "READ_ONLY"
    missing_question: str = "请补充完成这个请求所需的关键信息。"
    metadata_dependencies: tuple[str, ...] = ()

    def public_payload(self) -> dict:
        return {
            "capability_id": self.capability_id,
            "objective_kind": self.objective_kind,
            "description": self.description,
            "target_type": self.target_type,
            "evidence_requirement": self.evidence_requirement,
            "required_fields": list(self.required_fields),
            "accepted_fields": list(self.accepted_fields),
            "min_items": dict(self.min_items),
            "max_items": dict(self.max_items),
            "required_parameters": list(self.required_parameters),
            "parameter_schema": dict(self.parameter_schema),
            "normalizer_version": self.normalizer_version,
            "ungrounded_optional_defaults": dict(self.ungrounded_optional_defaults),
            "evidence_requirements": [self.evidence_requirement, *self.alternative_evidence_requirements],
            "supports_under_specified_goal": self.supports_under_specified_goal,
            "side_effect": self.side_effect,
            "metadata_dependencies": list(self.metadata_dependencies),
        }


@dataclass(frozen=True)
class CapabilityRegistration:
    spec: CapabilitySpec
    compiler: Compiler
    tools: Mapping[str, object] = field(default_factory=dict)
    normalizer: Normalizer | None = None
    operations: Mapping[str, OperationContract] = field(default_factory=dict)


class CapabilityRegistry:
    """Explicit, inspectable registry used by understanding and execution."""

    def __init__(
        self,
        registrations: tuple[CapabilityRegistration, ...] = (),
        *,
        metadata_providers: Mapping[str, CatalogMetadataProvider] | None = None,
    ):
        self._registrations: dict[str, CapabilityRegistration] = {}
        self._tools: dict[str, object] = {}
        self._operations: dict[str, OperationContract] = {}
        self._metadata_providers = dict(metadata_providers or {})
        for registration in registrations:
            self.register(registration)

    def register(self, registration: CapabilityRegistration) -> None:
        kind = registration.spec.objective_kind
        if not kind or kind in self._registrations:
            raise ValueError(f"Capability objective kind is missing or duplicated: {kind}")
        droppable = set(registration.spec.ungrounded_optional_defaults)
        undeclared = droppable - set(registration.spec.accepted_fields)
        if undeclared:
            raise ValueError(f"Ungrounded optional fields must be accepted by {kind}: {sorted(undeclared)}")
        required_droppable = droppable & set(registration.spec.required_fields)
        if required_droppable:
            raise ValueError(f"Required fields cannot be dropped by {kind}: {sorted(required_droppable)}")
        for name, handler in registration.tools.items():
            existing = self._tools.get(name)
            if existing is not None and existing is not handler:
                raise ValueError(f"Operation {name} has conflicting handlers")
        for name, contract in registration.operations.items():
            existing = self._operations.get(name)
            if existing is not None and existing != contract:
                raise ValueError(f"Operation {name} has conflicting contracts")
            existing_handler = self._tools.get(name)
            if existing_handler is not None and existing_handler is not contract.handler:
                raise ValueError(f"Operation {name} has conflicting handlers")
        self._registrations[kind] = registration
        self._tools.update(registration.tools)
        self._operations.update(registration.operations)
        self._tools.update({name: contract.handler for name, contract in registration.operations.items()})

    def registration(self, objective_kind: str) -> CapabilityRegistration:
        try:
            return self._registrations[objective_kind]
        except KeyError as exc:
            raise AgentError(
                "UNKNOWN_CAPABILITY",
                f"语义目标引用了未注册能力：{objective_kind}",
                retryable=False,
            ) from exc

    def compile(self, objective: Mapping[str, object], request: str) -> tuple[CompiledOperation, ...]:
        return self.registration(str(objective["kind"])).compiler(objective, request)

    def missing_fields(self, objective: Mapping[str, object]) -> tuple[str, ...]:
        spec = self.registration(str(objective["kind"])).spec
        missing = [field for field in spec.required_fields if objective.get(field) in (None, "", [])]
        for name, minimum in spec.min_items.items():
            value = objective.get(name)
            if not isinstance(value, (list, tuple)) or len(value) < minimum:
                missing.append(name)
        parameters = objective.get("parameters") if isinstance(objective.get("parameters"), dict) else {}
        missing.extend(f"parameters.{name}" for name in spec.required_parameters
                       if parameters.get(name) in (None, "", []))
        return tuple(dict.fromkeys(missing))

    def normalize_arguments(self, objective: Mapping[str, object], sources: tuple[str, ...]) -> tuple[dict, dict[str, dict]]:
        registration = self.registration(str(objective["kind"]))
        normalized = dict(objective)
        provenance: dict[str, dict] = {}
        if registration.normalizer is not None:
            normalized, provenance = registration.normalizer(normalized, sources)
        folded_sources = [source.casefold() for source in sources]

        def grounded(value: object) -> bool:
            if isinstance(value, str):
                return bool(value) and any(value.casefold() in source for source in folded_sources)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                variants = {str(value), f"{value:g}"}
                return any(
                    re.search(r"(?<![\d.])" + re.escape(candidate) + r"(?![\d.])", source)
                    for source in folded_sources
                    for candidate in variants
                )
            return False

        for field_name, safe_default in registration.spec.ungrounded_optional_defaults.items():
            value = normalized.get(field_name)
            if value in (None, "", [], {}):
                continue
            if isinstance(value, list):
                kept = [item for item in value if grounded(item)]
                if kept == value:
                    continue
                normalized[field_name] = kept if kept else safe_default
            elif grounded(value):
                continue
            else:
                normalized[field_name] = safe_default
            provenance[field_name] = {
                "name": field_name,
                "raw_text": str(value),
                "normalized_value": normalized[field_name],
                "resolver": "ungrounded_optional_filter",
                "resolver_version": "1",
                "source_quote": str(objective.get("source_quote") or ""),
                "confidence": 1.0,
            }
        self.validate_arguments(normalized)
        return normalized, provenance

    def validate_arguments(self, objective: Mapping[str, object]) -> None:
        """Enforce a capability's declared contract without parser branches."""
        spec = self.registration(str(objective["kind"])).spec
        protocol_fields = {
            "id", "kind", "source_quote", "desired_outcome", "target_type", "evidence_requirement",
        }
        supplied = {
            name for name, value in objective.items()
            if name not in protocol_fields and value not in (None, "", [], {})
        }
        unexpected = supplied - set(spec.accepted_fields)
        if unexpected:
            raise AgentError(
                "INVALID_CAPABILITY_ARGUMENTS",
                f"能力 {spec.objective_kind} 不接受字段：{', '.join(sorted(unexpected))}",
                retryable=False,
            )
        parameters = objective.get("parameters")
        if parameters not in (None, {}) and not isinstance(parameters, dict):
            raise AgentError("INVALID_CAPABILITY_ARGUMENTS", "parameters 必须是对象", retryable=False)
        parameters = parameters or {}
        unknown_parameters = set(parameters) - set(spec.parameter_schema)
        if unknown_parameters:
            raise AgentError(
                "INVALID_CAPABILITY_ARGUMENTS",
                f"能力 {spec.objective_kind} 不接受参数：{', '.join(sorted(unknown_parameters))}",
                retryable=False,
            )
        type_checks = {
            "string": lambda value: isinstance(value, str),
            "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
            "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": lambda value: isinstance(value, bool),
            "array": lambda value: isinstance(value, list),
            "object": lambda value: isinstance(value, dict),
        }
        invalid_parameters = [name for name, value in parameters.items()
                              if spec.parameter_schema.get(name) not in type_checks
                              or not type_checks[spec.parameter_schema[name]](value)]
        if invalid_parameters:
            raise AgentError(
                "INVALID_CAPABILITY_ARGUMENTS",
                f"能力 {spec.objective_kind} 的参数类型无效：{', '.join(sorted(invalid_parameters))}",
                retryable=False,
            )
        for name, maximum in spec.max_items.items():
            value = objective.get(name)
            if isinstance(value, (list, tuple)) and len(value) > maximum:
                raise AgentError(
                    "INVALID_CAPABILITY_ARGUMENTS",
                    f"能力 {spec.objective_kind} 的 {name} 最多接受 {maximum} 项",
                    retryable=False,
                )
        target_type = objective.get("target_type")
        if target_type and target_type != spec.target_type:
            raise AgentError(
                "INVALID_CAPABILITY_ARGUMENTS",
                f"能力 {spec.objective_kind} 的 target_type 必须是 {spec.target_type}",
                retryable=False,
            )
        evidence = objective.get("evidence_requirement")
        allowed_evidence = {spec.evidence_requirement, *spec.alternative_evidence_requirements}
        if evidence and evidence not in allowed_evidence:
            raise AgentError(
                "INVALID_CAPABILITY_ARGUMENTS",
                f"能力 {spec.objective_kind} 不支持证据类型 {evidence}",
                retryable=False,
            )

    def execution_tools(self) -> dict[str, object]:
        return dict(self._tools)

    def metadata_snapshot(self, objective_kind: str, dependency: str) -> CatalogMetadataSnapshot:
        """Resolve metadata only when the requesting capability declares it."""

        spec = self.registration(objective_kind).spec
        if dependency not in spec.metadata_dependencies:
            raise AgentError(
                "UNDECLARED_METADATA_DEPENDENCY",
                f"能力 {objective_kind} 未声明元数据依赖：{dependency}",
                retryable=False,
            )
        provider = self._metadata_providers.get(dependency)
        if provider is None:
            raise AgentError(
                "METADATA_PROVIDER_UNAVAILABLE",
                f"元数据依赖没有可用 Provider：{dependency}",
                retryable=False,
            )
        return provider.get_snapshot()

    def operation_contract(self, operation: str) -> OperationContract | None:
        return self._operations.get(operation)

    def prompt_catalog(self) -> list[dict]:
        return [registration.spec.public_payload() for registration in self._registrations.values()]

    @property
    def version(self) -> str:
        def identity(value: object | None) -> str | None:
            if value is None:
                return None
            value = getattr(value, "function", value)
            return f"{getattr(value, '__module__', type(value).__module__)}:{getattr(value, '__qualname__', type(value).__qualname__)}"

        manifest = []
        for registration in self._registrations.values():
            manifest.append({
                "spec": registration.spec.public_payload(),
                "compiler": identity(registration.compiler),
                "normalizer": identity(registration.normalizer),
                "operations": {
                    name: {
                        "handler": identity(contract.handler),
                        "required_result_fields": list(contract.required_result_fields),
                        "result_validator": identity(contract.result_validator),
                        "renderer": identity(contract.renderer),
                        "outcome_mapper": identity(contract.outcome_mapper),
                    }
                    for name, contract in sorted(registration.operations.items())
                },
            })
        encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _args(objective: Mapping[str, object], fields: tuple[str, ...]) -> dict:
    return {name: objective[name] for name in fields if objective.get(name) not in (None, "", [])}


def _single(specialist: str, operation: str, fields: tuple[str, ...]) -> Compiler:
    def compile_one(objective: Mapping[str, object], _request: str) -> tuple[CompiledOperation, ...]:
        return (CompiledOperation("result", specialist, operation, _args(objective, fields)),)

    return compile_one


def _refund_eligibility(objective: Mapping[str, object], _request: str) -> tuple[CompiledOperation, ...]:
    args = _args(objective, ("order_id",))
    return (
        CompiledOperation("order-state", "order", "get_order_status", args),
        CompiledOperation("eligibility", "refund", "check_refund_eligibility", args, ("order-state",)),
    )


def _refund_request(objective: Mapping[str, object], _request: str) -> tuple[CompiledOperation, ...]:
    args = _args(objective, ("order_id",))
    return (
        *_refund_eligibility(objective, _request),
        CompiledOperation("proposal", "refund", "prepare_refund", args, ("eligibility",)),
    )


def _payment_process(objective: Mapping[str, object], _request: str) -> tuple[CompiledOperation, ...]:
    args = _args(objective, ("order_id",))
    return (
        CompiledOperation("payment-state", "payment", "get_payment_status", args),
        CompiledOperation("proposal", "payment", "prepare_payment", args, ("payment-state",)),
    )


def _faq(objective: Mapping[str, object], _request: str) -> tuple[CompiledOperation, ...]:
    return (CompiledOperation("answer", "faq", "answer_faq", {"question": objective["question"]}),)


def _search(objective: Mapping[str, object], _request: str) -> tuple[CompiledOperation, ...]:
    return (CompiledOperation("catalog", "shopping", "search_products",
                              _args(objective, ("query", "category", "max_price", "excluded_terms"))),)


def _discover(objective: Mapping[str, object], request: str) -> tuple[CompiledOperation, ...]:
    arguments = _args(objective, ("query", "category", "max_price", "soft_preferences", "excluded_terms"))
    arguments["context_seed"] = request
    return (CompiledOperation("candidate-set", "shopping", "discover_products", arguments),)


def _compare(objective: Mapping[str, object], _request: str) -> tuple[CompiledOperation, ...]:
    return (CompiledOperation("comparison", "shopping", "compare_products",
                              {"product_ids": objective.get("product_ids", [])}),)


def _catalog_query(objective: Mapping[str, object], _request: str) -> tuple[CompiledOperation, ...]:
    arguments = compile_catalog_query(objective.get("catalog_query"))
    from src.runtime import current_runtime
    context = current_runtime()
    if context:
        context.record(
            "compilation_coverage",
            objective_id=objective.get("id"),
            operation=CATALOG_QUERY_OPERATION,
            coverage=arguments["compilation_coverage"],
        )
    return (CompiledOperation("catalog-result", "shopping", CATALOG_QUERY_OPERATION, arguments),)


def _general_answer(objective: Mapping[str, object], _request: str) -> tuple[CompiledOperation, ...]:
    return (CompiledOperation("answer", "knowledge", MODEL_ANSWER_OPERATION, {
        "question": objective["question"],
        "evidence_requirement": objective.get("evidence_requirement") or "MODEL_KNOWLEDGE",
    }),)


def _clock(objective: Mapping[str, object], _request: str) -> tuple[CompiledOperation, ...]:
    return (CompiledOperation("clock", "utility", "read_clock", {
        "timezone": objective.get("timezone") or "Asia/Shanghai",
    }),)


def _calculate(objective: Mapping[str, object], _request: str) -> tuple[CompiledOperation, ...]:
    return (CompiledOperation("calculation", "utility", "calculate", {
        "expression": objective["expression"],
    }),)


def _spec(kind: str, capability_id: str, description: str, target: str, evidence: str,
          compiler: Compiler, *, required: tuple[str, ...] = (), accepted: tuple[str, ...] = (),
          min_items: Mapping[str, int] | None = None, max_items: Mapping[str, int] | None = None,
          required_parameters: tuple[str, ...] = (), parameter_schema: Mapping[str, str] | None = None,
          normalizer_version: str | None = None,
          ungrounded_optional_defaults: Mapping[str, object] | None = None,
          alternative_evidence: tuple[str, ...] = (),
          under_specified: bool = False, side_effect: str = "READ_ONLY",
          metadata_dependencies: tuple[str, ...] = (),
          missing_question: str = "请补充完成这个请求所需的关键信息。",
          handlers: Mapping[str, object] | None = None, normalizer: Normalizer | None = None,
          operations: Mapping[str, OperationContract] | None = None) -> CapabilityRegistration:
    return CapabilityRegistration(
        CapabilitySpec(
            capability_id=capability_id,
            objective_kind=kind,
            description=description,
            target_type=target,
            evidence_requirement=evidence,
            required_fields=required,
            accepted_fields=accepted,
            min_items=min_items or {},
            max_items=max_items or {},
            required_parameters=required_parameters,
            parameter_schema=parameter_schema or {},
            normalizer_version=normalizer_version,
            ungrounded_optional_defaults=ungrounded_optional_defaults or {},
            alternative_evidence_requirements=alternative_evidence,
            supports_under_specified_goal=under_specified,
            side_effect=side_effect,
            metadata_dependencies=metadata_dependencies,
            missing_question=missing_question,
        ),
        compiler,
        handlers or {},
        normalizer,
        operations or {},
    )


def _builtin_registrations() -> tuple[CapabilityRegistration, ...]:
    from src.entity_resolution import normalize_clock_objective

    order_fields = ("order_id",)
    order_question = "请提供要处理的订单号，例如“查询订单 1004 的状态和物流”。"
    return (
        _spec(SYSTEM_TIME, "system.clock.read", "读取当前日期、时间和星期；需要当前时间事实时必须使用",
              "runtime_clock", "RUNTIME_ENVIRONMENT", _clock, accepted=("timezone",),
              under_specified=True, handlers={"read_clock": tools.read_clock},
              normalizer=normalize_clock_objective, normalizer_version="trusted_timezone/1"),
        _spec(CALCULATE, "system.math.calculate", "执行只包含数字、括号和基础运算符的受限算术",
              "arithmetic", "DETERMINISTIC_COMPUTATION", _calculate, required=("expression",),
              accepted=("expression",), missing_question="请提供要计算的具体算式。",
              handlers={"calculate": tools.calculate}),
        _spec(ORDER_STATUS, "order.status.read", "查询订单当前状态", "order", "BUSINESS_DATA",
              _single("order", "get_order_status", order_fields), required=order_fields,
              accepted=order_fields, missing_question=order_question,
              handlers={"get_order_status": tools.get_order_status}),
        _spec(ORDER_DETAILS, "order.details.read", "查询订单详情", "order", "BUSINESS_DATA",
              _single("order", "get_order_details", order_fields), required=order_fields,
              accepted=order_fields, missing_question=order_question,
              handlers={"get_order_details": tools.get_order_details}),
        _spec(SHIPMENT_TRACKING, "shipment.track", "查询订单物流与轨迹", "order", "BUSINESS_DATA",
              _single("order", "track_shipment", order_fields), required=order_fields,
              accepted=order_fields, missing_question=order_question,
              handlers={"track_shipment": tools.track_shipment}),
        _spec(REFUND_ELIGIBILITY, "refund.eligibility.read", "判断订单是否满足退款条件", "order",
              "BUSINESS_DATA", _refund_eligibility, required=order_fields, accepted=order_fields,
              missing_question=order_question,
              handlers={"get_order_status": tools.get_order_status,
                        "check_refund_eligibility": tools.check_refund_eligibility}),
        _spec(REFUND_REQUEST, "refund.request.prepare", "核验退款条件并准备退款申请", "order",
              "BUSINESS_DATA", _refund_request, required=order_fields, accepted=order_fields,
              side_effect="REQUIRES_APPROVAL", missing_question=order_question,
              handlers={"get_order_status": tools.get_order_status,
                        "check_refund_eligibility": tools.check_refund_eligibility}),
        _spec(PAYMENT_STATUS, "payment.status.read", "查询订单支付状态", "order", "BUSINESS_DATA",
              _single("payment", "get_payment_status", order_fields), required=order_fields,
              accepted=order_fields, missing_question=order_question,
              handlers={"get_payment_status": tools.get_payment_status}),
        _spec(PAYMENT_PROCESS, "payment.process.prepare", "查询支付状态并准备付款", "order",
              "BUSINESS_DATA", _payment_process, required=order_fields, accepted=order_fields,
              side_effect="REQUIRES_APPROVAL", missing_question=order_question,
              handlers={"get_payment_status": tools.get_payment_status}),
        _spec(FAQ, "policy.answer", "根据已验证服务政策回答问题", "policy", "POLICY",
              _faq, required=("question",), accepted=("question",),
              missing_question="请说明你想了解哪项服务规则。",
              handlers={"answer_faq": tools.answer_faq}),
        _spec(PRODUCT_SEARCH, "catalog.search", "按用户明确条件搜索真实商品目录", "product",
              "CATALOG", _search, accepted=("query", "category", "max_price", "excluded_terms"),
              under_specified=True, metadata_dependencies=(CATALOG_FACETS_DEPENDENCY,),
              handlers={"search_products": tools.search_products}),
        _spec(PRODUCT_DISCOVERY, "catalog.discover", "在偏好不完整时生成真实、差异化的探索候选并继续偏好获取",
              "product", "CATALOG", _discover,
              accepted=("query", "category", "max_price", "soft_preferences", "excluded_terms"),
              ungrounded_optional_defaults={
                  "query": "", "category": None, "max_price": None,
                  "soft_preferences": [], "excluded_terms": [],
              },
              under_specified=True, metadata_dependencies=(CATALOG_FACETS_DEPENDENCY,),
              handlers={"discover_products": tools.discover_products}),
        _spec(PRODUCT_COMPARE, "catalog.compare", "比较用户明确指定的两到三款真实商品", "product",
              "CATALOG", _compare, required=("product_ids",), accepted=("product_ids",),
              min_items={"product_ids": 2}, max_items={"product_ids": 3},
              metadata_dependencies=(CATALOG_FACETS_DEPENDENCY,),
              missing_question="请提供要比较的两到三个商品编号。",
              handlers={"compare_products": tools.compare_products}),
        _spec(CATALOG_QUERY, "catalog.query", "对真实商品目录执行受限的筛选、排序、分组、投影、聚合、分页或结果集追问",
              "catalog_result", "CATALOG", _catalog_query,
              required=("catalog_query",), accepted=("catalog_query",),
              metadata_dependencies=(CATALOG_FACETS_DEPENDENCY,),
              missing_question="请说明要查询的商品范围、条件或期望的结果形式。",
              handlers={CATALOG_QUERY_OPERATION: query_catalog}),
        _spec(GENERAL_ANSWER, "knowledge.answer", "回答无需业务工具的稳定知识、解释、写作和一般咨询；需要实时外部证据时明确能力边界",
              "knowledge", "MODEL_KNOWLEDGE", _general_answer, required=("question",),
              accepted=("question",), under_specified=True,
              alternative_evidence=("REALTIME_EXTERNAL",),
              missing_question="请把你想了解的问题再具体说一下。"),
    )


_DEFAULT_REGISTRY = CapabilityRegistry(
    _builtin_registrations(),
    metadata_providers={CATALOG_FACETS_DEPENDENCY: tools.get_catalog_metadata_provider()},
)
_ACTIVE_REGISTRY: ContextVar[CapabilityRegistry | None] = ContextVar("capability_registry", default=None)


def get_capability_registry() -> CapabilityRegistry:
    return _ACTIVE_REGISTRY.get() or _DEFAULT_REGISTRY


@contextmanager
def use_capability_registry(registry: CapabilityRegistry):
    """Temporarily install a registry for an isolated run or contract test."""

    token = _ACTIVE_REGISTRY.set(registry)
    try:
        yield registry
    finally:
        _ACTIVE_REGISTRY.reset(token)

