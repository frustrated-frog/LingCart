"""Runtime capability contracts for semantic request routing.

The dialogue model describes outcomes with stable protocol fields.  Business
coverage lives in this registry, so adding a capability does not require a new
branch in the dialogue parser or orchestrator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Callable, Mapping

from src import tools
from src.errors import AgentError
from src.intake import (
    FAQ,
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
    alternative_evidence_requirements: tuple[str, ...] = ()
    supports_under_specified_goal: bool = False
    side_effect: str = "READ_ONLY"
    missing_question: str = "请补充完成这个请求所需的关键信息。"

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
            "evidence_requirements": [self.evidence_requirement, *self.alternative_evidence_requirements],
            "supports_under_specified_goal": self.supports_under_specified_goal,
            "side_effect": self.side_effect,
        }


@dataclass(frozen=True)
class CapabilityRegistration:
    spec: CapabilitySpec
    compiler: Compiler
    tools: Mapping[str, object] = field(default_factory=dict)


class CapabilityRegistry:
    """Explicit, inspectable registry used by understanding and execution."""

    def __init__(self, registrations: tuple[CapabilityRegistration, ...] = ()):
        self._registrations: dict[str, CapabilityRegistration] = {}
        self._tools: dict[str, object] = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: CapabilityRegistration) -> None:
        kind = registration.spec.objective_kind
        if not kind or kind in self._registrations:
            raise ValueError(f"Capability objective kind is missing or duplicated: {kind}")
        for name, handler in registration.tools.items():
            existing = self._tools.get(name)
            if existing is not None and existing is not handler:
                raise ValueError(f"Operation {name} has conflicting handlers")
        self._registrations[kind] = registration
        self._tools.update(registration.tools)

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
        return tuple(dict.fromkeys(missing))

    def validate_arguments(self, objective: Mapping[str, object]) -> None:
        """Enforce a capability's declared contract without parser branches."""
        spec = self.registration(str(objective["kind"])).spec
        protocol_fields = {
            "id", "kind", "source_quote", "desired_outcome", "target_type", "evidence_requirement",
        }
        supplied = {
            name for name, value in objective.items()
            if name not in protocol_fields and value not in (None, "", [])
        }
        unexpected = supplied - set(spec.accepted_fields)
        if unexpected:
            raise AgentError(
                "INVALID_CAPABILITY_ARGUMENTS",
                f"能力 {spec.objective_kind} 不接受字段：{', '.join(sorted(unexpected))}",
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

    def prompt_catalog(self) -> list[dict]:
        return [registration.spec.public_payload() for registration in self._registrations.values()]

    @property
    def version(self) -> str:
        encoded = json.dumps(self.prompt_catalog(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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
                              _args(objective, ("query", "category", "max_price"))),)


def _discover(objective: Mapping[str, object], request: str) -> tuple[CompiledOperation, ...]:
    arguments = _args(objective, ("query", "category", "max_price", "soft_preferences"))
    arguments["context_seed"] = request
    return (CompiledOperation("candidate-set", "shopping", "discover_products", arguments),)


def _compare(objective: Mapping[str, object], _request: str) -> tuple[CompiledOperation, ...]:
    return (CompiledOperation("comparison", "shopping", "compare_products",
                              {"product_ids": objective.get("product_ids", [])}),)


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
          alternative_evidence: tuple[str, ...] = (),
          under_specified: bool = False, side_effect: str = "READ_ONLY",
          missing_question: str = "请补充完成这个请求所需的关键信息。",
          handlers: Mapping[str, object] | None = None) -> CapabilityRegistration:
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
            alternative_evidence_requirements=alternative_evidence,
            supports_under_specified_goal=under_specified,
            side_effect=side_effect,
            missing_question=missing_question,
        ),
        compiler,
        handlers or {},
    )


def _builtin_registrations() -> tuple[CapabilityRegistration, ...]:
    order_fields = ("order_id",)
    order_question = "请提供要处理的订单号，例如“查询订单 1004 的状态和物流”。"
    return (
        _spec(SYSTEM_TIME, "system.clock.read", "读取当前日期、时间和星期；需要当前时间事实时必须使用",
              "runtime_clock", "RUNTIME_ENVIRONMENT", _clock, accepted=("timezone",),
              under_specified=True, handlers={"read_clock": tools.read_clock}),
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
              "CATALOG", _search, accepted=("query", "category", "max_price"),
              under_specified=True, handlers={"search_products": tools.search_products}),
        _spec(PRODUCT_DISCOVERY, "catalog.discover", "在偏好不完整时生成真实、差异化的探索候选并继续偏好获取",
              "product", "CATALOG", _discover,
              accepted=("query", "category", "max_price", "soft_preferences"),
              under_specified=True, handlers={"discover_products": tools.discover_products}),
        _spec(PRODUCT_COMPARE, "catalog.compare", "比较用户明确指定的两到三款真实商品", "product",
              "CATALOG", _compare, required=("product_ids",), accepted=("product_ids",),
              min_items={"product_ids": 2}, max_items={"product_ids": 3},
              missing_question="请提供要比较的两到三个商品编号。",
              handlers={"compare_products": tools.compare_products}),
        _spec(GENERAL_ANSWER, "knowledge.answer", "回答无需业务工具的稳定知识、解释、写作和一般咨询；需要实时外部证据时明确能力边界",
              "knowledge", "MODEL_KNOWLEDGE", _general_answer, required=("question",),
              accepted=("question",), under_specified=True,
              alternative_evidence=("REALTIME_EXTERNAL",),
              missing_question="请把你想了解的问题再具体说一下。"),
    )


_DEFAULT_REGISTRY = CapabilityRegistry(_builtin_registrations())


def get_capability_registry() -> CapabilityRegistry:
    return _DEFAULT_REGISTRY

