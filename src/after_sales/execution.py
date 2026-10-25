"""政策感知自主执行：投影案件状态，只选择政策允许的确定性下一步。"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from src.after_sales.contracts import EvidenceReadinessState
from src.after_sales.service import _checksum


STATE_PROJECTOR_VERSION = "policy-aware-case-state-v2"
OBLIGATION_VERIFIER_VERSION = "policy-obligation-verifier-v2"
ACTION_PLANNER_VERSION = "policy-aware-next-action-v2"
AUTONOMOUS_OPERATION = "continue_after_sales_case"


class PolicyObligation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: str
    evidence_type: str
    description: str
    status: str
    validation_condition: str
    accepted_sources: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    readiness: EvidenceReadinessState | None = None


class CaseExecutionState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    projector_version: Literal["policy-aware-case-state-v2"] = STATE_PROJECTOR_VERSION
    state_id: str
    status: Literal["READY", "STALE", "TERMINAL"]
    reason_code: str
    case_id: str
    case_revision: int = Field(ge=0)
    case_status: str
    order_id: int | None = None
    policy_version_id: str | None = None
    policy_checksum: str | None = None
    policy_decision_id: str | None = None
    policy_outcome: str | None = None
    plan_id: str | None = None
    plan_revision: int | None = None
    obligations: tuple[PolicyObligation, ...] = ()
    next_actions: tuple[dict[str, Any], ...] = ()
    authority_refs: tuple[str, ...] = ()


class CaseActionDirective(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    planner_version: Literal["policy-aware-next-action-v2"] = ACTION_PLANNER_VERSION
    verifier_version: Literal["policy-obligation-verifier-v2"] = OBLIGATION_VERIFIER_VERSION
    directive_id: str
    state_id: str
    status: Literal["EXECUTE", "WAIT_USER_INPUT", "WAIT_APPROVAL", "BLOCKED", "COMPLETE", "ABSTAINED"]
    reason_code: str
    action_kind: str | None = None
    goal_id: str | None = None
    requirement_id: str | None = None
    operation: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    question: str | None = None


def _record(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def project_case_execution_state(result: Mapping[str, Any]) -> CaseExecutionState:
    """Project one immutable planning state from persisted case, policy and plan identities."""
    case = _record(result.get("case"))
    decision = _record(result.get("policy_decision") or case.get("latest_decision"))
    plans = [row for row in case.get("plans") or [] if isinstance(row, dict)]
    plan = plans[-1] if plans else {}
    case_id = str(result.get("case_id") or case.get("case_id") or "")
    case_revision = result.get("revision") if type(result.get("revision")) is int else case.get("revision")
    case_revision = case_revision if type(case_revision) is int and case_revision >= 0 else 0
    case_status = str(result.get("case_status") or case.get("status") or "UNKNOWN")
    goals_by_requirement = {
        str(row.get("requirement_id")): row
        for row in plan.get("goals") or []
        if isinstance(row, dict) and row.get("requirement_id")
    }
    obligations = tuple(
        PolicyObligation(
            requirement_id=str(row.get("requirement_id") or ""),
            evidence_type=str(row.get("evidence_type") or ""),
            description=str(row.get("description") or ""),
            status=str(row.get("status") or "UNKNOWN"),
            validation_condition=str(row.get("validation_condition") or "UNSUPPORTED"),
            accepted_sources=tuple(str(item) for item in row.get("accepted_sources") or []),
            evidence_refs=tuple(str(item) for item in row.get("evidence_refs") or []),
            readiness=goals_by_requirement.get(str(row.get("requirement_id") or ""), {}).get("readiness"),
        )
        for row in _record(decision.get("facts")).get("evidence_requirements") or []
        if isinstance(row, dict) and row.get("requirement_id")
    )
    terminal = case_status in {"SUBMITTED", "RESOLVED", "REJECTED", "CANCELLED"}
    identity_valid = bool(
        case_id
        and result.get("data_source") == "AFTER_SALES_CASE_LEDGER"
        and case.get("case_id") == case_id
        and case.get("revision") == case_revision
        and case.get("status") == case_status
        and decision.get("decision_id")
        and decision.get("valid") is not False
        and plan.get("valid") is True
        and plan.get("case_revision") == case_revision
        and plan.get("decision_id") == decision.get("decision_id")
        and plan.get("policy_version_id") == decision.get("policy_version_id")
        and plan.get("status") == case_status
    )
    status = "TERMINAL" if terminal else "READY" if identity_valid else "STALE"
    reason = "CASE_TERMINAL" if terminal else "BOUND_IDENTITIES_VERIFIED" if identity_valid else "IDENTITY_OR_PLAN_STALE"
    basis = {
        "projector_version": STATE_PROJECTOR_VERSION,
        "case_id": case_id,
        "case_revision": case_revision,
        "case_status": case_status,
        "policy_decision_id": decision.get("decision_id"),
        "policy_checksum": decision.get("policy_checksum"),
        "plan_id": plan.get("plan_id"),
        "plan_revision": plan.get("revision"),
        "obligations": [row.model_dump(mode="json") for row in obligations],
        "next_actions": deepcopy(plan.get("next_actions") or []),
    }
    return CaseExecutionState(
        state_id=_checksum(basis),
        status=status,
        reason_code=reason,
        case_id=case_id,
        case_revision=case_revision,
        case_status=case_status,
        order_id=result.get("order_id") if type(result.get("order_id")) is int else case.get("order_id"),
        policy_version_id=decision.get("policy_version_id"),
        policy_checksum=decision.get("policy_checksum"),
        policy_decision_id=decision.get("decision_id"),
        policy_outcome=decision.get("outcome"),
        plan_id=plan.get("plan_id"),
        plan_revision=plan.get("revision") if type(plan.get("revision")) is int else None,
        obligations=obligations,
        next_actions=tuple(deepcopy(row) for row in plan.get("next_actions") or [] if isinstance(row, dict)),
        authority_refs=tuple(str(item) for item in decision.get("evidence_refs") or []),
    )


def _directive(state: CaseExecutionState, *, status: str, reason_code: str,
               action: Mapping[str, Any] | None = None, operation: str | None = None,
               arguments: dict[str, Any] | None = None, requires_approval: bool = False,
               question: str | None = None) -> CaseActionDirective:
    action = action or {}
    payload = {
        "state_id": state.state_id,
        "status": status,
        "reason_code": reason_code,
        "action_kind": action.get("kind"),
        "goal_id": action.get("goal_id"),
        "requirement_id": action.get("requirement_id"),
        "operation": operation,
        "arguments": arguments or {},
        "requires_approval": requires_approval,
        "question": question,
    }
    return CaseActionDirective(directive_id=_checksum(payload), **payload)


def decide_next_case_action(state: CaseExecutionState) -> CaseActionDirective:
    """Choose one bounded action. Only an authoritative read can be auto-executed in V1."""
    if state.status == "TERMINAL":
        return _directive(state, status="COMPLETE", reason_code="CASE_TERMINAL")
    if state.status != "READY":
        return _directive(state, status="ABSTAINED", reason_code="CASE_STATE_NOT_EXECUTABLE")

    actions = list(state.next_actions)
    automatic = sorted(
        (row for row in actions if row.get("kind") == "COLLECT_BUSINESS_EVIDENCE"),
        key=lambda row: (str(row.get("goal_id") or ""), str(row.get("requirement_id") or "")),
    )
    if automatic:
        action = automatic[0]
        requirements = {row.requirement_id: row for row in state.obligations}
        requirement = requirements.get(str(action.get("requirement_id") or ""))
        valid = bool(
            requirement
            and requirement.status != "SATISFIED"
            and requirement.validation_condition == "AUTHORITATIVE_ORDER"
            and "ORDER_RECORD" in requirement.accepted_sources
            and action.get("operation") == AUTONOMOUS_OPERATION
            and not action.get("requires_approval")
        )
        if not valid:
            return _directive(
                state,
                status="ABSTAINED",
                reason_code="POLICY_OBLIGATION_NOT_AUTOMATABLE",
                action=action,
            )
        arguments = {
            "case_id": state.case_id,
            "expected_revision": state.case_revision,
            "requirement_id": requirement.requirement_id,
        }
        return _directive(
            state,
            status="EXECUTE",
            reason_code="AUTHORITATIVE_EVIDENCE_AVAILABLE",
            action=action,
            operation=AUTONOMOUS_OPERATION,
            arguments=arguments,
        )

    customer = next((row for row in actions if row.get("kind") == "REQUEST_CUSTOMER_EVIDENCE"), None)
    if customer:
        return _directive(
            state,
            status="WAIT_USER_INPUT",
            reason_code="CUSTOMER_EVIDENCE_REQUIRED",
            action=customer,
            question=str(customer.get("question") or "请补充政策要求的材料。"),
        )
    deferred = next((row for row in actions if row.get("kind") == "WAIT_FOR_CUSTOMER_EVIDENCE"), None)
    if deferred:
        requirements = {row.requirement_id: row for row in state.obligations}
        requirement = requirements.get(str(deferred.get("requirement_id") or ""))
        readiness = requirement.readiness if requirement else None
        if (
            requirement is None
            or requirement.status == "SATISFIED"
            or readiness is None
            or readiness.disposition != "DEFERRED"
            or readiness.requirement_id != requirement.requirement_id
            or readiness.evidence_type != requirement.evidence_type
            or readiness.policy_version_id != state.policy_version_id
            or readiness.policy_checksum != state.policy_checksum
            or deferred.get("claim_id") != readiness.claim_id
        ):
            return _directive(
                state,
                status="ABSTAINED",
                reason_code="EVIDENCE_READINESS_BINDING_INVALID",
                action=deferred,
            )
        return _directive(
            state,
            status="WAIT_USER_INPUT",
            reason_code="CUSTOMER_EVIDENCE_DEFERRED",
            action=deferred,
            question=str(deferred.get("question") or "案件已保留；准备好后请补充政策要求的材料。"),
        )
    approval = next((row for row in actions if row.get("kind") == "REQUEST_CUSTOMER_CONFIRMATION"), None)
    if approval:
        if approval.get("operation") != "submit_after_sales_case" or approval.get("requires_approval") is not True:
            return _directive(state, status="ABSTAINED", reason_code="APPROVAL_ACTION_INVALID", action=approval)
        return _directive(
            state,
            status="WAIT_APPROVAL",
            reason_code="EXPLICIT_APPROVAL_REQUIRED",
            action=approval,
            requires_approval=True,
        )
    unsupported = next((row for row in actions if row.get("kind") == "REQUIRES_HUMAN_CAPABILITY"), None)
    if unsupported:
        return _directive(
            state,
            status="BLOCKED",
            reason_code="UNSUPPORTED_CAPABILITY",
            action=unsupported,
        )
    if not actions and state.policy_outcome in {"INELIGIBLE", "ELIGIBLE"}:
        return _directive(state, status="COMPLETE", reason_code="NO_FURTHER_AUTONOMOUS_ACTION")
    return _directive(state, status="ABSTAINED", reason_code="NO_SUPPORTED_NEXT_ACTION")
