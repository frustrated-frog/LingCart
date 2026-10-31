"""从业务账本解析会话内案件，模型输出不能扩大访问范围。"""
from sqlalchemy import func, select

from src.after_sales.queries import current_case_projection
from src.after_sales.service import AfterSalesCaseService
from src.backend.db import AgentRun, AfterSalesCase
from src.errors import AgentError


_WAITING_REASON_BY_ACTION = {
    "COLLECT_BUSINESS_EVIDENCE": "AUTHORITATIVE_EVIDENCE_AVAILABLE",
    "REQUEST_CUSTOMER_EVIDENCE": "CUSTOMER_EVIDENCE_REQUIRED",
    "WAIT_FOR_CUSTOMER_EVIDENCE": "CUSTOMER_EVIDENCE_DEFERRED",
    "REQUIRES_HUMAN_CAPABILITY": "CAPABILITY_UNAVAILABLE",
}


def _requirements_context(session, case, decision: dict) -> list[dict]:
    """Project only current, policy-bound obligation state into model context."""

    if not decision or decision.get("valid") is not True:
        return []
    requirements = [
        requirement
        for requirement in decision.get("facts", {}).get("evidence_requirements", [])
        if isinstance(requirement, dict)
    ]
    policy_version_id = decision.get("policy_version_id")
    policy_checksum = decision.get("policy_checksum")
    readiness_by_requirement = {}
    if policy_version_id and policy_checksum:
        from src.after_sales.readiness import project_evidence_readiness

        readiness_by_requirement = project_evidence_readiness(
            session,
            case_id=case.case_id,
            policy_version_id=policy_version_id,
            policy_checksum=policy_checksum,
            requirements=requirements,
        )
    valid_plans = [plan for plan in case.plans if isinstance(plan, dict) and plan.get("valid") is True]
    latest_plan = valid_plans[-1] if valid_plans else {}
    action_by_requirement = {
        action.get("requirement_id"): action
        for action in latest_plan.get("next_actions") or []
        if isinstance(action, dict) and action.get("requirement_id")
    }
    projected = []
    for requirement in requirements:
        requirement_id = requirement.get("requirement_id")
        readiness = readiness_by_requirement.get(requirement_id)
        action = action_by_requirement.get(requirement_id)
        item = {
            "requirement_id": requirement_id,
            "evidence_type": requirement.get("evidence_type"),
            "description": requirement.get("description"),
            "status": requirement.get("status"),
            "question": (action or {}).get("question")
            or (requirement.get("semantic_assessment") or {}).get("question"),
        }
        if readiness is not None:
            item["readiness"] = {
                "disposition": readiness.disposition,
                "source_quote": readiness.source_quote,
            }
        if action is not None:
            item["next_action"] = {
                key: action.get(key)
                for key in ("kind", "requires_approval")
                if action.get(key) is not None
            }
            item["waiting_reason"] = _WAITING_REASON_BY_ACTION.get(action.get("kind"))
        projected.append(item)
    return projected


def current_case_context() -> list[dict]:
    from src import tools
    from src.runtime import current_runtime

    runtime = current_runtime()
    if runtime is None:
        return []
    if "case_context" in runtime.snapshot:
        return runtime.snapshot["case_context"]
    with tools._session() as session:
        run = session.get(AgentRun, runtime.run_id)
        if run is None:
            return []
        rows = session.scalars(select(AfterSalesCase).where(
            AfterSalesCase.origin_session_id == run.session_id,
            AfterSalesCase.status.not_in(["RESOLVED", "REJECTED", "CANCELLED", "SUBMITTED"]),
        ).order_by(AfterSalesCase.updated_at.desc()).limit(21)).all()
        if len(rows) > 20:
            # 不截断后再把最后一个匹配误认为唯一案件。
            rows = session.scalars(select(AfterSalesCase).where(
                AfterSalesCase.origin_session_id == run.session_id,
                AfterSalesCase.status.not_in(["RESOLVED", "REJECTED", "CANCELLED", "SUBMITTED"]),
                func.instr(run.request, AfterSalesCase.case_id) > 0,
            ).limit(21)).all()
            if not rows or len(rows) > 20:
                result = [{"overflow": True, "message": "案件过多，请通过案件详情明确编号"}]
                runtime.checkpoint(case_context=result)
                return result
        service = AfterSalesCaseService(session)
        result = []
        for row in rows:
            case = current_case_projection(session, service.get_case(row.case_id))
            decision = case.latest_decision or {}
            requirements = _requirements_context(session, case, decision)
            from src.after_sales.pending_clarification import current_open_for_case
            pending_clarification = current_open_for_case(
                session,
                session_id=run.session_id,
                case_id=row.case_id,
            )
            result.append({"case_id": row.case_id, "order_id": row.order_id,
                           "revision": row.revision, "status": row.status, "reason_code": row.reason_code,
                           "policy": {
                               "policy_version_id": decision.get("policy_version_id"),
                               "policy_checksum": decision.get("policy_checksum"),
                               "policy_decision_id": decision.get("decision_id"),
                               "valid": decision.get("valid") is True,
                           },
                           "pending_clarification": pending_clarification,
                           "requirements": requirements,
                           "evidence": [{"evidence_id": e["evidence_id"], "evidence_type": e["evidence_type"],
                                         "statement_excerpt": str(e['payload'].get('statement', ''))[:512],
                                         "status": e["status"]} for e in case.evidence[-32:]]})
        runtime.checkpoint(case_context=result)
        return result


def normalize_case_continuation(objective: dict, sources: tuple[str, ...]):
    normalized = dict(objective)
    if objective.get("clarification_id"):
        raise AgentError(
            "UNTRUSTED_CLARIFICATION_BINDING",
            "澄清问题编号只能由运行时从当前案件账本绑定",
            retryable=False,
        )
    cases = current_case_context()
    quote = str(objective.get("source_quote") or "")
    requested = objective.get("case_id")
    order_id = objective.get("order_id")
    # 多案件必须由用户明确指定案件或唯一订单；仅凭模型挑选不够。
    if len(cases) > 1:
        if requested and requested in quote:
            cases = [c for c in cases if c.get("case_id") == requested]
        else:
            cases = [c for c in cases if order_id and str(order_id) in quote and c.get("order_id") == order_id]
    elif requested:
        cases = [c for c in cases if c.get("case_id") == requested]
    if len(cases) != 1 or cases[0].get("overflow"):
        normalized["case_id"] = None
        normalized["case_revision"] = None
        return normalized, {}
    case = cases[0]
    if order_id and case["order_id"] != order_id:
        raise AgentError("CASE_CONTEXT_CONFLICT", "订单与案件引用不一致", retryable=False)
    statement = objective.get("evidence_text")
    pending_clarification = case.get("pending_clarification")
    requirement_id = objective.get("requirement_id")
    if (
        statement
        and isinstance(pending_clarification, dict)
        and requirement_id in {None, pending_clarification.get("requirement_id")}
    ):
        requirement_id = pending_clarification["requirement_id"]
        normalized["requirement_id"] = requirement_id
        normalized["clarification_id"] = pending_clarification["clarification_id"]
    if requirement_id and requirement_id not in {r['requirement_id'] for r in case['requirements']}:
        raise AgentError("EVIDENCE_REQUIREMENT_STALE", "证据要求不属于当前案件政策，请重新读取", retryable=False)
    readiness_types = {
        row.get("evidence_type")
        for row in objective.get("evidence_readiness_claims") or []
        if isinstance(row, dict)
    }
    known_types = {r.get("evidence_type") for r in case['requirements']}
    if readiness_types - known_types:
        raise AgentError("EVIDENCE_READINESS_NOT_APPLICABLE", "证据就绪度不属于当前案件政策要求", retryable=False)
    if statement and (not requirement_id or statement not in quote):
        raise AgentError("UNGROUNDED_EVIDENCE", "补充材料必须引用当前原文并指定政策证据要求", retryable=False)
    revoked = objective.get("revoke_evidence_id")
    if revoked and revoked not in {e['evidence_id'] for e in case['evidence']}:
        raise AgentError("EVIDENCE_NOT_FOUND", "撤回材料不属于当前案件", retryable=False)
    updates = objective.get("evidence_updates") or []
    requirements = {r['requirement_id']: r for r in case['requirements']}
    for update in updates:
        if update['requirement_id'] not in requirements or update['source_quote'] not in quote:
            raise AgentError('UNGROUNDED_EVIDENCE', '材料更新必须引用本轮原文及当前案件要求', retryable=False)
    normalized.update(case_id=case['case_id'], case_revision=case['revision'])
    provenance = {
        key: {"resolver": "session_case_ledger", "resolver_version": "2",
              "normalized_value": normalized[key], "source_quote": quote}
        for key in ("case_id", "case_revision")
    }
    if normalized.get("clarification_id"):
        for key in ("requirement_id", "clarification_id"):
            provenance[key] = {
                "resolver": "pending_clarification_ledger",
                "resolver_version": "1",
                "normalized_value": normalized[key],
                "source_quote": quote,
            }
    return normalized, provenance
