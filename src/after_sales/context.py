"""从业务账本解析会话内案件，模型输出不能扩大访问范围。"""
from sqlalchemy import select

from src.after_sales.service import AfterSalesCaseService
from src.backend.db import AgentRun, AfterSalesCase
from src.errors import AgentError


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
            result = [{"overflow": True, "message": "案件过多，请通过案件详情明确编号"}]
            runtime.checkpoint(case_context=result)
            return result
        service = AfterSalesCaseService(session)
        result = []
        for row in rows:
            case = service.get_case(row.case_id)
            decision = case.latest_decision or {}
            result.append({"case_id": row.case_id, "order_id": row.order_id,
                           "revision": row.revision, "status": row.status,
                           "requirements": decision.get("facts", {}).get("evidence_requirements", []),
                           "evidence": [{"evidence_id": e["evidence_id"], "evidence_type": e["evidence_type"],
                                         "status": e["status"]} for e in case.evidence]})
        runtime.checkpoint(case_context=result)
        return result


def normalize_case_continuation(objective: dict, sources: tuple[str, ...]):
    normalized = dict(objective)
    cases = current_case_context()
    quote = str(objective.get("source_quote") or "")
    requested = objective.get("case_id")
    order_id = objective.get("order_id")
    # 多案件必须由用户明确指定案件或唯一订单；仅凭模型挑选不够。
    if len(cases) > 1:
        cases = [c for c in cases if (requested and requested in quote and c.get("case_id") == requested)
                 or (order_id and str(order_id) in quote and c.get("order_id") == order_id)]
    elif requested:
        cases = [c for c in cases if c.get("case_id") == requested]
    if len(cases) != 1 or cases[0].get("overflow"):
        normalized["case_id"] = None
        normalized["case_revision"] = None
        return normalized, {}
    case = cases[0]
    if order_id and case["order_id"] != order_id:
        raise AgentError("CASE_CONTEXT_CONFLICT", "订单与案件引用不一致", retryable=False)
    requirement_id = objective.get("requirement_id")
    if requirement_id and requirement_id not in {r['requirement_id'] for r in case['requirements']}:
        raise AgentError("EVIDENCE_REQUIREMENT_STALE", "证据要求不属于当前案件政策，请重新读取", retryable=False)
    statement = objective.get("evidence_text")
    if statement and (not requirement_id or statement not in quote):
        raise AgentError("UNGROUNDED_EVIDENCE", "补充材料必须引用当前原文并指定政策证据要求", retryable=False)
    revoked = objective.get("revoke_evidence_id")
    if revoked and revoked not in {e['evidence_id'] for e in case['evidence']}:
        raise AgentError("EVIDENCE_NOT_FOUND", "撤回材料不属于当前案件", retryable=False)
    normalized.update(case_id=case['case_id'], case_revision=case['revision'])
    return normalized, {key: {"resolver": "session_case_ledger", "resolver_version": "1",
                              "normalized_value": normalized[key], "source_quote": quote}
                        for key in ("case_id", "case_revision")}
