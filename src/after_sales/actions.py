"""跨轮案件操作；调用者提供 ToolExecutor 的原子事务。"""
from datetime import UTC, datetime

from src.after_sales.contracts import EvidenceCommand
from src.after_sales.policy import AfterSalesPolicyEngine
from src.after_sales.service import AfterSalesCaseService, AfterSalesDomainError
from src.backend.db import AgentRun, Order
from src.errors import AgentError


def case_result(case):
    decision = case.latest_decision or {}
    return {"case_id": case.case_id, "order_id": case.order_id, "case_status": case.status.value,
            "revision": case.revision, "policy_decision": decision,
            "requires_confirmation": decision.get("outcome") == "ELIGIBLE" and case.status == "AWAITING_CONFIRMATION",
            "case": case.model_dump(mode="json"), "data_source": "AFTER_SALES_CASE_LEDGER"}


def continue_case(*, case_id, expected_revision, requirement_id, evidence_text, revoke_evidence_id):
    from src import tools
    from src.runtime import current_runtime

    runtime = current_runtime()
    if runtime is None:
        raise AgentError("CASE_SESSION_REQUIRED", "继续案件需要持久化会话上下文", retryable=False)
    now = datetime.now(UTC).replace(tzinfo=None)
    try:
        with tools._session() as session:
            run = session.get(AgentRun, runtime.run_id)
            service = AfterSalesCaseService(session)
            case = service.get_case(case_id)
            if run is None or case.origin_session_id != run.session_id:
                raise AgentError("CASE_CONTEXT_FORBIDDEN", "案件不属于当前会话", retryable=False)
            if case.revision != expected_revision:
                raise AgentError("CASE_REVISION_CONFLICT", "案件已变化，请重新读取后继续", retryable=False)
            if case.status.terminal or case.status == "SUBMITTED":
                raise AgentError("CASE_NOT_CONTINUABLE", "已提交或终止案件不能继续补证", retryable=False)
            evaluation = AfterSalesPolicyEngine(session).evaluate(case_id, as_of=now)
            requirements = {r["requirement_id"]: r for r in evaluation.facts["evidence_requirements"]}
            requirement = requirements.get(requirement_id)
            if requirement_id and requirement is None:
                raise AgentError("EVIDENCE_REQUIREMENT_STALE", "政策证据要求已经变化，请重新读取", retryable=False)
            if evidence_text and (not requirement or evidence_text not in run.request or not evidence_text.strip()):
                raise AgentError("UNGROUNDED_EVIDENCE", "补证内容必须来自本轮用户原文及当前政策要求", retryable=False)
            if evidence_text and (requirement['validation_condition'] != 'CUSTOMER_ATTESTATION'
                                  or 'CUSTOMER_STATEMENT' not in requirement['accepted_sources']):
                raise AgentError("EVIDENCE_SOURCE_NOT_ACCEPTED", "该要求不能用用户陈述替代权威证据", retryable=False)
            if revoke_evidence_id and requirement and evidence_text:
                old = next((e for e in case.evidence if e['evidence_id'] == revoke_evidence_id), None)
                if not old or old['evidence_type'] != requirement['evidence_type']:
                    raise AgentError("EVIDENCE_REPLACEMENT_CONFLICT", "替换材料必须属于同一证据要求", retryable=False)
            case = service.attach_run(case_id, run.run_id, session_id=run.session_id, role="EVIDENCE_UPDATE",
                                      idempotency_key=f"run:{run.run_id}", expected_revision=case.revision, now=now)
            if revoke_evidence_id:
                case = service.revoke_evidence(case_id, revoke_evidence_id, idempotency_key=f"revoke:{run.run_id}",
                                               expected_revision=case.revision, actor_id=run.session_id, now=now)
            command = None
            if evidence_text:
                command = EvidenceCommand(evidence_type=requirement['evidence_type'], source_type="CUSTOMER_STATEMENT",
                                          source_ref=f"agent_runs:{run.run_id}:request",
                                          payload={"statement": evidence_text}, observed_at=now)
            elif requirement and requirement['validation_condition'] == 'AUTHORITATIVE_ORDER' and 'ORDER_RECORD' in requirement['accepted_sources']:
                order = session.get(Order, case.order_id)
                if order:
                    command = EvidenceCommand(evidence_type=requirement['evidence_type'], source_type="ORDER_RECORD",
                                              source_ref=f"orders:{case.order_id}",
                                              payload={"order_id": case.order_id, "status": order.status}, observed_at=now)
            elif requirement_id:
                raise AgentError("EVIDENCE_TEXT_REQUIRED", "该政策要求用户补充原始陈述", retryable=False)
            if command:
                case = service.add_evidence(case_id, command, idempotency_key=f"evidence:{run.run_id}",
                                            expected_revision=case.revision, actor_id=run.session_id, now=now)
            evaluation = AfterSalesPolicyEngine(session).evaluate(case_id, as_of=now)
            case = service.record_policy_decision(case_id, evaluation, idempotency_key=f"policy:{evaluation.input_hash}",
                                                 expected_revision=case.revision)
            tools._commit(session)
            return case_result(case)
    except AfterSalesDomainError as exc:
        raise AgentError(exc.code, str(exc), retryable=exc.retryable) from exc
