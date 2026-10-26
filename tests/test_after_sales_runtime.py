"""售后 Capability 与 Stable Kernel 的审批、恢复和审计回归。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import select

import api
from src import orchestrator, tools
from src.backend.db import (
    AfterSalesCase,
    AfterSalesCaseRefund,
    AfterSalesCaseRun,
    Order,
    OrderItem,
    PendingAction,
    Payment,
    PolicyVersion,
    Refund,
    ShipmentEvent,
    ToolLedger,
    get_engine,
    get_session_factory,
)
from src.backend.seed import DEFAULT_AFTER_SALES_POLICY_VERSION_ID, seed
from src.after_sales.service import AfterSalesCaseService, _checksum
from src.durable import resume_run, run_conversation
from src.errors import AgentError
from src.persistence import RunStore
from src.runtime import ReplayPolicy, ToolEffect, ToolRiskLevel, tool_spec


MESSAGE = "订单 1001 的耳机出现间歇性异响，需要建立售后案件处理"
MULTI_ITEM_MESSAGE = "订单 1001 的行项目 OI-1001-1 出现异响，需要申请售后退款"
EVIDENCE_MISSING_MESSAGE = "订单 1001 的耳机与描述不符，请建立售后案件；原包装情况稍后才能确认。"
EVIDENCE_UNKNOWN_MESSAGE = "订单 1001 的耳机与描述不符，请建立售后案件；原包装情况目前不确定。"
EVIDENCE_DEFERRED_FOLLOWUP = "原包装情况稍后才能确认"
READINESS_CONTEXT_ACK = "好的，明白了"


class AfterSalesUnderstandingModel:
    def json_mode(self):
        return self

    def invoke(self, prompt):
        assert MESSAGE in str(prompt)
        return AIMessage(
            content=json.dumps(
                {
                    "reason": "需要建立可审计的售后案件并按政策核验",
                    "commands": [],
                    "objectives": [
                        {
                            "id": "after-sales-case",
                            "kind": "AFTER_SALES_CASE",
                            "source_quote": MESSAGE,
                            "order_id": 1001,
                            "reason_code": "NOT_AS_DESCRIBED",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )


class MissingReasonUnderstandingModel(AfterSalesUnderstandingModel):
    def invoke(self, prompt):
        assert MESSAGE in str(prompt)
        return AIMessage(
            content=json.dumps(
                {
                    "reason": "原因分类不明确",
                    "commands": [],
                    "objectives": [
                        {
                            "id": "after-sales-case",
                            "kind": "AFTER_SALES_CASE",
                            "source_quote": MESSAGE,
                            "order_id": 1001,
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )


class MultiItemAfterSalesUnderstandingModel(AfterSalesUnderstandingModel):
    def invoke(self, prompt):
        assert MULTI_ITEM_MESSAGE in str(prompt)
        return AIMessage(
            content=json.dumps(
                {
                    "reason": "明确指定了多商品订单中的售后行项目",
                    "commands": [],
                    "objectives": [
                        {
                            "id": "after-sales-case",
                            "kind": "AFTER_SALES_CASE",
                            "source_quote": MULTI_ITEM_MESSAGE,
                            "order_id": 1001,
                            "order_item_id": "OI-1001-1",
                            "reason_code": "NOT_AS_DESCRIBED",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )


class EvidenceBoundaryUnderstandingModel(AfterSalesUnderstandingModel):
    def invoke(self, prompt):
        prompt = str(prompt)
        assert EVIDENCE_MISSING_MESSAGE in prompt
        assert "Known evidence absence is not missing case-intake information" in prompt
        return AIMessage(
            content=json.dumps(
                {
                    "reason": "案件受理信息完整，原包装陈述属于后续政策证据",
                    "commands": [],
                    "objectives": [
                        {
                            "id": "after-sales-case",
                            "kind": "AFTER_SALES_CASE",
                            "source_quote": "订单 1001 的耳机与描述不符，请建立售后案件",
                            "evidence_context_quotes": ["原包装情况稍后才能确认"],
                            "order_id": 1001,
                            "reason_code": "NOT_AS_DESCRIBED",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )


class EvidenceReadinessUnderstandingModel(AfterSalesUnderstandingModel):
    def invoke(self, prompt):
        prompt = str(prompt)
        assert EVIDENCE_MISSING_MESSAGE in prompt
        assert "Evidence-readiness ledger for this Harness Profile" in prompt
        assert "after-sales-evidence-readiness/2026-01" in prompt
        return AIMessage(
            content=json.dumps(
                {
                    "reason": "案件受理信息完整，用户明确把原包装说明延期",
                    "commands": [],
                    "objectives": [
                        {
                            "id": "after-sales-case",
                            "kind": "AFTER_SALES_CASE",
                            "source_quote": "订单 1001 的耳机与描述不符，请建立售后案件",
                            "evidence_readiness_claims": [
                                {
                                    "evidence_type": "ORIGINAL_PACKAGING",
                                    "disposition": "DEFERRED",
                                    "source_quote": "原包装情况稍后才能确认",
                                }
                            ],
                            "order_id": 1001,
                            "reason_code": "NOT_AS_DESCRIBED",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )


class EvidenceReadinessUnknownUnderstandingModel(AfterSalesUnderstandingModel):
    def invoke(self, prompt):
        prompt = str(prompt)
        assert EVIDENCE_UNKNOWN_MESSAGE in prompt
        assert "Known evidence absence is not missing case-intake information" in prompt
        assert "Evidence-readiness ledger for this Harness Profile" in prompt
        return AIMessage(
            content=json.dumps(
                {
                    "reason": "案件受理信息完整，但用户只表达了原包装状态未知",
                    "commands": [],
                    "objectives": [
                        {
                            "id": "after-sales-case",
                            "kind": "AFTER_SALES_CASE",
                            "source_quote": "订单 1001 的耳机与描述不符，请建立售后案件",
                            "evidence_context_quotes": ["原包装情况目前不确定"],
                            "order_id": 1001,
                            "reason_code": "NOT_AS_DESCRIBED",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )


class ExistingCaseReadinessUnderstandingModel(AfterSalesUnderstandingModel):
    def invoke(self, prompt):
        prompt = str(prompt)
        assert EVIDENCE_DEFERRED_FOLLOWUP in prompt
        assert "Evidence-readiness ledger for this Harness Profile" in prompt
        assert '"session_cases":[{' in prompt
        return AIMessage(
            content=json.dumps(
                {
                    "reason": "现有案件的政策材料被用户明确延期",
                    "commands": [],
                    "objectives": [
                        {
                            "id": "after-sales-readiness",
                            "kind": "AFTER_SALES_CONTINUE",
                            "source_quote": EVIDENCE_DEFERRED_FOLLOWUP,
                            "evidence_readiness_claims": [
                                {
                                    "evidence_type": "ORIGINAL_PACKAGING",
                                    "disposition": "DEFERRED",
                                    "source_quote": EVIDENCE_DEFERRED_FOLLOWUP,
                                }
                            ],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        )


class ReadinessContextInspectionModel(AfterSalesUnderstandingModel):
    def invoke(self, prompt):
        payload = json.loads(str(prompt).split("\nConversation input:\n")[-1])
        assert payload["latest_human_message"] == READINESS_CONTEXT_ACK
        case = payload["session_cases"][0]
        requirement = case["requirements"][0]
        assert case["policy"]["valid"] is True
        assert requirement["readiness"]["disposition"] == "DEFERRED"
        assert requirement["readiness"]["source_quote"] == EVIDENCE_DEFERRED_FOLLOWUP
        assert requirement["next_action"]["kind"] == "WAIT_FOR_CUSTOMER_EVIDENCE"
        assert requirement["waiting_reason"] == "CUSTOMER_EVIDENCE_DEFERRED"
        return AIMessage(content=json.dumps({
            "reason": "用户知晓案件正在等待已延期的证据",
            "commands": [{
                "kind": "ACKNOWLEDGEMENT",
                "source_quote": READINESS_CONTEXT_ACK,
                "message": None,
            }],
            "objectives": [],
        }, ensure_ascii=False))


class SemanticModelMustNotRun:
    def json_mode(self):
        return self

    def invoke(self, _prompt):
        raise AssertionError("权威业务证据收集不应调用语义模型")


def require_authoritative_order_evidence(session):
    policy = session.get(PolicyVersion, DEFAULT_AFTER_SALES_POLICY_VERSION_ID)
    policy.rules = {
        **policy.rules,
        "required_evidence_by_reason": {
            **policy.rules["required_evidence_by_reason"],
            "NOT_AS_DESCRIBED": ["ORDER_FACT"],
        },
        "evidence_acceptance": {
            **policy.rules["evidence_acceptance"],
            "ORDER_FACT": {
                "accepted_sources": ["ORDER_RECORD"],
                "condition": "AUTHORITATIVE_ORDER",
                "description": "需要读取权威订单状态",
            },
        },
    }
    policy.checksum = _checksum(policy.rules)
    session.commit()


def require_original_packaging_evidence(session):
    policy = session.get(PolicyVersion, DEFAULT_AFTER_SALES_POLICY_VERSION_ID)
    policy.rules = {
        **policy.rules,
        "required_evidence_by_reason": {
            **policy.rules["required_evidence_by_reason"],
            "NOT_AS_DESCRIBED": ["ORIGINAL_PACKAGING"],
        },
        "evidence_acceptance": {
            **policy.rules["evidence_acceptance"],
            "ORIGINAL_PACKAGING": {
                "accepted_sources": ["CUSTOMER_STATEMENT"],
                "condition": "CUSTOMER_ATTESTATION",
                "description": "需要确认原包装及附件是否完整",
            },
        },
    }
    policy.checksum = _checksum(policy.rules)
    session.commit()


def require_mixed_authoritative_and_customer_evidence(session):
    policy = session.get(PolicyVersion, DEFAULT_AFTER_SALES_POLICY_VERSION_ID)
    policy.rules = {
        **policy.rules,
        "required_evidence_by_reason": {
            **policy.rules["required_evidence_by_reason"],
            "NOT_AS_DESCRIBED": ["ORDER_FACT", "ORIGINAL_PACKAGING"],
        },
        "evidence_acceptance": {
            **policy.rules["evidence_acceptance"],
            "ORDER_FACT": {
                "accepted_sources": ["ORDER_RECORD"],
                "condition": "AUTHORITATIVE_ORDER",
                "description": "需要读取权威订单状态",
            },
            "ORIGINAL_PACKAGING": {
                "accepted_sources": ["CUSTOMER_STATEMENT"],
                "condition": "CUSTOMER_ATTESTATION",
                "description": "需要确认原包装及附件是否完整",
            },
        },
    }
    policy.checksum = _checksum(policy.rules)
    session.commit()


@pytest.fixture()
def database(tmp_path, monkeypatch):
    path = tmp_path / "after-sales-runtime.db"
    engine = get_engine(f"sqlite:///{path}")
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
        delivered = session.scalar(
            select(ShipmentEvent).where(
                ShipmentEvent.order_id == 1001,
                ShipmentEvent.description.contains("签收"),
            )
        )
        delivered.occurred_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=2)
        session.commit()
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: AfterSalesUnderstandingModel())
    from src import llm
    from tests.evidence_fixtures import SemanticModel
    monkeypatch.setattr(llm, "get_llm", lambda **_kwargs: SemanticModel())
    yield path
    engine.dispose()


def test_after_sales_tools_declare_distinct_write_and_approval_policies(database):
    prepare = tool_spec("prepare_after_sales_case", tools.prepare_after_sales_case)
    readiness = tool_spec(
        "record_after_sales_evidence_readiness",
        tools.record_after_sales_evidence_readiness,
    )
    submit = tool_spec("submit_after_sales_case", tools.submit_after_sales_case)

    assert prepare.effect == ToolEffect.WRITE
    assert prepare.risk_level == ToolRiskLevel.GUARDED
    assert prepare.replay_policy == ReplayPolicy.IDEMPOTENT_WRITE
    assert prepare.transactional is True
    assert readiness.effect == ToolEffect.WRITE
    assert readiness.risk_level == ToolRiskLevel.GUARDED
    assert readiness.replay_policy == ReplayPolicy.IDEMPOTENT_WRITE
    assert readiness.transactional is True
    assert readiness.preflight is None
    assert submit.replay_policy == ReplayPolicy.REQUIRES_APPROVAL
    assert submit.approval_action == "SUBMIT_AFTER_SALES_CASE"
    assert submit.transactional is True


def test_missing_reason_classification_clarifies_without_opening_case(database, monkeypatch):
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: MissingReasonUnderstandingModel())

    result = run_conversation(MESSAGE, "after-sales-missing-reason", request_id="missing-reason")

    assert result["status"] == "WAITING_CLARIFICATION"
    assert result["terminal_reason"] == "MISSING_REQUIRED_INFORMATION"
    assert "售后原因" in result["reply"]
    assert result["trace"] == []
    with tools._session() as session:
        assert session.scalars(select(AfterSalesCase)).all() == []


def test_case_assessment_and_confirmed_submission_are_one_auditable_chain(database):
    first = run_conversation(MESSAGE, "after-sales", request_id="open-case")

    assert first["status"] == "WAITING_APPROVAL"
    assert [entry["tool"] for entry in first["trace"]] == ["prepare_after_sales_case"]
    assert first["trace"][0]["args"]["reason_code"] == "NOT_AS_DESCRIBED"
    assert first["trace"][0]["args"]["reason_taxonomy_version"] == "after-sales-reason/2026-01"
    assert len(first["trace"][0]["args"]["reason_taxonomy_fingerprint"]) == 64
    pending = first["pending_action"]
    assert pending["type"] == "SUBMIT_AFTER_SALES_CASE"
    assert pending["tool"] == "submit_after_sales_case"
    assert pending["requires_approval"] is True
    assert len(pending["action_fingerprint"]) == 64
    with tools._session() as session:
        case = session.get(AfterSalesCase, pending["case_id"])
        refund = session.scalar(select(Refund).where(Refund.order_id == 1001))
        assert case.status == "AWAITING_CONFIRMATION"
        assert case.reason_code == "NOT_AS_DESCRIBED"
        assert case.context["reason_classification"]["version"] == "after-sales-reason/2026-01"
        assert refund.status == "none"
        assert len(session.scalars(select(AfterSalesCaseRun).where(
            AfterSalesCaseRun.case_id == case.case_id
        )).all()) == 1

    completed = run_conversation(
        "确认退款",
        "after-sales",
        request_id="submit-case",
        action_id=pending["action_id"],
    )

    assert completed["status"] == "SUCCEEDED"
    assert completed["pending_action"] is None
    assert [entry["tool"] for entry in completed["trace"]] == ["submit_after_sales_case"]
    with tools._session() as session:
        case = session.get(AfterSalesCase, pending["case_id"])
        refund = session.scalar(select(Refund).where(Refund.order_id == 1001))
        link = session.scalar(select(AfterSalesCaseRefund).where(
            AfterSalesCaseRefund.case_id == case.case_id
        ))
        assert case.status == "SUBMITTED"
        assert refund.status == "requested"
        assert link.refund_id == refund.refund_id
        assert link.policy_decision_id == pending["policy_decision_id"]
        assert link.submitted_run_id == completed["run_id"]
        assert len(session.scalars(select(ToolLedger).where(
            ToolLedger.run_id == completed["run_id"],
            ToolLedger.tool_name == "submit_after_sales_case",
        )).all()) == 1

    repeated = run_conversation(
        "确认退款",
        "after-sales",
        request_id="submit-case",
        action_id=pending["action_id"],
    )
    assert repeated["cached"] is True
    with pytest.raises(AgentError) as used:
        run_conversation(
            "确认退款",
            "after-sales",
            request_id="submit-case-again",
            action_id=pending["action_id"],
        )
    assert used.value.code == "APPROVAL_STALE"

    observed_again = run_conversation(
        MESSAGE,
        "after-sales",
        request_id="observe-submitted-case",
    )
    assert observed_again["status"] == "SUCCEEDED"
    assert observed_again["pending_action"] is None
    assert observed_again["trace"][0]["result"]["case_id"] == pending["case_id"]
    assert observed_again["trace"][0]["result"]["case_status"] == "SUBMITTED"
    with tools._session() as session:
        assert len(session.scalars(select(AfterSalesCase)).all()) == 1


def test_policy_aware_loop_collects_authoritative_evidence_before_requesting_approval(database, monkeypatch):
    from src import llm

    with tools._session() as session:
        require_authoritative_order_evidence(session)
    monkeypatch.setattr(llm, "get_llm", lambda **_kwargs: SemanticModelMustNotRun())

    result = run_conversation(MESSAGE, "autonomous-after-sales", request_id="open-case")

    assert result["status"] == "WAITING_APPROVAL"
    assert [entry["tool"] for entry in result["trace"]] == [
        "prepare_after_sales_case",
        "continue_after_sales_case",
    ]
    event_types = [event["type"] for event in result["events"]]
    assert event_types.count("case_state_projected") == 2
    assert "case_autonomous_action_selected" in event_types
    assert "case_autonomous_action_completed" in event_types
    assert "case_execution_paused" in event_types
    autonomous = result["plan"]["steps"][0]["result"]["autonomous_execution"]
    assert autonomous["status"] == "WAIT_APPROVAL"
    assert autonomous["reason_code"] == "EXPLICIT_APPROVAL_REQUIRED"
    assert autonomous["executed_actions"][0]["operation"] == "continue_after_sales_case"
    with tools._session() as session:
        case = AfterSalesCaseService(session).get_case(result["pending_action"]["case_id"])
        evidence_types = {row["evidence_type"] for row in case.evidence}
        refund = session.scalar(select(Refund).where(Refund.order_id == 1001))
        continuation_rows = session.scalars(select(ToolLedger).where(
            ToolLedger.run_id == result["run_id"],
            ToolLedger.tool_name == "continue_after_sales_case",
        )).all()
        assert case.status == "AWAITING_CONFIRMATION"
        assert "ORDER_FACT" in evidence_types
        assert refund.status == "none"
        assert len(continuation_rows) == 1


def test_obligation_graph_profile_solves_all_candidates_before_asking_user(database, monkeypatch):
    from src import llm
    from src.harness import get_runner

    with tools._session() as session:
        require_mixed_authoritative_and_customer_evidence(session)
    monkeypatch.setattr(llm, "get_llm", lambda **_kwargs: SemanticModelMustNotRun())

    result = run_conversation(
        MESSAGE,
        "obligation-graph-after-sales",
        request_id="solve-mixed-evidence",
        runner=get_runner("after-sales-obligation-graph-v4"),
    )

    assert result["status"] == "PARTIAL"
    assert result["terminal_reason"] == "EVIDENCE_UNAVAILABLE"
    assert result["pending_action"] is None
    assert [entry["tool"] for entry in result["trace"]] == [
        "prepare_after_sales_case",
        "continue_after_sales_case",
    ]
    graph_events = [
        event for event in result["events"]
        if event["type"] == "case_obligation_graph_solved"
    ]
    assert len(graph_events) == 2
    first_candidates = {
        candidate["candidate_id"]: candidate
        for candidate in graph_events[0]["graph"]["candidates"]
    }
    first_selection = graph_events[0]["selection"]
    assert {candidate["mode"] for candidate in first_candidates.values()} == {
        "AUTONOMOUS",
        "ASK_USER",
    }
    assert first_candidates[first_selection["selected_candidate_id"]]["mode"] == "AUTONOMOUS"
    assert first_selection["reason_code"] == "TRUSTED_AUTONOMOUS_ACTION_DOMINATES"
    second_candidates = graph_events[1]["graph"]["candidates"]
    assert [candidate["mode"] for candidate in second_candidates] == ["ASK_USER"]
    assert graph_events[1]["selection"]["reason_code"] == "HIGHEST_INFORMATION_GAIN_QUESTION"
    autonomous = result["plan"]["steps"][0]["result"]["autonomous_execution"]
    assert autonomous["planner_version"] == "policy-aware-obligation-planner-v1"
    assert autonomous["action_selection_mode"] == "obligation_graph"
    assert autonomous["obligation_graph_ids"] == [event["graph"]["graph_id"] for event in graph_events]
    assert autonomous["status"] == "WAIT_USER_INPUT"
    assert autonomous["reason_code"] == "CUSTOMER_EVIDENCE_REQUIRED"
    with tools._session() as session:
        case = AfterSalesCaseService(session).get_case(result["trace"][0]["result"]["case_id"])
        assert "ORDER_FACT" in {item["evidence_type"] for item in case.evidence}
        assert session.scalar(select(Refund).where(Refund.order_id == 1001)).status == "none"


def test_observer_profile_persists_plan_without_consuming_authoritative_action(database):
    from src.harness import get_runner

    with tools._session() as session:
        require_authoritative_order_evidence(session)

    result = run_conversation(
        MESSAGE,
        "observer-after-sales",
        request_id="open-case",
        runner=get_runner("after-sales-plan-observer-v0"),
    )

    assert [entry["tool"] for entry in result["trace"]] == ["prepare_after_sales_case"]
    assert result["pending_action"] is None
    assert result["plan"]["steps"][0]["result"]["policy_decision"]["outcome"] == "NEEDS_EVIDENCE"
    assert not any(event["type"].startswith("case_autonomous_") for event in result["events"])
    assert any(event["type"] == "mechanism_applied"
               and event["mechanism_id"] == "case-plan-observer" for event in result["events"])
    with tools._session() as session:
        refund = session.scalar(select(Refund).where(Refund.order_id == 1001))
        assert refund.status == "none"


def test_evidence_boundary_profile_opens_case_before_waiting_for_customer_statement(database, monkeypatch):
    from src.harness import get_runner

    with tools._session() as session:
        require_original_packaging_evidence(session)
    monkeypatch.setattr(
        orchestrator,
        "get_llm",
        lambda **_kwargs: EvidenceBoundaryUnderstandingModel(),
    )

    result = run_conversation(
        EVIDENCE_MISSING_MESSAGE,
        "evidence-boundary-case",
        request_id="open-case-before-evidence",
        runner=get_runner("after-sales-evidence-boundary-v2"),
    )

    assert result["status"] == "PARTIAL"
    assert result["terminal_reason"] == "EVIDENCE_UNAVAILABLE"
    assert result["pending_action"] is None
    assert [entry["tool"] for entry in result["trace"]] == ["prepare_after_sales_case"]
    observation = result["trace"][0]["result"]
    assert observation["case_status"] == "AWAITING_EVIDENCE"
    assert observation["policy_decision"]["outcome"] == "NEEDS_EVIDENCE"
    assert result["plan"]["steps"][0]["result"]["autonomous_execution"]["status"] == "WAIT_USER_INPUT"
    assert any(event["type"] == "case_execution_paused" and event["status"] == "WAIT_USER_INPUT"
               for event in result["events"])
    with tools._session() as session:
        case = AfterSalesCaseService(session).get_case(observation["case_id"])
        refund = session.scalar(select(Refund).where(Refund.order_id == 1001))
        continuation = session.scalars(select(ToolLedger).where(
            ToolLedger.run_id == result["run_id"],
            ToolLedger.tool_name == "continue_after_sales_case",
        )).all()
        assert case.status == "AWAITING_EVIDENCE"
        assert {item["evidence_type"] for item in case.evidence} == {"ISSUE_DESCRIPTION"}
        assert refund.status == "none"
        assert continuation == []


def test_evidence_readiness_profile_persists_deferral_without_treating_it_as_evidence(database, monkeypatch):
    from src.harness import get_runner

    with tools._session() as session:
        require_original_packaging_evidence(session)
    monkeypatch.setattr(
        orchestrator,
        "get_llm",
        lambda **_kwargs: EvidenceReadinessUnderstandingModel(),
    )

    result = run_conversation(
        EVIDENCE_MISSING_MESSAGE,
        "evidence-readiness-case",
        request_id="open-case-with-readiness",
        runner=get_runner("after-sales-evidence-readiness-v3"),
    )

    assert result["status"] == "PARTIAL"
    assert result["terminal_reason"] == "EVIDENCE_UNAVAILABLE"
    assert result["pending_action"] is None
    assert [entry["tool"] for entry in result["trace"]] == ["prepare_after_sales_case"]
    args = result["trace"][0]["args"]
    assert args["evidence_readiness_claims"] == [{
        "evidence_type": "ORIGINAL_PACKAGING",
        "disposition": "DEFERRED",
        "source_quote": "原包装情况稍后才能确认",
    }]
    observation = result["trace"][0]["result"]
    autonomous = result["plan"]["steps"][0]["result"]["autonomous_execution"]
    assert autonomous["status"] == "WAIT_USER_INPUT"
    assert autonomous["reason_code"] == "CUSTOMER_EVIDENCE_DEFERRED"
    assert "暂时无法提供" in result["reply"]
    with tools._session() as session:
        case = AfterSalesCaseService(session).get_case(observation["case_id"])
        readiness_events = [event for event in case.events
                            if event["event_type"] == "EVIDENCE_READINESS_RECORDED"]
        assert len(readiness_events) == 1
        assert readiness_events[0]["payload"]["evidence_type"] == "ORIGINAL_PACKAGING"
        assert readiness_events[0]["payload"]["source_quote"] == "原包装情况稍后才能确认"
        assert {item["evidence_type"] for item in case.evidence} == {"ISSUE_DESCRIPTION"}
        plan = case.plans[-1]
        deferred = next(action for action in plan["next_actions"]
                        if action["requirement_id"].endswith(":ORIGINAL_PACKAGING"))
        assert deferred["kind"] == "WAIT_FOR_CUSTOMER_EVIDENCE"
        assert deferred["claim_id"] == readiness_events[0]["payload"]["claim_id"]
        refund = session.scalar(select(Refund).where(Refund.order_id == 1001))
        assert refund.status == "none"


def test_evidence_readiness_profile_inherits_case_first_boundary_for_unknown_evidence(database, monkeypatch):
    from src.harness import get_runner

    with tools._session() as session:
        require_original_packaging_evidence(session)
    monkeypatch.setattr(
        orchestrator,
        "get_llm",
        lambda **_kwargs: EvidenceReadinessUnknownUnderstandingModel(),
    )

    result = run_conversation(
        EVIDENCE_UNKNOWN_MESSAGE,
        "readiness-inherits-boundary",
        request_id="open-case-with-unknown-evidence",
        runner=get_runner("after-sales-evidence-readiness-v3"),
    )

    assert result["status"] == "PARTIAL"
    assert result["terminal_reason"] == "EVIDENCE_UNAVAILABLE"
    assert [entry["tool"] for entry in result["trace"]] == ["prepare_after_sales_case"]
    assert "evidence_readiness_claims" not in result["trace"][0]["args"]
    autonomous = result["plan"]["steps"][0]["result"]["autonomous_execution"]
    assert autonomous["status"] == "WAIT_USER_INPUT"
    assert autonomous["reason_code"] == "CUSTOMER_EVIDENCE_REQUIRED"
    with tools._session() as session:
        case = AfterSalesCaseService(session).get_case(result["trace"][0]["result"]["case_id"])
        assert not any(event["event_type"] == "EVIDENCE_READINESS_RECORDED" for event in case.events)
        assert case.plans[-1]["next_actions"][0]["kind"] == "REQUEST_CUSTOMER_EVIDENCE"


def test_evidence_readiness_profile_records_cross_turn_deferral_without_semantic_reassessment(database, monkeypatch):
    from src import llm
    from src.harness import get_runner

    with tools._session() as session:
        require_original_packaging_evidence(session)
    runner = get_runner("after-sales-evidence-readiness-v3")
    first = run_conversation(
        MESSAGE,
        "cross-turn-readiness",
        request_id="open-case-before-readiness",
        runner=runner,
    )
    assert first["status"] == "PARTIAL"
    assert first["terminal_reason"] == "EVIDENCE_UNAVAILABLE"

    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: ExistingCaseReadinessUnderstandingModel())
    monkeypatch.setattr(llm, "get_llm", lambda **_kwargs: SemanticModelMustNotRun())
    second = run_conversation(
        EVIDENCE_DEFERRED_FOLLOWUP,
        "cross-turn-readiness",
        request_id="record-readiness-followup",
        runner=runner,
    )

    assert second["status"] == "PARTIAL"
    assert second["terminal_reason"] == "EVIDENCE_UNAVAILABLE"
    assert [entry["tool"] for entry in second["trace"]] == ["record_after_sales_evidence_readiness"]
    result = second["trace"][0]["result"]
    autonomous = second["plan"]["steps"][0]["result"]["autonomous_execution"]
    assert autonomous["reason_code"] == "CUSTOMER_EVIDENCE_DEFERRED"
    with tools._session() as session:
        case = AfterSalesCaseService(session).get_case(result["case_id"])
        assert {item["evidence_type"] for item in case.evidence} == {"ISSUE_DESCRIPTION"}
        assert [run.role for run in session.scalars(select(AfterSalesCaseRun).where(
            AfterSalesCaseRun.case_id == case.case_id,
        ).order_by(AfterSalesCaseRun.linked_at)).all()] == ["INTAKE", "READINESS_UPDATE"]
        assert case.plans[-1]["next_actions"][0]["kind"] == "WAIT_FOR_CUSTOMER_EVIDENCE"
        assert session.scalar(select(Refund).where(Refund.order_id == 1001)).status == "none"

    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: ReadinessContextInspectionModel())
    third = run_conversation(
        READINESS_CONTEXT_ACK,
        "cross-turn-readiness",
        request_id="read-trusted-readiness-context",
        runner=runner,
    )
    assert third["status"] == "SUCCEEDED"
    assert third["trace"] == []


def test_multi_item_case_is_routed_to_manual_review_without_order_refund(database, monkeypatch):
    with tools._session() as session:
        session.add(
            OrderItem(
                item_id="OI-1001-2",
                order_id=1001,
                line_number=2,
                name="附加配件",
                quantity=1,
                unit_price_minor=1200,
                currency="CNY",
                attributes={},
            )
        )
        session.get(Order, 1001).amount = 91.99
        session.scalar(select(Payment).where(Payment.order_id == 1001)).amount = 91.99
        session.commit()
    monkeypatch.setattr(
        orchestrator,
        "get_llm",
        lambda **_kwargs: MultiItemAfterSalesUnderstandingModel(),
    )

    result = run_conversation(
        MULTI_ITEM_MESSAGE,
        "multi-item-case",
        request_id="open-multi-item-case",
    )

    observation = result["trace"][0]["result"]
    assert result["status"] == "PARTIAL"
    assert result["terminal_reason"] == "EVIDENCE_UNAVAILABLE"
    assert result["pending_action"] is None
    assert result["trace"][0]["args"]["order_item_id"] == "OI-1001-1"
    assert observation["case_status"] == "MANUAL_REVIEW"
    assert observation["policy_decision"]["reason_codes"] == [
        "REFUND_SCOPE_REQUIRES_LINE_ITEM_EXECUTION"
    ]
    with tools._session() as session:
        refund = session.scalar(select(Refund).where(Refund.order_id == 1001))
        assert refund.status == "none"


def test_changed_business_facts_invalidate_approval_and_allow_reassessment(database):
    first = run_conversation(MESSAGE, "stale-case", request_id="open-case")
    pending = first["pending_action"]
    with tools._session() as session:
        payment = session.scalar(select(Payment).where(Payment.order_id == 1001))
        payment.status = "refunded"
        session.commit()

    stale = run_conversation(
        "确认退款",
        "stale-case",
        request_id="stale-submit",
        action_id=pending["action_id"],
    )

    assert stale["status"] == "FAILED"
    assert stale["terminal_reason"] == "APPROVAL_STALE"
    assert stale["pending_action"] is None
    with tools._session() as session:
        action = session.get(PendingAction, pending["action_id"])
        refund = session.scalar(select(Refund).where(Refund.order_id == 1001))
        assert action.status == "CLOSED"
        assert refund.status == "none"

    reassessed = run_conversation(MESSAGE, "stale-case", request_id="reassess-case")
    observation = reassessed["trace"][0]["result"]
    assert reassessed["status"] == "SUCCEEDED"
    assert reassessed["terminal_reason"] == "BUSINESS_RULE_REJECTED"
    assert reassessed["pending_action"] is None
    assert observation["case_id"] == pending["case_id"]
    assert observation["policy_decision"]["outcome"] == "INELIGIBLE"
    assert observation["policy_decision"]["reason_codes"] == ["PAYMENT_STATUS_NOT_ELIGIBLE"]


def test_refund_scope_change_after_approval_is_rejected_without_refund(database):
    first = run_conversation(MESSAGE, "scope-change", request_id="open-case")
    pending = first["pending_action"]
    with tools._session() as session:
        session.add(
            OrderItem(
                item_id="OI-1001-2",
                order_id=1001,
                line_number=2,
                name="附加配件",
                quantity=1,
                unit_price_minor=1200,
                currency="CNY",
                attributes={},
            )
        )
        session.get(Order, 1001).amount = 91.99
        session.scalar(select(Payment).where(Payment.order_id == 1001)).amount = 91.99
        session.commit()

    rejected = run_conversation(
        "确认退款",
        "scope-change",
        request_id="submit-after-scope-change",
        action_id=pending["action_id"],
    )

    assert rejected["status"] == "FAILED"
    assert rejected["terminal_reason"] == "REFUND_SCOPE_MISMATCH"
    assert rejected["pending_action"] is None
    with tools._session() as session:
        refund = session.scalar(select(Refund).where(Refund.order_id == 1001))
        assert refund.status == "none"
        assert session.scalar(select(AfterSalesCaseRefund).where(
            AfterSalesCaseRefund.case_id == pending["case_id"]
        )) is None


class CrashAfterCaseWrite(BaseException):
    pass


class CrashAfterCaseToolResult(RunStore):
    def save_checkpoint(self, run_id, token, snapshot, event=None):
        if event and event.get("type") == "tool_result" and event.get("tool") == "prepare_after_sales_case":
            raise CrashAfterCaseWrite()
        super().save_checkpoint(run_id, token, snapshot, event)


class CrashAfterAutonomousCaseToolResult(RunStore):
    def save_checkpoint(self, run_id, token, snapshot, event=None):
        if (event and event.get("type") == "tool_result"
                and event.get("tool") == "continue_after_sales_case" and not event.get("reused")):
            raise CrashAfterCaseWrite()
        super().save_checkpoint(run_id, token, snapshot, event)


class CrashAfterReadinessToolResult(RunStore):
    def save_checkpoint(self, run_id, token, snapshot, event=None):
        if (event and event.get("type") == "tool_result"
                and event.get("tool") == "record_after_sales_evidence_readiness"
                and not event.get("reused")):
            raise CrashAfterCaseWrite()
        super().save_checkpoint(run_id, token, snapshot, event)


def test_case_creation_reuses_transactional_ledger_after_process_restart(database):
    with pytest.raises(CrashAfterCaseWrite):
        run_conversation(
            MESSAGE,
            "case-recovery",
            request_id="recoverable-case",
            store=CrashAfterCaseToolResult(),
        )
    active_run_id = RunStore().conversation("case-recovery")["active_run_id"]
    with tools._session() as session:
        assert len(session.scalars(select(AfterSalesCase)).all()) == 1
        assert len(session.scalars(select(ToolLedger).where(
            ToolLedger.run_id == active_run_id,
            ToolLedger.tool_name == "prepare_after_sales_case",
        )).all()) == 1

    tools.configure_db(get_engine(f"sqlite:///{database}"))
    resumed = resume_run("case-recovery", active_run_id)

    assert resumed["status"] == "WAITING_APPROVAL"
    assert resumed["trace"][0]["status"] == "REUSED"
    with tools._session() as session:
        cases = session.scalars(select(AfterSalesCase)).all()
        assert len(cases) == 1
        assert cases[0].revision == 5


def test_autonomous_case_step_reuses_transactional_ledger_after_process_restart(database, monkeypatch):
    from src import llm

    with tools._session() as session:
        require_authoritative_order_evidence(session)
    monkeypatch.setattr(llm, "get_llm", lambda **_kwargs: SemanticModelMustNotRun())

    with pytest.raises(CrashAfterCaseWrite):
        run_conversation(
            MESSAGE,
            "autonomous-case-recovery",
            request_id="recoverable-autonomous-case",
            store=CrashAfterAutonomousCaseToolResult(),
        )
    active_run_id = RunStore().conversation("autonomous-case-recovery")["active_run_id"]

    resumed = resume_run("autonomous-case-recovery", active_run_id)

    assert resumed["status"] == "WAITING_APPROVAL"
    assert [entry["tool"] for entry in resumed["trace"]] == [
        "prepare_after_sales_case",
        "continue_after_sales_case",
    ]
    assert all(entry["status"] == "REUSED" for entry in resumed["trace"])
    with tools._session() as session:
        case = AfterSalesCaseService(session).get_case(resumed["pending_action"]["case_id"])
        order_evidence = [row for row in case.evidence if row["evidence_type"] == "ORDER_FACT"]
        continuation_rows = session.scalars(select(ToolLedger).where(
            ToolLedger.run_id == active_run_id,
            ToolLedger.tool_name == "continue_after_sales_case",
        )).all()
        assert len(order_evidence) == 1
        assert len(continuation_rows) == 1


def test_obligation_graph_step_reuses_transactional_ledger_after_process_restart(database, monkeypatch):
    from src import llm
    from src.harness import get_runner

    with tools._session() as session:
        require_authoritative_order_evidence(session)
    monkeypatch.setattr(llm, "get_llm", lambda **_kwargs: SemanticModelMustNotRun())
    runner = get_runner("after-sales-obligation-graph-v4")

    with pytest.raises(CrashAfterCaseWrite):
        run_conversation(
            MESSAGE,
            "obligation-graph-recovery",
            request_id="recoverable-obligation-graph",
            store=CrashAfterAutonomousCaseToolResult(),
            runner=runner,
        )
    active_run_id = RunStore().conversation("obligation-graph-recovery")["active_run_id"]

    resumed = resume_run("obligation-graph-recovery", active_run_id)

    assert resumed["status"] == "WAITING_APPROVAL"
    assert [entry["tool"] for entry in resumed["trace"]] == [
        "prepare_after_sales_case",
        "continue_after_sales_case",
    ]
    assert all(entry["status"] == "REUSED" for entry in resumed["trace"])
    autonomous = resumed["plan"]["steps"][0]["result"]["autonomous_execution"]
    assert autonomous["action_selection_mode"] == "obligation_graph"
    assert len(autonomous["obligation_graph_ids"]) == 2
    with tools._session() as session:
        case = AfterSalesCaseService(session).get_case(resumed["pending_action"]["case_id"])
        order_evidence = [row for row in case.evidence if row["evidence_type"] == "ORDER_FACT"]
        continuation_rows = session.scalars(select(ToolLedger).where(
            ToolLedger.run_id == active_run_id,
            ToolLedger.tool_name == "continue_after_sales_case",
        )).all()
        assert len(order_evidence) == 1
        assert len(continuation_rows) == 1


def test_cross_turn_readiness_reuses_transactional_ledger_after_process_restart(database, monkeypatch):
    from src import llm
    from src.harness import get_runner

    with tools._session() as session:
        require_original_packaging_evidence(session)
    runner = get_runner("after-sales-evidence-readiness-v3")
    opened = run_conversation(
        MESSAGE,
        "readiness-recovery",
        request_id="open-before-readiness-recovery",
        runner=runner,
    )
    case_id = opened["trace"][0]["result"]["case_id"]
    monkeypatch.setattr(orchestrator, "get_llm", lambda **_kwargs: ExistingCaseReadinessUnderstandingModel())
    monkeypatch.setattr(llm, "get_llm", lambda **_kwargs: SemanticModelMustNotRun())

    with pytest.raises(CrashAfterCaseWrite):
        run_conversation(
            EVIDENCE_DEFERRED_FOLLOWUP,
            "readiness-recovery",
            request_id="recoverable-readiness",
            runner=runner,
            store=CrashAfterReadinessToolResult(),
        )
    active_run_id = RunStore().conversation("readiness-recovery")["active_run_id"]
    with tools._session() as session:
        case = AfterSalesCaseService(session).get_case(case_id)
        assert len([event for event in case.events
                    if event["event_type"] == "EVIDENCE_READINESS_RECORDED"]) == 1
        assert len(session.scalars(select(ToolLedger).where(
            ToolLedger.run_id == active_run_id,
            ToolLedger.tool_name == "record_after_sales_evidence_readiness",
        )).all()) == 1

    resumed = resume_run("readiness-recovery", active_run_id)

    assert resumed["status"] == "PARTIAL"
    assert resumed["trace"][0]["status"] == "REUSED"
    with tools._session() as session:
        case = AfterSalesCaseService(session).get_case(case_id)
        assert len([event for event in case.events
                    if event["event_type"] == "EVIDENCE_READINESS_RECORDED"]) == 1
        assert session.scalar(select(Refund).where(Refund.order_id == 1001)).status == "none"


def test_after_sales_read_api_returns_case_timeline_policy_and_refund_link(database):
    first = run_conversation(MESSAGE, "case-api", request_id="open-case")
    pending = first["pending_action"]
    run_conversation(
        "确认退款",
        "case-api",
        request_id="submit-case",
        action_id=pending["action_id"],
    )
    client = TestClient(api.app)

    listing = client.get(
        "/business/after-sales/cases",
        params={"session_id": "case-api", "status": "SUBMITTED"},
    )
    detail = client.get(f"/business/after-sales/cases/{pending['case_id']}")

    assert listing.status_code == 200
    listed_cases = listing.json()["cases"]
    assert len(listed_cases) == 1
    assert listed_cases[0]["case_id"] == pending["case_id"]
    assert listed_cases[0]["order_id"] == 1001
    assert listed_cases[0]["status"] == "SUBMITTED"
    assert listed_cases[0]["policy_outcome"] == "ELIGIBLE"
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["case"]["case_id"] == pending["case_id"]
    assert [event["sequence"] for event in payload["case"]["events"]] == [1, 2, 3, 4, 5, 6, 7]
    assert {row["role"] for row in payload["runs"]} == {"INTAKE", "SUBMISSION"}
    assert payload["refund"]["refund_id"] == listed_cases[0]["refund_id"]
    assert payload["policy"]["policy_version_id"] == DEFAULT_AFTER_SALES_POLICY_VERSION_ID
    assert len(payload["policy"]["checksum"]) == 64
    assert client.get("/business/after-sales/cases/missing").status_code == 404
