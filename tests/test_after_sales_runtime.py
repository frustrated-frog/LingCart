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
    PendingAction,
    Payment,
    Refund,
    ShipmentEvent,
    ToolLedger,
    get_engine,
    get_session_factory,
)
from src.backend.seed import DEFAULT_AFTER_SALES_POLICY_VERSION_ID, seed
from src.durable import resume_run, run_conversation
from src.errors import AgentError
from src.persistence import RunStore
from src.runtime import ReplayPolicy, ToolEffect, ToolRiskLevel, tool_spec


MESSAGE = "订单 1001 的耳机出现间歇性异响，需要建立售后案件处理"


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
    yield path
    engine.dispose()


def test_after_sales_tools_declare_distinct_write_and_approval_policies(database):
    prepare = tool_spec("prepare_after_sales_case", tools.prepare_after_sales_case)
    submit = tool_spec("submit_after_sales_case", tools.submit_after_sales_case)

    assert prepare.effect == ToolEffect.WRITE
    assert prepare.risk_level == ToolRiskLevel.GUARDED
    assert prepare.replay_policy == ReplayPolicy.IDEMPOTENT_WRITE
    assert prepare.transactional is True
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


class CrashAfterCaseWrite(BaseException):
    pass


class CrashAfterCaseToolResult(RunStore):
    def save_checkpoint(self, run_id, token, snapshot, event=None):
        if event and event.get("type") == "tool_result" and event.get("tool") == "prepare_after_sales_case":
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
        assert cases[0].revision == 4


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
    assert [event["sequence"] for event in payload["case"]["events"]] == [1, 2, 3, 4, 5, 6]
    assert {row["role"] for row in payload["runs"]} == {"INTAKE", "SUBMISSION"}
    assert payload["refund"]["refund_id"] == listed_cases[0]["refund_id"]
    assert payload["policy"]["policy_version_id"] == DEFAULT_AFTER_SALES_POLICY_VERSION_ID
    assert len(payload["policy"]["checksum"]) == 64
    assert client.get("/business/after-sales/cases/missing").status_code == 404
