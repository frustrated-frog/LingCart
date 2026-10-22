"""Crash/restart boundaries use a file-backed DB and fresh store instances."""
from copy import deepcopy
import threading

import pytest
from sqlalchemy import select

from src import tools
from src.backend.db import AgentRun, ConversationMessage, PendingAction, Payment, ToolLedger, get_engine, get_session_factory
from src.backend.seed import seed
from src.durable import load_conversation, resume_run, run_conversation
from src.errors import AgentError
from src.persistence import RunStore


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "durable.db"
    engine = get_engine(f"sqlite:///{path}")
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)
    yield path
    engine.dispose()


def test_request_retry_returns_same_run_and_one_answer(database):
    first = run_conversation("查询订单 1002 的物流", "session", request_id="same")
    repeated = run_conversation("查询订单 1002 的物流", "session", request_id="same")
    assert repeated["run_id"] == first["run_id"] and repeated["cached"]
    assert len(RunStore().conversation("session")["messages"]) == 2
    with pytest.raises(AgentError, match="不同的需求"):
        run_conversation("另一个请求", "session", request_id="same")


def test_final_answer_stream_and_public_statuses_are_durable(database):
    from src.response_stream import response_text
    from src.trace_projector import project_trace

    result = run_conversation("查询订单 1002 的状态", "stream-events", request_id="stream-once")
    events = result["events"]
    completed = next(event for event in events if event["type"] == "response_completed")
    types = [event["type"] for event in events]
    statuses = [event["status"] for event in events if event["type"] == "agent_status_changed"]

    assert response_text(events, completed["message_id"]) == result["reply"]
    assert types.index("response_started") < types.index("response_completed") < types.index("terminal")
    assert statuses[0] == "UNDERSTANDING"
    assert "USING_TOOL" in statuses
    assert "VERIFYING" in statuses
    assert statuses[-2:] == ["RESPONDING", "COMPLETED"]
    projected = project_trace(events)
    assert projected["integrity"]["response_stream"] is True
    assert projected["integrity"]["complete"] is True


def test_request_id_cannot_rebind_to_another_harness_profile(database):
    from src.harness import get_runner

    baseline = get_runner("adaptive-v1")
    candidate = get_runner("adaptive-objective-context-v1")
    first = run_conversation("查询订单 1002 的状态", "profile-idempotency",
                             request_id="stable", runner=baseline)

    with pytest.raises(AgentError) as error:
        run_conversation("查询订单 1002 的状态", "profile-idempotency",
                         request_id="stable", runner=candidate)

    assert error.value.code == "REQUEST_CONFIGURATION_CONFLICT"
    repeated = run_conversation("查询订单 1002 的状态", "profile-idempotency",
                                request_id="stable", runner=baseline)
    assert repeated["cached"] and repeated["run_id"] == first["run_id"]


def test_profile_runner_provenance_is_persisted_with_the_run(database):
    from src.harness import get_runner

    result = run_conversation("查询订单 1002 的状态", "profile", runner=get_runner("baseline-v1"))
    stored = RunStore().run(result["run_id"])["result"]
    assert result["events"][0]["type"] == "harness_profile_selected"
    assert result["harness"]["profile_id"] == "baseline-v1"
    assert stored["harness"] == result["harness"]


def test_adaptive_profile_is_restored_without_external_runner(database, monkeypatch):
    from src.harness import get_registry
    from src.harness.contracts import RunRequest
    from src.runtime import BudgetLimits, current_runtime

    class Crash(BaseException):
        pass

    registry = get_registry()
    original = registry._engines[("adaptive", "1")]

    class RestartableAdaptive:
        engine_id = "adaptive"
        engine_version = "1"
        execution_mode = "adaptive"

        def __init__(self):
            self.calls = 0
            self.observed = []

        def provenance(self):
            return original.provenance()

        def execute(self, request: RunRequest):
            self.calls += 1
            context = current_runtime()
            self.observed.append((context.limits, context.mechanisms["context"].mechanism_id))
            if self.calls == 1:
                raise Crash()
            return {"reply": "已按原 Profile 恢复", "trace": [], "agent": "shopping", "route": "ADAPTIVE",
                    "status": "SUCCEEDED", "terminal_reason": "COMPLETED", "pending_action": None,
                    "usage": {}, "events": context.events}

    engine = RestartableAdaptive()
    monkeypatch.setitem(registry._engines, ("adaptive", "1"), engine)
    runner = registry.resolve("adaptive-v1")
    with pytest.raises(Crash):
        run_conversation("推荐预算500元的耳机", "profile-resume", runner=runner)
    active = RunStore().conversation("profile-resume")["active_run_id"]

    result = resume_run("profile-resume", active)

    assert engine.calls == 2
    assert all(limits == BudgetLimits.adaptive() for limits, _ in engine.observed)
    assert [mechanism for _, mechanism in engine.observed] == ["evidence-reference-context"] * 2
    assert result["harness"]["profile_id"] == "adaptive-v1"
    assert result["events"][0]["type"] == "harness_profile_selected"


def test_wrong_resume_runner_is_rejected_before_it_can_hold_the_lease(database):
    from src.harness import get_runner

    class Crash(BaseException):
        pass

    class CrashBeforeEngine(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            super().save_checkpoint(run_id, token, snapshot, event)
            if event and event["type"] == "harness_profile_selected":
                raise Crash()

    candidate = get_runner("adaptive-objective-context-v1")
    with pytest.raises(Crash):
        run_conversation("查询订单1002状态", "wrong-resume-runner", runner=candidate,
                         store=CrashBeforeEngine())
    run_id = RunStore().conversation("wrong-resume-runner")["active_run_id"]

    with pytest.raises(AgentError) as missing_contract:
        RunStore().claim("wrong-resume-runner", "resume", "", {}, resume_id=run_id)
    assert missing_contract.value.code == "EXECUTION_CONTRACT_REQUIRED"
    assert RunStore().run(run_id)["status"] == "INTERRUPTED"

    with pytest.raises(AgentError) as error:
        resume_run("wrong-resume-runner", run_id, runner=get_runner("baseline-v1"))

    assert error.value.code == "HARNESS_PROFILE_CONFLICT"
    assert RunStore().run(run_id)["status"] == "INTERRUPTED"

    changed_implementation = get_runner("adaptive-objective-context-v1")
    changed_implementation._provenance["engine"]["implementation_sha256"] = "changed-after-crash"
    with pytest.raises(AgentError) as mismatch:
        resume_run("wrong-resume-runner", run_id, runner=changed_implementation)

    assert mismatch.value.code == "HARNESS_IMPLEMENTATION_MISMATCH"
    assert RunStore().run(run_id)["status"] == "INTERRUPTED"
    assert resume_run("wrong-resume-runner", run_id)["harness"]["profile_id"] == candidate.profile.profile_id


def test_completed_run_keeps_a_versioned_terminal_checkpoint(database):
    result = run_conversation("查询订单 1002 的物流", "versioned")

    checkpoint = RunStore().run(result["run_id"])["checkpoint"]
    assert checkpoint["schema_version"] == 2
    assert checkpoint["state_version"] > 1
    assert checkpoint["run_id"] == result["run_id"]
    assert checkpoint["status"] == "SUCCEEDED"
    assert checkpoint["current_node"] == "TERMINAL"
    assert checkpoint["phase"] == "TERMINAL"
    assert checkpoint["terminal_reason"] == "COMPLETED"
    assert len(checkpoint["checksum"]) == 64


def test_stale_checkpoint_writer_is_rejected(database):
    store = RunStore()
    claim = store.claim("conflict", "request", "查询订单 1002", {"max_seconds": 30})
    first_writer = deepcopy(claim["snapshot"])
    stale_writer = deepcopy(claim["snapshot"])

    store.save_checkpoint(claim["run_id"], claim["token"], first_writer)
    with pytest.raises(AgentError) as conflict:
        store.save_checkpoint(claim["run_id"], claim["token"], stale_writer)
    assert conflict.value.code == "CHECKPOINT_CONFLICT"

    store.interrupt(claim["run_id"], claim["token"])


class CrashAfterWrite(BaseException):
    pass


class CrashingStore(RunStore):
    def save_checkpoint(self, run_id, token, snapshot, event=None):
        if event and event.get("type") == "tool_result" and event.get("tool") == "process_payment":
            raise CrashAfterWrite()
        super().save_checkpoint(run_id, token, snapshot, event)


def test_restart_after_write_commit_reuses_ledger_and_saves_one_answer(database):
    first = run_conversation("帮我支付订单 1003", "pay")
    with pytest.raises(CrashAfterWrite):
        run_conversation("确认使用 card 支付", "pay", action_id=first["pending_action"]["action_id"], store=CrashingStore())
    active = RunStore().conversation("pay")["active_run_id"]
    assert tools.get_payment_status(1003)["status"] == "paid"
    # Recreate engine/session factory to emulate losing every process-local object.
    tools.configure_db(get_engine(f"sqlite:///{database}"))
    result = resume_run("pay", active)
    assert result["status"] == "SUCCEEDED"
    assert result["trace"][0]["status"] == "REUSED"
    with tools._session() as session:
        assert len(session.scalars(select(ToolLedger).where(ToolLedger.tool_name == "process_payment")).all()) == 1
        assert len(session.scalars(select(ConversationMessage).where(ConversationMessage.run_id == active, ConversationMessage.role == "assistant")).all()) == 1
    assert resume_run("pay", active)["cached"]
    assert RunStore().conversation("pay")["active_run_id"] is None


def test_approval_restores_on_restart_and_is_session_scoped(database):
    first = run_conversation("订单 1001 我想退款", "refund")
    tools.configure_db(get_engine(f"sqlite:///{database}"))
    restored = load_conversation("refund")
    assert restored["memory"].pending_action == first["pending_action"]
    with pytest.raises(AgentError) as error:
        run_conversation("确认继续", "other", action_id=first["pending_action"]["action_id"])
    assert error.value.code == "APPROVAL_STALE"
    assert tools.check_refund_eligibility(1001)["eligible"]
    completed = run_conversation("确认继续", "refund", action_id=first["pending_action"]["action_id"])
    assert completed["status"] == "SUCCEEDED"


def test_missing_payment_method_keeps_approval_available(database):
    from src.backend.db import Payment
    with tools._session() as session:
        session.scalar(select(Payment).where(Payment.order_id == 1003)).method = ""
        session.commit()
    first = run_conversation("帮我支付订单 1003", "method")
    missing = run_conversation("确认", "method")
    assert missing["status"] == "WAITING_CLARIFICATION"
    confirmed = run_conversation("确认使用 card 支付", "method")
    assert confirmed["status"] == "SUCCEEDED"


def test_only_one_worker_can_claim_a_conversation(database):
    store = RunStore()
    entered = threading.Event()
    release = threading.Event()
    output = []
    def blocking_runner(message, **kwargs):
        entered.set()
        assert release.wait(2)
        from src.orchestrator import run_orchestrator
        return run_orchestrator(message, **kwargs)
    worker = threading.Thread(target=lambda: output.append(run_conversation("查询订单 1002 的状态", "busy", request_id="one", runner=blocking_runner)))
    worker.start()
    assert entered.wait(2)
    try:
        with pytest.raises(AgentError) as error:
            run_conversation("查询订单 1001 的状态", "busy", request_id="two")
        assert error.value.code == "CONVERSATION_BUSY"
    finally:
        release.set()
        worker.join(3)
    assert output[0]["status"] == "SUCCEEDED"


def test_crash_inside_business_transaction_rolls_back_both_write_and_ledger(database, monkeypatch):
    import src.direct_executor as direct
    original = tools.process_payment
    def crashing_payment(order_id: int, method: str) -> dict:
        original(order_id, method)
        raise CrashAfterWrite()
    # The fault happens inside the real local DB transaction.
    from src.runtime import ToolSpec, tool_spec
    schema = tool_spec("process_payment", original).schema
    first = run_conversation("帮我支付订单 1003", "rollback")
    monkeypatch.setitem(direct._APPROVED_TOOLS, "process_payment", ToolSpec(crashing_payment, schema, True, True))
    monkeypatch.setattr(tools, "process_payment", crashing_payment)
    with pytest.raises(CrashAfterWrite):
        run_conversation("确认", "rollback")
    assert tools.get_payment_status(1003)["status"] == "pending"
    with tools._session() as session:
        assert not session.scalars(select(ToolLedger).where(ToolLedger.tool_name == "process_payment")).all()
    monkeypatch.setitem(direct._APPROVED_TOOLS, "process_payment", original)
    monkeypatch.setattr(tools, "process_payment", original)
    result = resume_run("rollback", RunStore().conversation("rollback")["active_run_id"])
    assert result["status"] == "SUCCEEDED"
    assert tools.get_payment_status(1003)["status"] == "paid"


def test_model_decision_and_tool_result_survive_restart(database, monkeypatch):
    from langchain_core.messages import AIMessage
    from src.agents import base
    from src.agents.order_agent import get_order_status
    class StopBeforeFinal(RunStore):
        def save_checkpoint(self, run_id, token, snapshot, event=None):
            if event and event["type"] == "model_started" and snapshot["models"]:
                raise CrashAfterWrite()
            super().save_checkpoint(run_id, token, snapshot, event)
    class Scripted:
        def __init__(self):
            self.calls = 0
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            self.calls += 1
            return AIMessage(content="", tool_calls=[{"name": "get_order_status", "args": {"order_id": 1002}, "id": "one"}])
    first_model = Scripted()
    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: first_model)
    def runner(message, **kwargs):
        return base.run_tool_agent("order", "Get real order status", [get_order_status], message,
                                   kwargs.get("history"), run_id=__import__('src.runtime', fromlist=['current_runtime']).current_runtime().run_id)
    with pytest.raises(CrashAfterWrite):
        run_conversation("物流怎么了", "model", store=StopBeforeFinal(), runner=runner)
    assert first_model.calls == 1
    class FinalModel(Scripted):
        def invoke(self, messages):
            self.calls += 1
            assert any(getattr(m, "type", None) == "tool" for m in messages)
            return AIMessage(content="订单 1002 运输中")
    next_model = FinalModel()
    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: next_model)
    result = resume_run("model", RunStore().conversation("model")["active_run_id"], runner=runner)
    assert result["status"] == "SUCCEEDED"
    assert next_model.calls == 1
    assert result["trace"][0]["status"] == "REUSED"
    assert any(e["type"] == "model_reused" for e in RunStore().run(result["run_id"])["events"])


def test_confirmation_word_in_read_only_question_cannot_charge(database):
    first = run_conversation("帮我支付订单 1003", "explicit")
    query = run_conversation("帮我确认一下订单 1003 的物流", "explicit")
    assert query["status"] == "SUCCEEDED"
    assert query["pending_action"]["action_id"] == first["pending_action"]["action_id"]
    assert tools.get_payment_status(1003)["status"] == "pending"
    policy = run_conversation("取消订单有什么政策", "explicit")
    assert policy["pending_action"]["action_id"] == first["pending_action"]["action_id"]
    assert tools.get_payment_status(1003)["status"] == "pending"


def test_confirmation_for_different_business_action_cannot_authorize_payment(database):
    first = run_conversation("帮我支付订单 1003", "mismatch")
    result = run_conversation("确认退款", "mismatch")
    assert result["status"] == "WAITING_CLARIFICATION"
    assert result["pending_action"] == first["pending_action"]
    assert tools.get_payment_status(1003)["status"] == "pending"


def test_external_write_with_uncertain_outcome_is_not_replayed(database):
    from src.runtime import BudgetLimits, RunRuntime, ToolExecutor, ToolSpec, TraceRecorder, runtime_scope, tool_spec
    store = RunStore()
    first = run_conversation("帮我支付订单 1003", "external")
    claim = store.claim("external", "write", "确认", BudgetLimits().__dict__)
    calls = []
    def external_payment(order_id: int, method: str):
        calls.append(order_id)
        raise TimeoutError("external outcome unknown")
    approval = {**first['pending_action'], 'tool': 'process_payment', 'args': {'order_id': 1003, 'method': 'card'}}
    context = RunRuntime(claim["run_id"], store=store, token=claim["token"], snapshot=claim["snapshot"], approval=approval)
    spec = ToolSpec(external_payment, tool_spec("process_payment", tools.process_payment).schema, True, False)
    with runtime_scope(context):
        executor = ToolExecutor({"process_payment": spec}, TraceRecorder(context.run_id))
        with pytest.raises(AgentError) as first_error:
            executor.execute("process_payment", {"order_id": 1003, "method": "card"})
        assert first_error.value.code == "TOOL_TIMEOUT"
        with pytest.raises(AgentError) as repeated:
            executor.execute("process_payment", {"order_id": 1003, "method": "card"})
        assert repeated.value.code == "UNCERTAIN_TOOL_OUTCOME"
    assert calls == [1003]
    store.interrupt(claim["run_id"], claim["token"])


def test_structured_intake_clarification_survives_service_restart(database, monkeypatch):
    import json
    from langchain_core.messages import AIMessage
    class Model:
        def invoke(self, messages):
            order_id = 1002 if '订单 1002' in messages else None
            return AIMessage(content=json.dumps({"reason": "复合请求", "commands": [], "objectives": [
                {"id": "status", "kind": "ORDER_STATUS", "source_quote": "包裹怎么样", "order_id": order_id},
                {"id": "policy", "kind": "FAQ", "source_quote": "讲讲售后保障", "question": "讲讲售后保障"},
            ]}, ensure_ascii=False))
    monkeypatch.setattr("src.orchestrator.get_llm", lambda **_kwargs: Model())
    first = run_conversation("包裹怎么样，另外讲讲售后保障", "model-slots")
    assert first["status"] == "WAITING_CLARIFICATION"
    assert RunStore().conversation("model-slots")["state"]["clarification"]["missing_slots"] == ["order_id"]
    tools.configure_db(get_engine(f"sqlite:///{database}"))
    second = run_conversation("1002", "model-slots")
    assert second["status"] == "PARTIAL"
    assert second['terminal_reason'] == 'INSUFFICIENT_POLICY_EVIDENCE'
    assert {entry["tool"] for entry in second["trace"]} == {"get_order_status", "answer_faq"}


def test_ledger_recovery_closes_original_execution_evidence(database):
    from src.trace_projector import project_trace
    first = run_conversation('帮我支付订单1003', 'audit-recovery')
    with pytest.raises(CrashAfterWrite):
        run_conversation('确认使用 card 支付', 'audit-recovery', action_id=first['pending_action']['action_id'], store=CrashingStore())
    active = RunStore().conversation('audit-recovery')['active_run_id']
    result = resume_run('audit-recovery', active)
    assert result['status'] == 'SUCCEEDED'
    assert project_trace(result['events'])['integrity']['complete'] is True
    reused = next(e for e in result['events'] if e['type'] == 'tool_result' and e['tool'] == 'process_payment')
    assert reused['source_ledger'] and reused['execution_id']


def test_cancel_after_committed_write_recovers_real_outcome(database):
    from src.session_control import ControlStore
    from src.trace_projector import project_trace
    first = run_conversation('帮我支付订单1003', 'cancel-recovery')
    with pytest.raises(CrashAfterWrite):
        run_conversation('确认使用 card 支付', 'cancel-recovery', action_id=first['pending_action']['action_id'], store=CrashingStore())
    active = RunStore().conversation('cancel-recovery')['active_run_id']
    ControlStore().enqueue('cancel-recovery', active, 'CANCEL', command_id='cancel-stop', expected_revision=0)
    result = resume_run('cancel-recovery', active)
    assert result['status'] == 'CANCELLED'
    assert tools.get_payment_status(1003)['status'] == 'paid'
    assert any(t['tool'] == 'process_payment' and t['result']['success'] for t in result['trace'])
    assert project_trace(result['events'])['integrity']['complete'] is True
