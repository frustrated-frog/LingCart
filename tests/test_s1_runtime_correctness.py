"""生产边界的反例：未知策略、审批替换、串行执行、预算与审计。"""
import json
import threading
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from src.agents import base
from src.context_budget import ContextBudgetError, compact_context
from src.errors import AgentError
from src.runtime import (
    ApprovalRequired, BudgetLimits, ConcurrencyPolicy, ReplayPolicy, RunRuntime, ToolEffect,
    ToolExecutor, ToolRiskLevel, ToolSpec, TraceRecorder, runtime_scope, tool_spec,
)
from src.runtime_state import new_runtime_snapshot, restore_runtime_snapshot
from src.trace_projector import project_trace


def _write_spec(function, **kwargs):
    return ToolSpec(function, effect=ToolEffect.WRITE, risk_level=ToolRiskLevel.GUARDED,
                    replay_policy=ReplayPolicy.AT_MOST_ONCE,
                    concurrency_policy=ConcurrencyPolicy.SERIAL_PER_RUN, **kwargs)


def test_unknown_and_same_named_tools_require_explicit_policy():
    calls = []

    def cancel_order(order_id: int):
        calls.append(order_id)

    for name in ("cancel_order", "get_order_status", "process_payment"):
        with pytest.raises(AgentError) as caught:
            ToolExecutor({name: cancel_order}, TraceRecorder("unregistered"))
        assert caught.value.code == "TOOL_POLICY_MISSING"
    assert calls == []


def test_registered_structured_tools_keep_production_policy():
    from src.agents.order_agent import ORDER_TOOLS
    from src.agents.payment_agent import PAYMENT_TOOLS
    from src.agents.refund_agent import REFUND_TOOLS
    from src.agents.faq_agent import FAQ_TOOLS

    for item in [*ORDER_TOOLS, *PAYMENT_TOOLS, *REFUND_TOOLS, *FAQ_TOOLS]:
        spec = tool_spec(item.name, item)
        assert spec.schema is item.args_schema
        assert spec.effect == (ToolEffect.WRITE if item.name in {"process_payment", "initiate_refund"} else ToolEffect.READ)


@pytest.mark.parametrize("policy", [
    {"effect": "WRITE"},
    {"effect": "WRITE", "risk_level": "HIGH", "replay_policy": "REPLAY_SAFE", "concurrency_policy": "SERIAL_PER_RUN"},
    {"effect": "WRITE", "risk_level": "HIGH", "replay_policy": "AT_MOST_ONCE", "concurrency_policy": "PARALLEL_SAFE"},
    {"effect": "READ", "risk_level": "HIGH"},
    {"side_effect": True, "effect": "READ"},
    {"transactional": True},
    {"approval_action": "CANCEL_ORDER"},
    {"effect": "DELETE"},
])
def test_invalid_policy_combinations_are_rejected(policy):
    with pytest.raises(AgentError) as caught:
        ToolSpec(lambda: None, **policy)
    assert caught.value.code == "INVALID_TOOL_POLICY"


def test_generic_approval_is_bound_to_all_arguments_and_reused_once():
    calls = []

    def cancel_order(order_id: int, reason: str):
        calls.append((order_id, reason))
        return {"order_id": order_id, "status": "cancelled"}

    spec = ToolSpec(cancel_order, side_effect=True, approval_action="CANCEL_ORDER")
    executor = ToolExecutor({"cancel_order": spec}, TraceRecorder("approval"))
    with pytest.raises(ApprovalRequired) as pending:
        executor.execute("cancel_order", {"order_id": 1, "reason": "用户要求"})
    assert calls == []
    approved = ToolExecutor({"cancel_order": spec}, TraceRecorder("approval"), approved_action=pending.value.pending)
    with pytest.raises(ApprovalRequired):
        approved.execute("cancel_order", {"order_id": 1, "reason": "被替换的原因"})
    assert calls == []
    args = {"order_id": 1, "reason": "用户要求"}
    approved.execute("cancel_order", args)
    assert approved.execute("cancel_order", args)[1]["status"] == "REUSED"
    assert calls == [(1, "用户要求")]


def test_failed_at_most_once_read_is_not_retried_or_replayed():
    calls = []

    def fetch_once(value: str):
        calls.append(value)
        raise TimeoutError("timeout")

    context = RunRuntime("once", BudgetLimits(read_attempts=3))
    spec = ToolSpec(fetch_once, replay_policy=ReplayPolicy.AT_MOST_ONCE)
    with runtime_scope(context):
        with pytest.raises(AgentError) as first:
            ToolExecutor({"fetch": spec}, TraceRecorder(context.run_id)).execute("fetch", {"value": "a"})
        assert first.value.code == "TOOL_TIMEOUT"
        with pytest.raises(AgentError) as repeated:
            ToolExecutor({"fetch": spec}, TraceRecorder(context.run_id)).execute("fetch", {"value": "a"})
        assert repeated.value.code == "TOOL_REPLAY_FORBIDDEN"
    assert calls == ["a"]


def test_uncertain_write_is_shared_across_executors_without_store():
    calls = []

    def write(value: str):
        calls.append(value)
        raise TimeoutError("外部结果未知")

    spec = _write_spec(write)
    context = RunRuntime("uncertain")
    with runtime_scope(context):
        with pytest.raises(AgentError):
            ToolExecutor({"write": spec}, TraceRecorder(context.run_id)).execute("write", {"value": "a"})
        with pytest.raises(AgentError) as repeated:
            ToolExecutor({"write": spec}, TraceRecorder(context.run_id)).execute("write", {"value": "a"})
    assert repeated.value.code == "UNCERTAIN_TOOL_OUTCOME"
    assert calls == ["a"]


def test_serial_policy_applies_across_executors_in_same_run():
    active = peak = 0
    lock = threading.Lock()
    barrier = threading.Barrier(3)
    failures = []

    def write(value: int):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(.02)
        with lock:
            active -= 1
        return {"value": value}

    spec = _write_spec(write)
    context = RunRuntime("serial", BudgetLimits(call_timeout=1))

    def worker(value):
        try:
            with runtime_scope(context):
                executor = ToolExecutor({"write": spec}, TraceRecorder(context.run_id))
                barrier.wait(timeout=2)
                executor.execute("write", {"value": value})
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=worker, args=(value,)) for value in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=2)
    for thread in threads:
        thread.join(2)
    assert not failures
    assert not any(thread.is_alive() for thread in threads)
    assert peak == 1


def test_timed_out_worker_keeps_serial_gate_until_business_call_finishes():
    release = threading.Event()
    started = threading.Event()
    calls = []

    def write(value: str):
        calls.append(value)
        started.set()
        release.wait(2)
        return {"value": value}

    spec = _write_spec(write)
    context = RunRuntime("serial-timeout", BudgetLimits(call_timeout=.02))
    try:
        with runtime_scope(context):
            for value in ("first", "second"):
                with pytest.raises(AgentError) as caught:
                    ToolExecutor({"write": spec}, TraceRecorder(context.run_id)).execute("write", {"value": value})
                assert caught.value.code == "TOOL_TIMEOUT"
        assert started.is_set()
        assert calls == ["first"]
    finally:
        release.set()


def test_context_preserves_all_system_messages_and_current_goal():
    system = "系统要求必须完整。" * 20
    goal = "当前订单1002目标必须完整。" * 20
    messages = [SystemMessage(content=system), HumanMessage(content="旧历史" * 300),
                SystemMessage(content="补充系统约束"), HumanMessage(content=goal)]
    compacted, stats = compact_context(messages, max_chars=1000, max_tool_observation_chars=100, max_history_message_chars=20)
    assert [message.content for message in compacted if message.type == "system"] == [system, "补充系统约束"]
    assert compacted[-1].content == goal
    assert stats["output_chars"] <= 1000


def test_compacted_tool_json_preserves_business_evidence_and_reference():
    observation = {"order_id": 1002, "status": "shipped", "amount": 100.25, "reference": "evidence:order-1002",
                   "details": "冗长辅助文字" * 1000, "products": [{"product_id": "P1", "price": 99, "summary": "介绍" * 1000}]}
    messages = [SystemMessage(content="系统"), HumanMessage(content="查询1002"),
                AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"order_id": 1002}, "id": "one"}]),
                ToolMessage(content=json.dumps(observation, ensure_ascii=False), tool_call_id="one")]
    compacted, stats = compact_context(messages, max_chars=700, max_tool_observation_chars=400, max_history_message_chars=50)
    result = json.loads(compacted[-1].content)
    assert result["order_id"] == 1002 and result["status"] == "shipped"
    assert result["amount"] == 100.25 and result["reference"] == "evidence:order-1002"
    assert result["products"][0]["product_id"] == "P1" and result["products"][0]["price"] == 99
    assert result["_context_compacted"] is True
    assert compacted[-1].tool_call_id == "one"
    assert stats["output_chars"] <= 700


@pytest.mark.parametrize("messages", [
    "系统约束与当前目标" * 100,
    [SystemMessage(content="系统" * 100), HumanMessage(content="目标")],
    [SystemMessage(content="系统"), HumanMessage(content="目标" * 100)],
    [SystemMessage(content="系统"), HumanMessage(content="目标"),
     AIMessage(content="", tool_calls=[{"name": "lookup", "args": {}, "id": "one"}]),
     ToolMessage(content=json.dumps({"order_id": 1002, "reason": "关键拒绝原因" * 100}), tool_call_id="one")],
])
def test_minimal_context_budget_exhaustion_is_typed_and_never_calls_model(messages):
    class ForbiddenModel:
        def invoke(self, _messages):
            pytest.fail("预算不足时不应调用模型")

    context = RunRuntime("too-large", BudgetLimits(max_context_chars=100, max_tool_observation_chars=50))
    with pytest.raises(ContextBudgetError) as error:
        context.model_call(ForbiddenModel(), messages, purpose="test")
    assert error.value.code == "CONTEXT_BUDGET_EXHAUSTED"
    assert context.counters["model_calls"] == 0
    assert not context.events


def test_missing_schema_does_not_downgrade_a_tampered_v2_snapshot():
    snapshot = new_runtime_snapshot("run", conversation={}, history=[])
    del snapshot["schema_version"]
    snapshot["conversation"] = {"tampered": True}
    with pytest.raises(AgentError) as error:
        restore_runtime_snapshot(snapshot, "run")
    assert error.value.code == "CHECKPOINT_CORRUPT"


def test_only_well_formed_legacy_snapshot_is_migrated():
    legacy = {"conversation": {}, "history": [], "phase": "MODEL_CALL", "sequence": 2, "models": {}, "budget": {}}
    restored = restore_runtime_snapshot(legacy, "run")
    assert restored["schema_version"] == 2 and restored["state_version"] == 3
    for invalid in ({}, {**legacy, "sequence": True}, {**legacy, "sequence": "2"}, {**legacy, "current_node": "MODEL_CALL"}):
        with pytest.raises(AgentError) as error:
            restore_runtime_snapshot(invalid, "run")
        assert error.value.code == "CHECKPOINT_CORRUPT"


def test_blocked_loop_without_valid_evidence_fails_and_closes_batch(monkeypatch):
    @tool
    def broken(value: str):
        """用于验证失败循环的显式只读工具。"""
        raise RuntimeError("无业务证据")

    class Model:
        def bind_tools(self, _tools):
            return self
        def invoke(self, _messages):
            return AIMessage(content="", tool_calls=[
                {"name": "broken", "args": {"value": "a"}, "id": "one"},
                {"name": "broken", "args": {"value": "b"}, "id": "two"},
            ])

    monkeypatch.setattr(base, "get_llm", lambda **_kwargs: Model())
    result = base.run_tool_agent("test", "系统", [ToolSpec(broken)], "完成当前目标")
    assert result["status"] == "FAILED" and result["terminal_reason"] == "REPEATED_TOOL_CALL"
    assert result["events"][-1]["type"] == "terminal"
    requests = sum(len(message.tool_calls) for message in result["messages"] if isinstance(message, AIMessage))
    assert len([message for message in result["messages"] if isinstance(message, ToolMessage)]) == requests
    assert project_trace(result["events"])["integrity"]["complete"] is True


def test_trace_checks_pair_identity_single_terminal_and_input_order():
    common = {"run_id": "run", "tool": "lookup", "action_fingerprint": "fp", "execution_id": "execution"}
    valid = [dict(common, sequence=1, type="tool_call"), dict(common, sequence=2, type="tool_result", status="SUCCEEDED"),
             {"run_id": "run", "sequence": 3, "type": "terminal", "status": "SUCCEEDED"}]
    assert project_trace(valid)["integrity"]["complete"] is True
    forged = [valid[0], dict(valid[1], tool="different"), valid[2]]
    assert project_trace(forged)["integrity"]["tool_pairs"] is False
    assert project_trace([valid[1], valid[0], valid[2]])["integrity"]["sequential"] is False
    assert project_trace([*valid, dict(valid[2], sequence=4)])["integrity"]["single_terminal"] is False
    assert project_trace([dict(valid[0], run_id=None), *valid[1:]])["integrity"]["single_run"] is False


def test_trace_rejects_equal_protocol_counts_with_mismatched_call_id():
    common = {"run_id": "run", "tool": "lookup", "action_fingerprint": "fp", "protocol_id": "request"}
    events = [dict(common, sequence=1, type="tool_requested", tool_call_id="one"),
              dict(common, sequence=2, type="tool_protocol", tool_call_id="different"),
              {"run_id": "run", "sequence": 3, "type": "terminal", "status": "FAILED"}]
    assert project_trace(events)["integrity"]["protocol_pairs"] is False
    assert project_trace(events)["integrity"]["complete"] is False


def test_trace_does_not_treat_unexecuted_success_or_unproven_reuse_as_complete():
    common = {"run_id": "run", "tool": "lookup", "action_fingerprint": "fp", "protocol_id": "request", "tool_call_id": "one"}
    forged_success = [dict(common, sequence=1, type="tool_requested"),
                      dict(common, sequence=2, type="tool_protocol", status="SUCCEEDED"),
                      {"run_id": "run", "sequence": 3, "type": "terminal", "status": "SUCCEEDED"}]
    assert project_trace(forged_success)["integrity"]["complete"] is False
    unproven_reuse = [{"run_id": "run", "sequence": 1, "type": "tool_result", "tool": "lookup", "status": "REUSED"},
                      {"run_id": "run", "sequence": 2, "type": "terminal", "status": "SUCCEEDED"}]
    assert project_trace(unproven_reuse)["integrity"]["complete"] is None


def test_custom_approval_cannot_use_partial_legacy_business_fields():
    calls = []

    def cancel_order(order_id: int, reason: str):
        calls.append(order_id)

    spec = ToolSpec(cancel_order, side_effect=True, approval_action="CANCEL_ORDER")
    executor = ToolExecutor({"cancel_order": spec}, TraceRecorder("approval-partial"),
                            approved_action={"type": "CANCEL_ORDER", "order_id": 1})
    with pytest.raises(ApprovalRequired):
        executor.execute("cancel_order", {"order_id": 1, "reason": "未经绑定的参数"})
    assert calls == []


def test_custom_tool_with_builtin_name_requires_explicit_argument_binding():
    def cancel_order(order_id: int, reason: str):
        return {'cancelled': True}
    executor = ToolExecutor({'process_payment': ToolSpec(cancel_order, side_effect=True, approval_action='CANCEL_ORDER')},
                            TraceRecorder('custom-name'), approved_action={'type': 'CANCEL_ORDER', 'order_id': 1003})
    with pytest.raises(ApprovalRequired):
        executor.execute('process_payment', {'order_id': 1003, 'reason': '未经授权'})


@pytest.mark.parametrize('status,event_type', [('SUCCEEDED', 'tool_result'), ('FAILED', 'error'), ('WAITING_APPROVAL', 'approval_required')])
def test_one_evidence_cannot_satisfy_two_tool_protocols(status, event_type):
    common = {'run_id': 'r', 'tool': 'lookup', 'action_fingerprint': 'fp'}
    events = [dict(common, sequence=1, type='tool_requested', protocol_id='p1', tool_call_id='one'),
              dict(common, sequence=2, type='tool_requested', protocol_id='p2', tool_call_id='two')]
    if event_type != 'approval_required':
        events.append(dict(common, sequence=3, type='tool_call', execution_id='e1'))
    events.append(dict(common, sequence=len(events)+1, type=event_type, status=status,
                       **({'execution_id': 'e1'} if event_type != 'approval_required' else {})))
    for protocol, call in [('p1', 'one'), ('p2', 'two')]:
        events.append(dict(common, sequence=len(events)+1, type='tool_protocol', protocol_id=protocol, tool_call_id=call, status=status))
    events.append({'run_id': 'r', 'sequence': len(events)+1, 'type': 'terminal', 'status': 'FAILED'})
    assert project_trace(events)['integrity']['complete'] is False
