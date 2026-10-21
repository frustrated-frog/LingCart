"""SQLite run store. Local business writes and their ledger commit atomically."""
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import hashlib
import json
import uuid

from sqlalchemy import select, text

from src import tools
from src.backend.db import AgentRun, AgentStep, Checkpoint, Conversation, ConversationMessage, PendingAction, ToolLedger
from src.errors import AgentError
from src.runtime_state import (
    advance_runtime_snapshot,
    new_runtime_snapshot,
    restore_runtime_snapshot,
    seal_runtime_snapshot,
)


def utcnow():
    return datetime.now(UTC).replace(tzinfo=None)


def json_value(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _stored_execution_contract(snapshot):
    contract = snapshot.get("execution_contract")
    if isinstance(contract, dict):
        return contract
    harness = snapshot.get("harness")
    profile_id = snapshot.get("harness_profile_id")
    if harness is None and profile_id:
        harness = {"profile_id": profile_id, "legacy_profile_reference": True}
    mode = snapshot.get("execution_mode")
    if mode is None and isinstance(harness, dict):
        mode = (harness.get("engine") or {}).get("execution_mode")
    if harness is None:
        # Pre-contract runs used these exact built-in paths. Keep their cached
        # idempotency semantics without weakening newly recorded contracts.
        harness = {"profile_id": "adaptive-v1" if mode == "adaptive" else "baseline-v1",
                   "legacy_profile_reference": True}
    return {"execution_mode": mode or "auto", "harness": harness}


def _contracts_match(stored, requested, *, require_exact_execution=False):
    if stored.get("execution_mode") != requested.get("execution_mode"):
        return False
    left, right = stored.get("harness"), requested.get("harness")
    if (left is None) != (right is None):
        return False
    if left is None:
        return True
    if left.get("profile_id") != right.get("profile_id"):
        return False
    if left.get("profile_sha256") and right.get("profile_sha256") and left["profile_sha256"] != right["profile_sha256"]:
        return False
    if require_exact_execution:
        from src.harness.registry import harness_execution_matches
        if not harness_execution_matches(left, right):
            return False
    return True


@contextmanager
def transaction():
    with tools._SessionLocal() as session:
        if session.bind.dialect.name == "sqlite":
            session.execute(text("BEGIN IMMEDIATE"))
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise


class RunStore:
    def outcomes(self, run_id):
        with tools._session() as session:
            return [dict(call_key=r.call_key, tool=r.tool_name, args=r.args, status=r.status, result=r.result)
                    for r in session.scalars(select(ToolLedger).where(ToolLedger.run_id == run_id))]

    def _lease(self, session, run_id, token):
        run = session.get(AgentRun, run_id)
        if run is None or run.lease_token != token or run.status != "RUNNING":
            raise AgentError("RUN_LEASE_LOST", "任务执行权已失效，停止继续操作", retryable=False)
        return run

    def conversation(self, session_id):
        with tools._session() as session:
            conversation = session.get(Conversation, session_id)
            if conversation is None:
                return {"state": {}, "messages": [], "active_run_id": None}
            messages = session.scalars(select(ConversationMessage).where(ConversationMessage.session_id == session_id)
                                       .order_by(ConversationMessage.message_id.desc()).limit(40)).all()
            run_ids = {message.run_id for message in messages if message.role == "assistant"}
            results = {
                run.run_id: run.result
                for run in session.scalars(select(AgentRun).where(AgentRun.run_id.in_(run_ids))).all()
            } if run_ids else {}
            return {"state": conversation.state, "active_run_id": conversation.active_run_id,
                    "messages": [{"role": m.role, "content": m.content, "run_id": m.run_id,
                                  **({"result": results.get(m.run_id)} if m.role == "assistant" else {})}
                                 for m in reversed(messages)]}

    def claim(self, session_id, request_id, request, limits, *, resume_id=None, action_id=None,
              execution_contract=None):
        requested_contract = json_value(execution_contract) if execution_contract is not None else None
        with transaction() as session:
            conversation = session.get(Conversation, session_id)
            if conversation is None:
                conversation = Conversation(session_id=session_id, state={})
                session.add(conversation)
                session.flush()
            run = session.get(AgentRun, resume_id) if resume_id else session.scalar(
                select(AgentRun).where(AgentRun.session_id == session_id, AgentRun.request_id == request_id))
            if resume_id and (run is None or run.session_id != session_id):
                raise AgentError("RUN_NOT_FOUND", "没有找到该会话的任务", retryable=False)
            if run is not None:
                if not resume_id and run.request != request:
                    raise AgentError("REQUEST_ID_CONFLICT", "同一请求编号不能用于不同的需求", retryable=False)
                checkpoint = session.get(Checkpoint, run.run_id)
                snapshot = restore_runtime_snapshot(checkpoint.payload, run.run_id)
                if (run.result is None and snapshot.get("execution_contract") is not None
                        and requested_contract is None):
                    raise AgentError(
                        "EXECUTION_CONTRACT_REQUIRED",
                        "继续未完成任务前必须校验原始执行契约",
                        retryable=False,
                    )
                if requested_contract is not None and not _contracts_match(
                        _stored_execution_contract(snapshot), requested_contract,
                        require_exact_execution=run.result is None):
                    stored_contract = _stored_execution_contract(snapshot)
                    stored_harness = stored_contract.get("harness") or {}
                    requested_harness = requested_contract.get("harness") or {}
                    implementation_changed = (
                        resume_id
                        and stored_contract.get("execution_mode") == requested_contract.get("execution_mode")
                        and stored_harness.get("profile_id") == requested_harness.get("profile_id")
                        and stored_harness.get("profile_sha256") == requested_harness.get("profile_sha256")
                        and not stored_harness.get("legacy_profile_reference")
                        and not requested_harness.get("legacy_profile_reference")
                    )
                    if implementation_changed:
                        from src.harness.registry import harness_execution_matches
                        implementation_changed = not harness_execution_matches(
                            stored_harness, requested_harness)
                    code = ("HARNESS_IMPLEMENTATION_MISMATCH" if implementation_changed else
                            "HARNESS_PROFILE_CONFLICT" if resume_id else "REQUEST_CONFIGURATION_CONFLICT")
                    message = ("当前源码无法恢复任务开始时的机制实现" if implementation_changed else
                               "恢复请求的 Harness Profile 或执行模式与原任务不一致" if resume_id else
                               "同一请求编号不能切换 Harness Profile 或执行模式")
                    raise AgentError(code, message, retryable=False)
                if not resume_id and run.limits != limits:
                    raise AgentError("REQUEST_CONFIGURATION_CONFLICT",
                                     "同一请求编号不能改变运行预算", retryable=False)
                if run.result is not None:
                    return {"cached": run.result, "run_id": run.run_id}
                if run.lease_token and run.lease_until and run.lease_until > utcnow():
                    raise AgentError("RUN_IN_PROGRESS", f"任务 {run.run_id} 正在执行", retryable=True)
            if conversation.active_run_id and (run is None or conversation.active_run_id != run.run_id):
                raise AgentError("CONVERSATION_BUSY", f"请先恢复任务 {conversation.active_run_id}", retryable=True)
            if run is None:
                pending = (conversation.state or {}).get("pending_action")
                if action_id is not None and action_id != (pending or {}).get("action_id"):
                    raise AgentError("APPROVAL_STALE", "待确认操作发生变化，请重新查看当前会话", retryable=False)
                run = AgentRun(run_id=uuid.uuid4().hex, session_id=session_id, request_id=request_id, request=request,
                               status="RUNNING", created_at=utcnow(), limits=limits)
                session.add(run)
                session.flush()
                history = session.scalars(select(ConversationMessage).where(ConversationMessage.session_id == session_id)
                                          .order_by(ConversationMessage.message_id.desc()).limit(40)).all()
                snapshot = new_runtime_snapshot(
                    run.run_id,
                    conversation=conversation.state,
                    history=[{"role": m.role, "content": m.content} for m in reversed(history)],
                    execution_contract=requested_contract,
                )
                session.add(Checkpoint(run_id=run.run_id, payload=snapshot))
                session.add(ConversationMessage(session_id=session_id, run_id=run.run_id, role="user", content=request))
            run.status = "RUNNING"
            run.lease_token = uuid.uuid4().hex
            run.lease_until = utcnow() + timedelta(seconds=run.limits.get("max_seconds", 30) + 15)
            conversation.active_run_id = run.run_id
            checkpoint = session.get(Checkpoint, run.run_id)
            snapshot = restore_runtime_snapshot(checkpoint.payload, run.run_id)
            if snapshot.get("status") == "INTERRUPTED":
                snapshot = advance_runtime_snapshot(
                    snapshot, run.run_id, status="RUNNING", current_node=snapshot.get("current_node", "INTAKE")
                )
                checkpoint.payload = json_value(snapshot)
            pending = (snapshot.get("conversation") or {}).get("pending_action")
            if pending and self._is_confirmation(run.request, pending):
                action = session.get(PendingAction, pending.get("action_id"))
                if action is None or action.session_id != session_id:
                    raise AgentError("INVALID_APPROVAL", "待确认操作不存在或不属于当前会话", retryable=False)
                # A missing payment method asks for clarification without consuming approval.
                from src.direct_executor import _payment_method
                executable = pending.get("type") != "PROCESS_PAYMENT" or _payment_method(run.request) or pending.get("method")
                if action.status == "OPEN" and executable:
                    chosen_method = _payment_method(run.request)
                    if pending.get("type") == "PROCESS_PAYMENT" and chosen_method:
                        pending = {**pending, "method": chosen_method}
                        if pending.get("tool") == "process_payment" and isinstance(pending.get("args"), dict):
                            from src.runtime import ToolExecutor
                            pending["args"] = {**pending["args"], "method": chosen_method}
                            pending["action_fingerprint"] = ToolExecutor.fingerprint("process_payment", pending["args"])
                        action.payload = dict(pending)
                        snapshot = {**snapshot, "conversation": {**snapshot.get("conversation", {}), "pending_action": pending}}
                        snapshot = seal_runtime_snapshot(snapshot, run.run_id)
                        checkpoint.payload = json_value(snapshot)
                    action.status = "APPROVED"
                    action.approved_run_id = run.run_id
                elif action.status != "OPEN" and action.approved_run_id != run.run_id:
                    raise AgentError("APPROVAL_ALREADY_USED", "该确认已由另一项任务使用", retryable=False)
            return {"run_id": run.run_id, "session_id": session_id, "request": run.request,
                    "token": run.lease_token, "limits": run.limits, "snapshot": snapshot,
                    "events": [step.payload for step in session.scalars(select(AgentStep).where(AgentStep.run_id == run.run_id)
                                                                      .order_by(AgentStep.sequence))]}

    @staticmethod
    def _is_confirmation(request, pending=None):
        from src.intake import APPROVAL_CONFIRMED, parse_approval_response
        return parse_approval_response(request, pending) == APPROVAL_CONFIRMED

    def save_checkpoint(self, run_id, token, snapshot, event=None):
        with transaction() as session:
            self._lease(session, run_id, token)
            checkpoint = session.get(Checkpoint, run_id)
            stored = restore_runtime_snapshot(checkpoint.payload, run_id)
            incoming = seal_runtime_snapshot(snapshot, run_id)
            if incoming["state_version"] != stored["state_version"]:
                raise AgentError("CHECKPOINT_CONFLICT", "任务状态已被其他执行者更新", retryable=False)
            next_snapshot = advance_runtime_snapshot(
                incoming,
                run_id,
                status="RUNNING",
                current_node=incoming.get("phase", incoming.get("current_node", "INTAKE")),
            )
            checkpoint.payload = json_value(next_snapshot)
            snapshot.clear()
            snapshot.update(next_snapshot)
            if event is not None:
                session.add(AgentStep(run_id=run_id, sequence=event["sequence"], payload=json_value(event)))

    def ledger_key(self, run_id, tool_name, args, approval=None):
        scope = (approval or {}).get("action_id") or run_id
        data = [scope, tool_name, args if not approval else None]
        return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def ledger(self, call_key, tool_name, args):
        with tools._session() as session:
            row = session.get(ToolLedger, call_key)
            if row is None:
                return None
            if row.tool_name != tool_name or row.args != args:
                raise AgentError("IDEMPOTENCY_CONFLICT", "同一操作编号的工具或参数发生变化", retryable=False)
            return {"status": row.status, "result": row.result}

    def save_ledger(self, run_id, token, key, name, args, status, result=None):
        with transaction() as session:
            self._lease(session, run_id, token)
            row = session.get(ToolLedger, key)
            if row is None:
                row = ToolLedger(call_key=key, run_id=run_id, tool_name=name, args=args, status=status)
                session.add(row)
            row.status = status
            row.result = json_value(result)

    def execute_write(self, run_id, token, key, name, args, approval, invoke):
        with transaction() as session:
            self._lease(session, run_id, token)
            previous = session.get(ToolLedger, key)
            if previous is not None:
                if previous.args != args or previous.tool_name != name:
                    raise AgentError("IDEMPOTENCY_CONFLICT", "该操作已使用其他参数执行", retryable=False)
                if previous.status == "SUCCEEDED":
                    return previous.result
            action = session.get(PendingAction, approval.get("action_id")) if approval else None
            if action is None or action.approved_run_id != run_id or action.status != "APPROVED":
                raise AgentError("INVALID_APPROVAL", "操作缺少当前任务的有效确认", retryable=False)
            expected = "PROCESS_PAYMENT" if name == "process_payment" else "INITIATE_REFUND"
            if action.payload.get("type") != expected or action.payload.get("order_id") != args.get("order_id"):
                raise AgentError("INVALID_APPROVAL", "确认内容与实际操作不一致", retryable=False)
            with tools.use_transaction(session):
                if name == "process_payment" and action.payload.get("amount") is not None:
                    payment = tools.get_payment_status(args["order_id"])
                    if payment.get("amount") != action.payload["amount"]:
                        raise AgentError("APPROVAL_STALE", "支付金额发生变化，请重新核验并确认", retryable=False)
                result = invoke()
            if previous is None:
                session.add(ToolLedger(call_key=key, run_id=run_id, tool_name=name, args=args,
                                       status="SUCCEEDED", result=json_value(result)))
            else:
                previous.status, previous.result = "SUCCEEDED", json_value(result)
            action.status = "COMPLETED"
            return result

    def finish(self, run_id, token, result, state):
        with transaction() as session:
            run = self._lease(session, run_id, token)
            events = [step.payload for step in session.scalars(select(AgentStep).where(
                AgentStep.run_id == run_id).order_by(AgentStep.sequence))]
            terminal = {'run_id': run_id, 'sequence': (events[-1]['sequence'] if events else 0) + 1,
                        'timestamp': datetime.now(UTC).isoformat(), 'type': 'terminal',
                        'status': result['status'], 'reason': result.get('terminal_reason'), 'error': result.get('error')}
            if not any(e.get('type') == 'terminal' for e in events):
                session.add(AgentStep(run_id=run_id, sequence=terminal['sequence'], payload=json_value(terminal)))
                events.append(terminal)
            result['events'] = events
            conversation = session.get(Conversation, run.session_id)
            pending = result.get("pending_action")
            if pending:
                pending = dict(pending)
                pending.setdefault("action_id", uuid.uuid4().hex)
                action = session.get(PendingAction, pending["action_id"])
                if action is None:
                    session.add(PendingAction(action_id=pending["action_id"], session_id=run.session_id,
                                              source_run_id=run_id, status="OPEN", payload=pending))
                result["pending_action"] = pending
                state["pending_action"] = pending
            old_action = (conversation.state or {}).get("pending_action")
            if old_action and (not pending or pending.get("action_id") != old_action.get("action_id")):
                action = session.get(PendingAction, old_action.get("action_id"))
                if action and action.status in {"OPEN", "APPROVED"}:
                    action.status = "CANCELLED" if result.get("status") == "CANCELLED" else "CLOSED"
            run.result = json_value(result)
            run.status = result["status"]
            run.lease_token, run.lease_until = None, None
            conversation.state = json_value(state)
            conversation.active_run_id = None
            session.add(ConversationMessage(session_id=run.session_id, run_id=run_id, role="assistant", content=result["reply"]))
            checkpoint = session.get(Checkpoint, run_id)
            checkpoint.payload = json_value(advance_runtime_snapshot(
                checkpoint.payload,
                run_id,
                status=result["status"],
                current_node="TERMINAL",
                terminal_reason=result.get("terminal_reason"),
                sequence=events[-1]['sequence'],
            ))
        return result

    def interrupt(self, run_id, token):
        with transaction() as session:
            run = self._lease(session, run_id, token)
            run.status, run.lease_token, run.lease_until = "INTERRUPTED", None, None
            checkpoint = session.get(Checkpoint, run_id)
            checkpoint.payload = json_value(advance_runtime_snapshot(
                checkpoint.payload,
                run_id,
                status="INTERRUPTED",
                current_node=checkpoint.payload.get("current_node", checkpoint.payload.get("phase", "INTAKE")),
            ))

    def run(self, run_id):
        with tools._session() as session:
            run = session.get(AgentRun, run_id)
            if run is None:
                return None
            checkpoint = session.get(Checkpoint, run_id)
            return {"run_id": run_id, "session_id": run.session_id, "request": run.request, "status": run.status,
                    "created_at": run.created_at.replace(tzinfo=UTC).isoformat(), "result": run.result,
                    "checkpoint": checkpoint.payload if checkpoint else {},
                    "events": [step.payload for step in session.scalars(select(AgentStep).where(AgentStep.run_id == run_id).order_by(AgentStep.sequence))]}
