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


def utcnow():
    return datetime.now(UTC).replace(tzinfo=None)


def json_value(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


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
            return {"state": conversation.state, "active_run_id": conversation.active_run_id,
                    "messages": [{"role": m.role, "content": m.content, "run_id": m.run_id} for m in reversed(messages)]}

    def claim(self, session_id, request_id, request, limits, *, resume_id=None, action_id=None):
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
                snapshot = {"phase": "INTAKE", "conversation": conversation.state,
                            "history": [{"role": m.role, "content": m.content} for m in reversed(history)],
                            "budget": {}, "models": {}, "sequence": 0}
                session.add(Checkpoint(run_id=run.run_id, payload=snapshot))
                session.add(ConversationMessage(session_id=session_id, run_id=run.run_id, role="user", content=request))
            run.status = "RUNNING"
            run.lease_token = uuid.uuid4().hex
            run.lease_until = utcnow() + timedelta(seconds=run.limits.get("max_seconds", 30) + 15)
            conversation.active_run_id = run.run_id
            checkpoint = session.get(Checkpoint, run.run_id)
            snapshot = checkpoint.payload
            pending = (snapshot.get("conversation") or {}).get("pending_action")
            if pending and self._is_confirmation(run.request):
                action = session.get(PendingAction, pending.get("action_id"))
                if action is None or action.session_id != session_id:
                    raise AgentError("INVALID_APPROVAL", "待确认操作不存在或不属于当前会话", retryable=False)
                # A missing payment method asks for clarification without consuming approval.
                from src.direct_executor import _payment_method
                executable = pending.get("type") != "PROCESS_PAYMENT" or _payment_method(run.request) or pending.get("method")
                if action.status == "OPEN" and executable:
                    action.status = "APPROVED"
                    action.approved_run_id = run.run_id
                elif action.status != "OPEN" and action.approved_run_id != run.run_id:
                    raise AgentError("APPROVAL_ALREADY_USED", "该确认已由另一项任务使用", retryable=False)
            return {"run_id": run.run_id, "session_id": session_id, "request": run.request,
                    "token": run.lease_token, "limits": run.limits, "snapshot": snapshot,
                    "events": [step.payload for step in session.scalars(select(AgentStep).where(AgentStep.run_id == run.run_id)
                                                                      .order_by(AgentStep.sequence))]}

    @staticmethod
    def _is_confirmation(request):
        from src.intake import APPROVAL_CONFIRMED, parse_approval_response
        return parse_approval_response(request) == APPROVAL_CONFIRMED

    def save_checkpoint(self, run_id, token, snapshot, event=None):
        with transaction() as session:
            self._lease(session, run_id, token)
            checkpoint = session.get(Checkpoint, run_id)
            checkpoint.payload = json_value(snapshot)
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
            session.get(Checkpoint, run_id).payload = {**session.get(Checkpoint, run_id).payload, "phase": "TERMINAL"}
        return result

    def interrupt(self, run_id, token):
        with transaction() as session:
            run = self._lease(session, run_id, token)
            run.status, run.lease_token, run.lease_until = "INTERRUPTED", None, None

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
