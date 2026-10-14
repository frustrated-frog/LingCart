"""One durable conversation entry point shared by HTTP and the workspace."""
import uuid

from langchain_core.messages import AIMessage, HumanMessage

from src.errors import AgentError
from src.memory import ConversationMemory
from src.orchestrator import run_orchestrator
from src.persistence import RunStore
from src.runtime import BudgetLimits, RunRuntime, runtime_scope


def load_conversation(session_id, store=None):
    store = store or RunStore()
    saved = store.conversation(session_id)
    state = saved["state"] or {}
    memory = ConversationMemory(last_order_id=state.get("last_order_id"), pending_action=state.get("pending_action"))
    messages = []
    for message in saved["messages"]:
        memory.turns.append((HumanMessage if message["role"] == "user" else AIMessage)(content=message["content"]))
        item = dict(message)
        if message["role"] == "assistant":
            item["result"] = (store.run(message["run_id"]) or {}).get("result") or {}
        messages.append(item)
    return {**saved, "memory": memory, "messages": messages}


def run_conversation(message, session_id, *, request_id=None, action_id=None, limits=None, resume_id=None,
                     store=None, runner=None):
    store = store or RunStore()
    claim = store.claim(session_id, request_id or uuid.uuid4().hex, message,
                        (limits or BudgetLimits()).__dict__, resume_id=resume_id, action_id=action_id)
    if "cached" in claim:
        # A retried old request never overwrites the current conversation state.
        return {**claim["cached"], "cached": True}
    snapshot = claim["snapshot"]
    state = dict(snapshot.get("conversation") or {})
    history = [(HumanMessage if item["role"] == "user" else AIMessage)(content=item["content"])
               for item in snapshot.get("history", [])]
    pending = state.get("pending_action")
    approval = pending if pending and store._is_confirmation(claim["request"], pending) else None
    context = RunRuntime(claim["run_id"], BudgetLimits(**claim["limits"]), store=store, token=claim["token"],
                         snapshot=snapshot, events=claim["events"], approval=approval)
    try:
        with runtime_scope(context):
            kwargs = {"history": history}
            if pending:
                kwargs["pending_action"] = pending
            result = dict((runner or run_orchestrator)(claim["request"], **kwargs))
            result["run_id"] = claim["run_id"]
            result.setdefault("status", "SUCCEEDED")
            result.setdefault("terminal_reason", "COMPLETED")
            result.setdefault("events", context.events)
            result.setdefault("trace", [])
            result.setdefault("usage", {})
            result.setdefault("pending_action", None)
            result.setdefault("route", result.get("agent", "runtime").upper())
            memory = ConversationMemory(last_order_id=state.get("last_order_id"))
            memory.update_last_order_id(result["trace"])
            state.update(last_order_id=memory.last_order_id, pending_action=result["pending_action"])
            state["clarification"] = result.get("intake") if result.get("terminal_reason") == "MISSING_ORDER_ID" else None
            return store.finish(claim["run_id"], claim["token"], result, state)
    except BaseException:
        # Simulated/process interruptions retain the saved prefix for explicit resume.
        # An expired/replaced owner cannot change the newer owner's run.
        try:
            store.interrupt(claim["run_id"], claim["token"])
        except AgentError:
            pass
        raise


def resume_run(session_id, run_id, **kwargs):
    return run_conversation("", session_id, resume_id=run_id, **kwargs)
