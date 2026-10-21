"""One durable conversation entry point shared by HTTP and the workspace."""
import uuid
import json

from langchain_core.messages import AIMessage, HumanMessage

from src.errors import AgentError
from src.memory import ConversationMemory
from src.orchestrator import run_orchestrator
from src.persistence import RunStore
from src.runtime import BudgetLimits, RunRuntime, runtime_scope


def _trace_identity(item):
    from src import tools
    from src.runtime import tool_spec
    args = item.get('args')
    try:
        spec = tool_spec(item['tool'], getattr(tools, item['tool']))
        if isinstance(args, dict) and not set(args) - set(spec.schema.model_fields):
            args = spec.schema.model_validate(args, strict=True).model_dump()
    except (AgentError, AttributeError, ValueError):
        pass
    status = item.get('status')
    if status == 'REUSED':
        status = 'SUCCEEDED'
    return json.dumps([item.get('tool'), args, item.get('result'), status], sort_keys=True, default=str)


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
    if runner is None and resume_id is None:
        from src.harness import get_runner
        runner = get_runner("baseline-v1")
    if limits is None and runner is not None:
        from src.adaptive import run_adaptive
        if runner is run_adaptive or getattr(runner, "execution_mode", None) == "adaptive":
            limits = BudgetLimits.adaptive()
    store = store or RunStore()
    from src.harness import resolve_claim_runner, runner_execution_contract
    execution_contract = None
    if resume_id is not None and runner is None:
        saved = store.run(resume_id)
        if not saved or saved["session_id"] != session_id:
            raise AgentError("RUN_NOT_FOUND", "没有找到该会话的任务", retryable=False)
        if saved["result"] is None:
            # Verify persisted code identity before claim() can grant a lease.
            # The transaction then compares this same contract atomically.
            runner = resolve_claim_runner(saved)
            execution_contract = runner_execution_contract(runner)
    else:
        execution_contract = runner_execution_contract(runner)
    claim = store.claim(session_id, request_id or uuid.uuid4().hex, message,
                        (limits or BudgetLimits()).__dict__, resume_id=resume_id, action_id=action_id,
                        execution_contract=execution_contract)
    if "cached" in claim:
        # A retried old request never overwrites the current conversation state.
        return {**claim["cached"], "cached": True}
    result = execute_claim(claim, store=store, runner=runner)
    from src.session_control import SessionController
    SessionController().drain(session_id, runner=runner)
    return result


def execute_claim(claim, *, store=None, runner=None):
    store = store or RunStore()
    from src.harness import resolve_claim_runner, unwrap_harness_runner
    runner = resolve_claim_runner(claim, runner)
    harness_runner = unwrap_harness_runner(runner)
    snapshot = claim["snapshot"]
    state = dict(snapshot.get("conversation") or {})
    history = [(HumanMessage if item["role"] == "user" else AIMessage)(content=item["content"])
               for item in snapshot.get("history", [])]
    pending = state.get("pending_action")
    approval = pending if pending and store._is_confirmation(claim["request"], pending) else None
    context = RunRuntime(claim["run_id"], BudgetLimits(**claim["limits"]), store=store, token=claim["token"],
                         snapshot=snapshot, events=claim["events"], approval=approval)
    context.defer_terminal = True
    try:
        with runtime_scope(context):
            if harness_runner:
                harness_runner.bind(context)
            from src.recovery import recover_evidence
            recover_evidence(context)
            kwargs = {"history": history}
            if pending:
                kwargs["pending_action"] = pending
            from src.session_control import ControlSignal, ControlStore, apply_controls
            ControlStore().reopen_controls(context.run_id)
            for _ in range(6):
                try:
                    apply_controls(context)
                    context.require()
                    steering = context.snapshot.get("steering", [])
                    effective = claim["request"] + ("\n运行中补充要求：\n" + "\n".join(steering) if steering else "")
                    from src.preferences import PreferenceStore
                    preferences = context.snapshot.get('preferences')
                    if preferences is None:
                        preferences = PreferenceStore().for_session(claim['session_id'])
                        context.checkpoint(preferences=preferences)
                    context_factory = harness_runner.build_context if harness_runner else None
                    if context_factory:
                        context.context_engine = context_factory(effective, preferences=preferences,
                            observations=context.snapshot.get('context', {}).get('observations', []),
                            revision=context.snapshot.get('goal_revision', 0))
                    else:
                        from src.context_engine import ContextEngine
                        context.context_engine = ContextEngine(effective, preferences=preferences,
                            observations=context.snapshot.get('context', {}).get('observations', []),
                            revision=context.snapshot.get('goal_revision', 0))
                    from src.adaptive import run_adaptive
                    # Autonomous mode receives the actual request plus separate
                    # preference data. Rules must not inject a synthetic hard budget.
                    adaptive_runner = runner is run_adaptive or getattr(runner, "execution_mode", None) == "adaptive"
                    effective = context.context_engine.goal if adaptive_runner else context.context_engine.effective_request()
                    context.checkpoint(context=context.context_engine.payload())
                    if steering:
                        kwargs.pop("pending_action", None)
                        state["pending_action"] = None
                        context.approval = None
                    result = dict((runner or run_orchestrator)(effective, **kwargs))
                    apply_controls(context)
                    if ControlStore().close_controls(context.run_id):
                        break
                except ControlSignal as signal:
                    if signal.kind == "STEER":
                        context.pending_terminal = None
                        context.snapshot.pop("plan_definition", None)
                        context.snapshot.pop("plan", None)
                        context.record("replan_requested", source="user", goal_revision=context.snapshot.get("goal_revision"))
                        continue
                    result = dict(reply="已停止后续步骤，已完成操作及其结果仍保留在记录中。", status="CANCELLED",
                                  terminal_reason="USER_CANCELLED_RUN", agent="runtime", route="CONTROL", pending_action=None,
                                  trace=[dict(agent=e.get("agent", "runtime"), tool=e["tool"], args=e.get("args", {}),
                                              result=e.get("result"), status=e.get("status")) for e in context.events if e["type"] == "tool_result"],
                                  events=context.events)
                    ControlStore().close_controls(context.run_id)
                    break
                except AgentError as error:
                    trace = [dict(agent=e.get('agent', 'runtime'), tool=e['tool'], args=e.get('args', {}),
                                  result=e.get('result'), status=e.get('status')) for e in context.events if e['type'] == 'tool_result']
                    result = dict(reply=str(error), status='PARTIAL' if trace else 'FAILED', terminal_reason=error.code,
                                  agent='runtime', route='CONTROL', pending_action=None, trace=trace,
                                  error={'code': error.code, 'message': str(error), 'retryable': error.retryable})
                    ControlStore().close_controls(context.run_id)
                    break
            else:
                raise AgentError("STEERING_BUDGET_EXHAUSTED", "运行调整次数已耗尽", retryable=False)
            result["run_id"] = claim["run_id"]
            result.setdefault("status", "SUCCEEDED")
            result.setdefault("terminal_reason", "COMPLETED")
            try:
                context.hooks.emit(context, 'verify', {'result': result})
            except AgentError as error:
                result.update(status='PARTIAL' if result.get('trace') else 'FAILED', terminal_reason=error.code,
                              error={'code': error.code, 'message': str(error), 'retryable': False})
            context.defer_terminal = False
            # 终态事件与最终结果由 finish 在同一事务写入。
            result["events"] = list(context.events)
            result.setdefault("trace", [])
            known = {_trace_identity(t) for t in result['trace']}
            prior = []
            for event in context.events:
                if event['type'] != 'tool_result':
                    continue
                item = dict(agent=event.get('agent', 'runtime'), tool=event['tool'], args=event.get('args', {}),
                            result=event.get('result'), status=event.get('status'))
                identity = _trace_identity(item)
                if identity not in known:
                    prior.append(item)
                    known.add(identity)
            result['trace'] = prior + result['trace']
            result.setdefault("usage", {})
            result.setdefault("pending_action", None)
            result.setdefault("route", result.get("agent", "runtime").upper())
            memory = ConversationMemory(last_order_id=state.get("last_order_id"))
            memory.update_last_order_id(result["trace"])
            state.update(last_order_id=memory.last_order_id, pending_action=result["pending_action"])
            state["clarification"] = result.get("intake") if result.get("terminal_reason") == "MISSING_ORDER_ID" else None
            catalog_artifacts = [
                artifact for artifact in result.get("artifacts", [])
                if isinstance(artifact, dict) and artifact.get("artifact_type") == "catalog_query_result"
                and (artifact.get("verification") or {}).get("status") == "VERIFIED"
            ]
            if catalog_artifacts:
                artifact = catalog_artifacts[-1]
                state["catalog_result_set"] = {
                    "result_set_id": artifact["result_set_id"],
                    "source_run_id": result["run_id"],
                    "source_result_set_id": artifact.get("source_result_set_id"),
                    "catalog_snapshot": artifact["catalog_snapshot"],
                    "catalog_data_fingerprint": artifact["catalog_data_fingerprint"],
                    "population_data_fingerprint": artifact["population_data_fingerprint"],
                    "result_data_fingerprint": artifact["result_data_fingerprint"],
                    "query_spec": artifact["query_spec"],
                    "population_ids": artifact["population_ids"],
                    "selected_ids": artifact["selected_ids"],
                    "delivered_ids": artifact["delivered_ids"],
                    "member_ids": artifact["member_ids"],
                    "matched_count": artifact["matched_count"],
                    "delivered_count": artifact["delivered_count"],
                }
            constraint_artifacts = [
                artifact for artifact in result.get("artifacts", [])
                if isinstance(artifact, dict) and artifact.get("artifact_type") == "constraint_plan_result"
                and (artifact.get("verification") or {}).get("status") == "VERIFIED"
            ]
            if constraint_artifacts:
                artifact = constraint_artifacts[-1]
                state["constraint_solution_set"] = {
                    "solution_set_id": artifact["solution_set_id"],
                    "source_run_id": result["run_id"],
                    "version": artifact["version"],
                    "plan_spec": artifact["plan_spec"],
                    "provider_registry_version": artifact["provider_registry_version"],
                    "domains": [{
                        "goal_id": domain["goal_id"],
                        "provider": domain["provider"],
                        "provider_fingerprint": domain["provider_fingerprint"],
                        "candidate_count": domain["candidate_count"],
                    } for domain in artifact["domains"]],
                    "feasibility": artifact["feasibility"],
                    "solution_ids": [solution["solution_id"] for solution in artifact["solutions"]],
                    "search": artifact["search"],
                    "verification": artifact["verification"],
                }
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
