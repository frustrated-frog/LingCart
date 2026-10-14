"""One shared tool executor for all specialist agents."""
import json
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.errors import AgentError, classify_exception
from src.llm import extract_usage, get_llm
from src.loop_guard import LoopAction, ProgressAwareLoopGuard
from src.runtime import ApprovalRequired, BudgetLimits, RunRuntime, ToolExecutor, TraceRecorder, current_runtime, runtime_scope

_EMPTY_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") if isinstance(block, dict) and block.get("type") == "text" else block if isinstance(block, str) else "" for block in content)
    return str(content)


def sum_usage(*usages):
    total = dict(_EMPTY_USAGE)
    for usage in usages:
        for key in total:
            value = (usage or {}).get(key)
            total[key] = None if value is None or total[key] is None else total[key] + value
    return total


def _serialize_tool_result(result):
    return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)


def _append_protocol_result(messages, recorder, call, call_id, status, result, *, agent_name, reason=None):
    payload = result if result is not None else {
        "status": status,
        "reason": reason,
        "message": "前序工具已触发运行时收敛，本调用未执行",
    }
    messages.append(ToolMessage(content=_serialize_tool_result(payload), tool_call_id=call_id))
    recorder.record(
        "tool_protocol",
        agent=agent_name,
        tool=call.get("name", ""),
        tool_call_id=call_id,
        status=status,
        reason=reason,
        action_fingerprint=ToolExecutor.fingerprint(call.get("name", ""), call.get("args")),
    )


def _cancel_remaining_calls(messages, recorder, calls, start, *, agent_name, round_index, reason):
    for index, call in enumerate(calls[start:], start=start):
        call_id = call.get("id") or f"call-{round_index}-{index + 1}"
        _append_protocol_result(
            messages,
            recorder,
            call,
            call_id,
            "CANCELLED_BY_RUNTIME",
            None,
            agent_name=agent_name,
            reason=reason,
        )


def run_tool_agent(agent_name, system_prompt, agent_tools, user_message, history=None,
                   max_rounds=4, max_tool_calls=8, run_id=None):
    run_id = run_id or uuid.uuid4().hex
    active = current_runtime()
    if active is None:
        context = RunRuntime(run_id, BudgetLimits(max_model_calls=max_rounds + 1, max_tool_calls=max_tool_calls))
        with runtime_scope(context):
            return _run_loop(agent_name, system_prompt, agent_tools, user_message, history, max_rounds, max_tool_calls, run_id)
    return _run_loop(agent_name, system_prompt, agent_tools, user_message, history, max_rounds, max_tool_calls, run_id)


def _run_loop(agent_name, system_prompt, agent_tools, user_message, history, max_rounds, max_tool_calls, run_id):
    context = current_runtime()
    recorder = TraceRecorder(run_id)
    executor = ToolExecutor({tool.name: tool for tool in agent_tools}, recorder, max_calls=max_tool_calls)
    decision_model = get_llm(purpose="agent")
    final_model = get_llm(purpose="final_response")
    bound_model = decision_model.bind_tools(agent_tools)
    messages = [SystemMessage(content=system_prompt), *(history or []), HumanMessage(content=user_message)]
    trace, errors, loop_guard = [], [], ProgressAwareLoopGuard()
    usage = dict(_EMPTY_USAGE)
    rounds = 0
    reason, status, reply, pending = "COMPLETED", "SUCCEEDED", "", None
    schemas = [tool.args_schema.model_json_schema() for tool in agent_tools]
    while rounds <= max_rounds:
        final_round = rounds == max_rounds or context.counters["model_calls"] >= context.limits.max_model_calls - 1 or executor.call_count >= max_tool_calls or context.counters["tool_calls"] >= context.limits.max_tool_calls
        cached_next = context.models.get(str(context.model_cursor + 1))
        if cached_next:
            final_round = json.loads(cached_next["input"])[0] == "final_response"
        try:
            response = context.model_call(final_model if final_round else bound_model, messages,
                                          purpose="final_response" if final_round else "agent_decision", agent=agent_name,
                                          allow_tools=not final_round, schemas=[] if final_round else schemas)
        except AgentError as error:
            reason = error.code if "BUDGET" in error.code or not trace else "MODEL_FAILED_AFTER_TOOL_RESULT"
            errors.append({"code": error.code, "message": str(error), "retryable": error.retryable})
            status = "PARTIAL" if trace else "FAILED"
            break
        usage = sum_usage(usage, extract_usage(response))
        # Cache-compatible AIMessage history even for deterministic test doubles.
        calls = getattr(response, "tool_calls", [])
        try:
            message = AIMessage(content=response.content, tool_calls=calls)
        except (ValueError, TypeError):
            errors.append({"code": "MODEL_FORMAT_ERROR", "message": "模型返回无效的工具调用格式", "retryable": False})
            reason, status, reply = "MODEL_FORMAT_ERROR", "PARTIAL" if trace else "FAILED", ""
            break
        messages.append(message)
        reply = as_text(response.content).strip()
        if not calls:
            break
        if final_round:
            reason, status = context.stop_reason("tool") or "FINAL_TOOL_CALL_FORBIDDEN", "PARTIAL" if trace else "FAILED"
            errors.append({"code": reason, "message": "最后一轮模型禁止再次请求工具", "retryable": False})
            reply = ""
            recorder.record("error", code=reason, message="最后一轮模型禁止再次请求工具", retryable=False)
            _cancel_remaining_calls(
                messages, recorder, calls, 0, agent_name=agent_name, round_index=rounds, reason=reason
            )
            break
        stop = False
        for call_index, call in enumerate(calls):
            name, args = call.get("name", ""), call.get("args")
            call_id = call.get("id") or f"call-{rounds}-{call_index + 1}"
            fingerprint = ToolExecutor.fingerprint(name, args)
            loop_decision = loop_guard.before_call(fingerprint)
            if loop_decision.action == LoopAction.BLOCK:
                reason = "REPEATED_TOOL_CALL"
                recorder.record(
                    "loop_blocked", agent=agent_name, tool=name, code=reason,
                    message="工具调用未产生新进展，已停止循环", retryable=False,
                    occurrence=loop_decision.occurrence,
                    stagnant_results=loop_decision.stagnant_results,
                    action_fingerprint=fingerprint,
                )
                _append_protocol_result(
                    messages, recorder, call, call_id, "CANCELLED_BY_RUNTIME", None,
                    agent_name=agent_name, reason=reason,
                )
                _cancel_remaining_calls(
                    messages, recorder, calls, call_index + 1,
                    agent_name=agent_name, round_index=rounds, reason=reason,
                )
                reply = ""
                stop = True
                break
            if loop_decision.action == LoopAction.WARN:
                recorder.record(
                    "loop_warning", agent=agent_name, tool=name,
                    message="检测到重复工具调用；若结果仍无变化将停止循环",
                    occurrence=loop_decision.occurrence,
                    action_fingerprint=fingerprint,
                )
            prior_event_count = len(recorder.events)
            protocol_status = "FAILED"
            try:
                result, event = executor.execute(name, args)
                protocol_status = event["status"]
                loop_guard.after_result(fingerprint, result)
                trace.append(dict(run_id=run_id, agent=agent_name, tool=name, args=args,
                                  result=result, status=event["status"], duration_ms=event["duration_ms"]))
            except ApprovalRequired as approval:
                for event in recorder.events[prior_event_count:]:
                    if event["type"] == "tool_result":
                        trace.append(dict(run_id=run_id, agent=agent_name, tool=event["tool"], args=event.get("args", {}),
                                          result=event["result"], status=event["status"], duration_ms=event["duration_ms"]))
                pending = approval.pending
                status, reason = "WAITING_APPROVAL", "APPROVAL_REQUIRED"
                reply = f"订单 {pending['order_id']} 的操作需要你明确确认，尚未修改支付或退款状态。"
                _append_protocol_result(
                    messages, recorder, call, call_id, "WAITING_APPROVAL",
                    {"status": "WAITING_APPROVAL", "requires_approval": True},
                    agent_name=agent_name, reason=reason,
                )
                _cancel_remaining_calls(
                    messages, recorder, calls, call_index + 1,
                    agent_name=agent_name, round_index=rounds, reason=reason,
                )
                stop = True
                break
            except AgentError as error:
                payload = {"code": error.code, "message": str(error), "retryable": error.retryable}
                errors.append(payload)
                result = {"error": str(error), "code": error.code}
                loop_guard.after_result(fingerprint, result)
                trace.append(dict(run_id=run_id, agent=agent_name, tool=name, args=args if isinstance(args, dict) else {}, result=result, status="FAILED", duration_ms=None))
                if "BUDGET" in error.code or error.code == "UNCERTAIN_TOOL_OUTCOME":
                    reason, status, stop = error.code, "PARTIAL" if any(t["status"] == "SUCCEEDED" for t in trace) else "FAILED", True
                    reply = ""
                    _append_protocol_result(
                        messages, recorder, call, call_id, "FAILED", result,
                        agent_name=agent_name, reason=reason,
                    )
                    _cancel_remaining_calls(
                        messages, recorder, calls, call_index + 1,
                        agent_name=agent_name, round_index=rounds, reason=reason,
                    )
                    break
            _append_protocol_result(
                messages, recorder, call, call_id, protocol_status,
                result, agent_name=agent_name,
            )
        if stop:
            break
        rounds += 1
    successes = [entry for entry in trace if entry["status"] in {"SUCCEEDED", "REUSED", "BUSINESS_REJECTED"}]
    if status == "SUCCEEDED" and errors:
        status, reason = "PARTIAL" if successes else "FAILED", errors[-1]["code"]
    elif status == "SUCCEEDED" and not successes:
        status, reason, reply = "FAILED", "NO_VERIFIED_TOOL_RESULT", "没有获得可验证的业务结果。"
        errors.append({"code": reason, "message": reply, "retryable": False})
    if not reply:
        reply = _serialize_tool_result(successes[-1]["result"]) if successes else "任务没有产生可验证的结果。"
    recorder.record("terminal", status=status, reason=reason, error=errors[-1] if errors else None)
    return dict(run_id=run_id, reply=reply, trace=trace, events=recorder.events, messages=messages,
                agent=agent_name, usage=usage, status=status, terminal_reason=reason,
                error=errors[-1] if errors else None, pending_action=pending)
