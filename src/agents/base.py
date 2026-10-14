import json
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.errors import AgentError, ErrorCode, classify_exception
from src.llm import extract_usage, get_llm
from src.runtime import TraceRecorder, ToolExecutor

_EMPTY_USAGE = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def as_text(content) -> str:
    """Gemini sometimes returns content as a list of typed blocks instead of a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def sum_usage(*usages: dict) -> dict:
    total = dict(_EMPTY_USAGE)
    for usage in usages:
        for key in total:
            total[key] += (usage or {}).get(key) or 0
    return total


def _serialize_tool_result(result) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


def run_tool_agent(
    agent_name: str,
    system_prompt: str,
    agent_tools: list,
    user_message: str,
    history: list | None = None,
    max_rounds: int = 4,
    max_tool_calls: int = 8,
    run_id: str | None = None,
) -> dict:
    """Bounded tool loop with typed failures and per-call observability."""

    run_id = run_id or uuid.uuid4().hex
    recorder = TraceRecorder(run_id)
    tools_by_name = {tool.name: tool for tool in agent_tools}
    llm = get_llm().bind_tools(agent_tools)

    messages: list = [SystemMessage(content=system_prompt)]
    messages += history or []
    messages.append(HumanMessage(content=user_message))

    trace: list[dict] = []
    usage = dict(_EMPTY_USAGE)
    first_model_started = time.perf_counter()
    ai_msg: AIMessage = llm.invoke(messages)
    recorder.record(
        "model_call",
        purpose="agent_decision",
        agent=agent_name,
        status="SUCCEEDED",
        duration_ms=round((time.perf_counter() - first_model_started) * 1000, 2),
        usage=extract_usage(ai_msg),
    )
    usage = sum_usage(usage, extract_usage(ai_msg))
    messages.append(ai_msg)

    rounds = 0
    tool_calls = 0
    seen_calls: set[str] = set()
    errors: list[dict] = []
    successful_tools = 0
    terminal_reason = "COMPLETED"

    while ai_msg.tool_calls and rounds < max_rounds:
        repeated = False
        for call in ai_msg.tool_calls:
            tool_name = call.get("name", "")
            args = call.get("args")
            call_id = call.get("id") or f"call-{tool_calls + 1}"

            if not isinstance(args, dict):
                error = AgentError(ErrorCode.INVALID_TOOL_ARGUMENTS, "工具参数必须是对象", retryable=False)
                error_payload = {"code": error.code, "message": str(error), "retryable": False}
                errors.append(error_payload)
                recorder.record("error", agent=agent_name, tool=tool_name, args=args, **error_payload)
                messages.append(ToolMessage(content=_serialize_tool_result({"error": error_payload}), tool_call_id=call_id))
                continue

            fingerprint = ToolExecutor.fingerprint(tool_name, args)
            if fingerprint in seen_calls:
                error_payload = {
                    "code": "REPEATED_TOOL_CALL",
                    "message": f"检测到重复工具调用：{tool_name}",
                    "retryable": False,
                }
                errors.append(error_payload)
                recorder.record("error", agent=agent_name, tool=tool_name, args=args, **error_payload)
                terminal_reason = "REPEATED_TOOL_CALL"
                repeated = True
                break
            seen_calls.add(fingerprint)

            if tool_calls >= max_tool_calls:
                error_payload = {
                    "code": ErrorCode.TOOL_BUDGET_EXHAUSTED,
                    "message": "工具调用预算已耗尽",
                    "retryable": False,
                }
                errors.append(error_payload)
                recorder.record("error", agent=agent_name, tool=tool_name, args=args, **error_payload)
                terminal_reason = ErrorCode.TOOL_BUDGET_EXHAUSTED
                repeated = True
                break

            tool_calls += 1
            recorder.record("tool_call", agent=agent_name, tool=tool_name, args=args, call_index=tool_calls)
            started_at = time.perf_counter()

            if tool_name not in tools_by_name:
                error = AgentError(ErrorCode.UNKNOWN_TOOL, f"模型请求了未授权工具：{tool_name}", retryable=False)
                error_payload = {"code": error.code, "message": str(error), "retryable": False}
                errors.append(error_payload)
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                recorder.record(
                    "error", agent=agent_name, tool=tool_name, args=args, duration_ms=duration_ms, **error_payload
                )
                trace.append(
                    {
                        "run_id": run_id,
                        "agent": agent_name,
                        "tool": tool_name,
                        "args": args,
                        "result": {"error": str(error), "code": error.code},
                        "status": "FAILED",
                        "duration_ms": duration_ms,
                    }
                )
                messages.append(ToolMessage(content=_serialize_tool_result(trace[-1]["result"]), tool_call_id=call_id))
                continue

            try:
                result = tools_by_name[tool_name].invoke(args)
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                tool_status = "BUSINESS_ERROR" if isinstance(result, dict) and result.get("error") else "SUCCEEDED"
                if tool_status == "SUCCEEDED":
                    successful_tools += 1
                recorder.record(
                    "tool_result",
                    agent=agent_name,
                    tool=tool_name,
                    args=args,
                    result=result,
                    status=tool_status,
                    duration_ms=duration_ms,
                )
            except Exception as exc:  # noqa: BLE001 - preserve typed tool failure in the run
                failure = classify_exception(exc, source="tool")
                duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
                error_payload = {
                    "code": failure.code,
                    "message": str(failure),
                    "retryable": failure.retryable,
                }
                errors.append(error_payload)
                result = {"error": str(failure), "code": failure.code}
                tool_status = "FAILED"
                recorder.record(
                    "error",
                    agent=agent_name,
                    tool=tool_name,
                    args=args,
                    duration_ms=duration_ms,
                    **error_payload,
                )

            trace.append(
                {
                    "run_id": run_id,
                    "agent": agent_name,
                    "tool": tool_name,
                    "args": args,
                    "result": result,
                    "status": tool_status,
                    "duration_ms": duration_ms,
                }
            )
            messages.append(ToolMessage(content=_serialize_tool_result(result), tool_call_id=call_id))

        if repeated:
            break

        model_started = time.perf_counter()
        try:
            ai_msg = llm.invoke(messages)
        except Exception as exc:  # do not replay successful side effects or read calls
            failure = classify_exception(exc, source="model")
            error_payload = {"code": failure.code, "message": str(failure), "retryable": failure.retryable}
            errors.append(error_payload)
            recorder.record(
                "error",
                agent=agent_name,
                source="model",
                duration_ms=round((time.perf_counter() - model_started) * 1000, 2),
                **error_payload,
            )
            terminal_reason = "MODEL_FAILED_AFTER_TOOL_RESULT" if trace else failure.code
            break
        recorder.record(
            "model_call",
            purpose="compose_response",
            agent=agent_name,
            status="SUCCEEDED",
            duration_ms=round((time.perf_counter() - model_started) * 1000, 2),
            usage=extract_usage(ai_msg),
        )
        usage = sum_usage(usage, extract_usage(ai_msg))
        messages.append(ai_msg)
        rounds += 1

    if ai_msg.tool_calls and rounds >= max_rounds:
        terminal_reason = "MODEL_ROUND_BUDGET_EXHAUSTED"

    reply = as_text(ai_msg.content).strip()
    if not reply and trace:
        last_result = trace[-1]["result"]
        reply = last_result if isinstance(last_result, str) else _serialize_tool_result(last_result)

    status = "SUCCEEDED"
    if errors and successful_tools == 0:
        status = "FAILED"
    elif errors and terminal_reason == "REPEATED_TOOL_CALL" and trace:
        # A duplicate after a real result is a controlled stop; return the
        # verified observation instead of failing or spending more quota.
        status = "SUCCEEDED"
    if not reply:
        reply = "任务没有产生可验证的结果。"
        status = "FAILED"
        terminal_reason = "EMPTY_MODEL_RESPONSE"

    recorder.record("terminal", status=status, reason=terminal_reason, error=errors[-1] if errors else None)
    return {
        "run_id": run_id,
        "reply": reply,
        "trace": trace,
        "events": recorder.events,
        "messages": messages,
        "agent": agent_name,
        "usage": usage,
        "status": status,
        "terminal_reason": terminal_reason,
        "error": errors[-1] if errors else None,
    }
