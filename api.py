"""FastAPI backend for the AI Customer Support Agent.

Exposes the same multi-agent orchestration logic as the Streamlit UI over a
clean JSON API, so the system can be consumed by any frontend, mobile app, or
another service. Endpoints:

    GET  /health   liveness check
    POST /chat     durable message execution with request deduplication
    GET /conversations/{session_id}  saved state and active run
    GET /runs/{run_id}  saved checkpoint, result and events
    POST /runs/{run_id}/resume  explicitly resume interrupted work
"""

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.logging_config import configure_logging
from src.durable import run_conversation, resume_run
from src.errors import AgentError
from src.persistence import RunStore
from src.orchestrator import run_orchestrator

configure_logging()
logger = logging.getLogger("support_agent.api")

app = FastAPI(
    title="AI Customer Support Agent API",
    description=(
        "Multi-agent customer support system: an orchestrator routes each message "
        "to a specialist (order/refund/payment/FAQ) agent that calls real backend tools."
    ),
    version="1.0.0",
)




class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The customer's message")
    session_id: str | None = Field(None, max_length=64, description="Omit to start a new conversation")
    request_id: str | None = Field(None, min_length=1, max_length=64, description="Reuse when retrying the same request")
    action_id: str | None = Field(None, min_length=1, max_length=64, description="Current pending action shown to the user")


class ToolCall(BaseModel):
    agent: str
    tool: str
    args: dict
    result: dict | str | None = None


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ChatResponse(BaseModel):
    session_id: str
    run_id: str
    reply: str
    agent: str
    route: str
    status: str
    terminal_reason: str
    trace: list[ToolCall]
    events: list[dict] = Field(default_factory=list)
    intake: dict = Field(default_factory=dict)
    budget: dict = Field(default_factory=dict)
    plan: dict | None = None
    pending_action: dict | None = None
    error: dict | None = None
    latency_ms: float | None = None
    usage: Usage
    cached: bool = False


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "request",
        extra={
            "extra_fields": {
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": elapsed_ms,
            }
        },
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex
    try:
        result = run_conversation(req.message, session_id, request_id=req.request_id,
                                  action_id=req.action_id, runner=run_orchestrator)
    except AgentError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": str(exc), "retryable": exc.retryable}) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"code": "RUN_INTERRUPTED", "session_id": session_id,
                            "run_id": RunStore().conversation(session_id)["active_run_id"],
                            "message": "任务中断，可通过恢复接口继续执行"}) from exc
    return _response(session_id, result)


def _response(session_id, result):
    return ChatResponse(
        session_id=session_id,
        run_id=result.get("run_id", uuid.uuid4().hex),
        reply=result["reply"],
        agent=result["agent"],
        route=result.get("route", result.get("agent", "UNKNOWN").upper()),
        status=result.get("status", "SUCCEEDED"),
        terminal_reason=result.get("terminal_reason", "COMPLETED"),
        trace=result["trace"],
        events=result.get("events") or [],
        intake=result.get("intake") or {},
        budget=result.get("budget") or {},
        plan=result.get("plan"),
        pending_action=result.get("pending_action"),
        error=result.get("error"),
        latency_ms=result.get("latency_ms"),
        usage=Usage(**(result.get("usage") or {})),
        cached=result.get("cached", False),
    )


class ResumeRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)


@app.get("/conversations/{session_id}")
def conversation(session_id: str):
    return RunStore().conversation(session_id)


@app.get("/runs/{run_id}")
def get_run(run_id: str, session_id: str):
    run = RunStore().run(run_id)
    if not run or run["session_id"] != session_id:
        raise HTTPException(404, "没有找到该会话的任务")
    return run


@app.post("/runs/{run_id}/resume", response_model=ChatResponse)
def resume(run_id: str, req: ResumeRequest):
    try:
        return _response(req.session_id, resume_run(req.session_id, run_id, runner=run_orchestrator))
    except AgentError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": str(exc), "retryable": exc.retryable}) from exc
