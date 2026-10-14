"""FastAPI backend for the AI Customer Support Agent.

Exposes the same multi-agent orchestration logic as the Streamlit UI over a
clean JSON API, so the system can be consumed by any frontend, mobile app, or
another service. Endpoints:

    GET  /health   liveness check
    POST /chat     send a message, get back the agent's reply + trace + usage
"""

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.logging_config import configure_logging
from src.memory import ConversationMemory
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

# In-memory conversation store keyed by session_id. A real deployment would
# back this with Redis/a database; in-memory is fine for a single-process demo.
_sessions: dict[str, ConversationMemory] = {}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The customer's message")
    session_id: str | None = Field(None, description="Omit to start a new conversation")


class ToolCall(BaseModel):
    agent: str
    tool: str
    args: dict
    result: dict | str | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


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
    memory = _sessions.setdefault(session_id, ConversationMemory())

    augmented_message = memory.context_note() + req.message
    try:
        orchestrator_kwargs = {"history": memory.turns}
        if memory.pending_action is not None:
            orchestrator_kwargs["pending_action"] = memory.pending_action
        result = run_orchestrator(augmented_message, **orchestrator_kwargs)
    except Exception as exc:  # pragma: no cover - orchestrator already retries + escalates internally
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    memory.add_turn(req.message, result["reply"])
    memory.update_last_order_id(result["trace"])
    memory.update_pending_action(result)

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
        pending_action=result.get("pending_action"),
        error=result.get("error"),
        latency_ms=result.get("latency_ms"),
        usage=Usage(**(result.get("usage") or {})),
        cached=result.get("cached", False),
    )
