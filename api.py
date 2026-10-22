"""FastAPI backend and production web host for LingCart.

Exposes the multi-agent runtime through a clean JSON API and serves the built
React application in production. Core endpoints:

    GET  /health   liveness check
    GET  /ready    database and model-configuration readiness
    GET  /models/status  sanitized model gateway status
    POST /chat     durable message execution with request deduplication
    GET /conversations/{session_id}  saved state and active run
    GET /runs/{run_id}  saved checkpoint, result and events
    POST /runs/{run_id}/resume  explicitly resume interrupted work
"""

import hmac
import ipaddress
import logging
import os
from pathlib import Path
import time
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from src import tools
from src.after_sales import CaseDetail, CaseListItem, CaseStatus
from src.logging_config import configure_logging
from src.durable import run_conversation, resume_run
from src.errors import AgentError
from src.llm import get_model_status
from src.persistence import RunStore
from src.harness import get_product_runner as get_runner, get_registry, persisted_harness
from src.trace_projector import project_trace

configure_logging()
logger = logging.getLogger("support_agent.api")

app = FastAPI(
    title="AI Customer Support Agent API",
    description=(
        "Multi-agent customer support system: an orchestrator routes each message "
        "to a specialist (order/refund/payment/FAQ) agent that calls real backend tools."
    ),
    version="1.1.0",
)

_development_origins = "http://127.0.0.1:5173,http://localhost:5173"
app.add_middleware(
    CORSMiddleware,
    allow_origins=[item.strip() for item in os.getenv("LINGCART_WEB_ORIGINS", _development_origins).split(",")
                   if item.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _loopback_client(request: Request) -> bool:
    """Only trust a directly connected loopback client, never Host/XFF headers."""

    host = request.client.host if request.client else ""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _local_browser_origin(request: Request) -> bool:
    host = request.url.hostname or ""
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@app.middleware("http")
async def protect_engineering_api(request: Request, call_next):
    """Keep research traces private unless accessed locally or with an operator key.

    The engineering endpoints contain prompts, checkpoints and tool arguments.  A
    configured key therefore switches the API into explicit operator mode; without
    one, only a direct loopback connection is accepted.  Proxy headers are ignored
    deliberately because trusting them requires deployment-specific proxy setup.
    """

    if request.url.path == "/engineering" or request.url.path.startswith("/engineering/"):
        expected = os.getenv("LINGCART_ENGINEERING_KEY", "").strip()
        supplied = request.headers.get("x-engineering-key", "")
        forwarded = any(request.headers.get(name) for name in ("forwarded", "x-forwarded-for", "x-real-ip"))
        allowed = (bool(expected) and hmac.compare_digest(supplied, expected)) or (
            not expected and not forwarded and _loopback_client(request) and _local_browser_origin(request)
        )
        if not allowed:
            return JSONResponse(
                status_code=403,
                content={"detail": {
                    "code": "ENGINEERING_ACCESS_DENIED",
                    "message": "工程控制台仅允许本机访问，或需要有效的 Engineering Key",
                }},
            )
    return await call_next(request)




class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The customer's message")
    session_id: str | None = Field(None, max_length=64, description="Omit to start a new conversation")
    request_id: str | None = Field(None, min_length=1, max_length=64, description="Reuse when retrying the same request")
    action_id: str | None = Field(None, min_length=1, max_length=64, description="Current pending action shown to the user")
    harness_profile: str | None = Field(None, pattern=r"^[a-z][a-z0-9-]*$",
                                        description="Versioned Harness Profile; defaults to baseline-v1")


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
    harness: dict = Field(default_factory=dict)
    plan: dict | None = None
    pending_action: dict | None = None
    error: dict | None = None
    latency_ms: float | None = None
    usage: Usage
    artifacts: list[dict] = Field(default_factory=list)
    cached: bool = False


class ProviderStatus(BaseModel):
    provider: str
    model: str
    slot: int
    custom_base_url: bool
    health: str = "READY"
    failure_count: int = 0
    retry_after_seconds: float = 0.0
    last_failure_kind: str | None = None


class ModelStatusResponse(BaseModel):
    ready: bool
    configured: bool
    live_verified: bool = False
    message: str
    error_code: str | None = None
    purpose: str | None = None
    max_attempts: int | None = None
    request_timeout: float | None = None
    circuit_failure_threshold: int | None = None
    circuit_cooldown_seconds: float | None = None
    providers: list[ProviderStatus] = Field(default_factory=list)
    active: dict | None = None
    last_error: dict | None = None


class ComponentReadiness(BaseModel):
    ready: bool
    message: str
    latency_ms: float | None = None


class ReadinessResponse(BaseModel):
    status: str
    ready: bool
    message: str
    checks: dict[str, ComponentReadiness]


class AfterSalesCaseListResponse(BaseModel):
    cases: list[CaseListItem]


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


@app.get("/models/status", response_model=ModelStatusResponse)
def model_status():
    """返回脱敏后的模型配置与最近调用状态，不发起计费请求。"""

    return get_model_status()


@app.get("/ready", response_model=ReadinessResponse)
def readiness():
    """检查数据库连通性和模型配置；实时模型调用请使用自检脚本。"""

    database = tools.get_database_status()
    model = get_model_status()
    checks = {
        "database": ComponentReadiness(**database),
        "model_gateway": ComponentReadiness(ready=model["ready"], message=model["message"]),
    }
    ready = all(check.ready for check in checks.values())
    payload = ReadinessResponse(
        status="ready" if ready else "not_ready",
        ready=ready,
        message="服务已就绪" if ready else "服务尚未就绪，请检查失败项",
        checks=checks,
    )
    if not ready:
        return JSONResponse(status_code=503, content=payload.model_dump())
    return payload


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex
    try:
        result = run_conversation(req.message, session_id, request_id=req.request_id,
                                  action_id=req.action_id, runner=get_runner(req.harness_profile or "baseline-v1"))
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
        harness=result.get("harness") or {},
        plan=result.get("plan"),
        pending_action=result.get("pending_action"),
        error=result.get("error"),
        latency_ms=result.get("latency_ms"),
        usage=Usage(**(result.get("usage") or {})),
        artifacts=result.get("artifacts") or [],
        cached=result.get("cached", False),
    )


class ResumeRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)


def _product_resume_runner(session_id, run_id):
    saved = RunStore().run(run_id)
    if not saved or saved['session_id'] != session_id:
        raise HTTPException(404, '没有找到该会话的任务')
    harness = persisted_harness(saved)
    return get_runner(harness['profile_id']) if harness else None


@app.get("/conversations/{session_id}")
def conversation(session_id: str):
    return RunStore().conversation(session_id)


@app.get("/runs/{run_id}")
def get_run(run_id: str, session_id: str):
    run = RunStore().run(run_id)
    if not run or run["session_id"] != session_id:
        raise HTTPException(404, "没有找到该会话的任务")
    return run


@app.get("/runs/{run_id}/trace")
def get_run_trace(run_id: str, session_id: str):
    """返回统一的运行轨迹投影，供 Agent Lab 和诊断客户端使用。"""

    run = RunStore().run(run_id)
    if not run or run["session_id"] != session_id:
        raise HTTPException(404, "没有找到该会话的任务")
    return project_trace(run["events"])


@app.post("/runs/{run_id}/resume", response_model=ChatResponse)
def resume(run_id: str, req: ResumeRequest):
    try:
        runner = _product_resume_runner(req.session_id, run_id)
        return _response(req.session_id, resume_run(req.session_id, run_id, runner=runner))
    except AgentError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": str(exc), "retryable": exc.retryable}) from exc


class StartRunRequest(ChatRequest):
    execution_mode: str = Field('auto', pattern='^(auto|adaptive)$')


class ControlRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    message: str = Field('', max_length=12000)
    command_id: str = Field(..., min_length=1, max_length=64, strict=True, pattern=r'\S')
    expected_revision: int = Field(..., ge=0, strict=True)


def _control_error(error):
    return HTTPException(404 if error.code == 'RUN_NOT_FOUND' else 409,
                         detail={'code': error.code, 'message': str(error), 'retryable': error.retryable})


@app.post('/runs', status_code=202)
def start_run(req: StartRunRequest):
    from src.session_control import SessionController
    try:
        runner = get_runner(req.harness_profile) if req.harness_profile else None
        handle = SessionController().start(req.session_id or uuid.uuid4().hex, req.message,
                    request_id=req.request_id, action_id=req.action_id, execution_mode=req.execution_mode,
                    harness_profile=req.harness_profile, runner=runner)
        return JSONResponse(status_code=200 if handle.get('cached') else 202, content=handle)
    except AgentError as error:
        raise _control_error(error) from error


def _enqueue(run_id, req, kind):
    from src.session_control import ControlStore
    try:
        return ControlStore().enqueue(req.session_id, run_id, kind, message=req.message,
                         command_id=req.command_id, expected_revision=req.expected_revision)
    except AgentError as error:
        raise _control_error(error) from error


@app.post('/runs/{run_id}/steer', status_code=202)
def steer_run(run_id: str, req: ControlRequest):
    return _enqueue(run_id, req, 'STEER')


@app.post('/runs/{run_id}/cancel', status_code=202)
def cancel_run(run_id: str, req: ControlRequest):
    return _enqueue(run_id, req, 'CANCEL')


@app.post('/runs/{run_id}/follow-up', status_code=202)
def follow_up_run(run_id: str, req: ControlRequest):
    return _enqueue(run_id, req, 'FOLLOW_UP')


@app.get('/runs/{run_id}/commands')
def run_commands(run_id: str, session_id: str):
    from src.session_control import ControlStore
    try:
        store = ControlStore()
        return {'revision': store.revision(run_id), 'commands': store.commands(session_id, run_id)}
    except AgentError as error:
        raise _control_error(error) from error


@app.post('/conversations/{session_id}/drain', status_code=202)
def drain_queue(session_id: str):
    """进程重启后显式继续已落库的后续任务，不重放中断的写操作。"""
    from src.session_control import SessionController
    return {'next_run': SessionController().drain(session_id)}


@app.get('/harness/profiles')
def harness_profiles():
    """列出 Profile 及产品可用状态；不暴露密钥或模型凭据。"""
    return {'profiles': [{**profile.canonical_payload(), 'profile_sha256': profile.fingerprint,
                          'product_eligible': profile.research.status in {'baseline', 'promoted'}}
                         for profile in get_registry().profiles()]}


@app.get('/engineering/overview')
def engineering_overview():
    """返回工程控制台首页所需的真实、脱敏系统状态。"""
    from src.engineering_console import engineering_overview as build_overview

    return build_overview()


@app.get('/engineering/runs')
def engineering_runs(limit: int = Query(100, ge=1, le=500), status: str | None = None,
                     profile_id: str | None = None):
    from src.engineering_console import run_catalog

    runs = run_catalog(limit=limit, include_result=False)
    if status:
        runs = [run for run in runs if run['status'] == status]
    if profile_id:
        runs = [run for run in runs if run['profile']['profile_id'] == profile_id]
    return {'runs': runs}


@app.get('/engineering/runs/{run_id}')
def engineering_run(run_id: str):
    from src.engineering_console import run_detail
    from src.session_control import ControlStore

    detail = run_detail(run_id)
    if detail is None:
        raise HTTPException(404, '没有找到该运行')
    saved = RunStore().run(run_id)
    result = detail['result']
    events = result.get('events', [])
    return {
        **detail,
        'checkpoint': (saved or {}).get('checkpoint', {}),
        'tool_ledger': RunStore().outcomes(run_id) if saved else [],
        'commands': ControlStore().commands(saved['session_id'], run_id) if saved else [],
        'trace_projection': project_trace(events),
    }


@app.get('/engineering/datasets')
def engineering_datasets():
    from src.engineering_console import CATEGORY_LABELS, dataset_catalog

    return {'category_labels': CATEGORY_LABELS, 'datasets': dataset_catalog(include_cases=False)}


@app.get('/engineering/datasets/{dataset_id}')
def engineering_dataset(dataset_id: str):
    from src.engineering_console import CATEGORY_LABELS, dataset_catalog

    dataset = next((item for item in dataset_catalog(include_cases=True)
                    if item['dataset_id'] == dataset_id), None)
    if dataset is None:
        raise HTTPException(404, '没有找到该任务集')
    return {'category_labels': CATEGORY_LABELS, 'dataset': dataset}


@app.get('/engineering/experiments')
def engineering_experiments():
    from src.engineering_console import experiment_catalog

    return {'experiments': experiment_catalog(include_report=True)}


@app.get('/business/products')
def business_products(query: str = '', category: str | None = None, max_price: float | None = Query(None, gt=0),
                      available_only: bool = True):
    return tools.search_products(query=query, category=category, max_price=max_price,
                                 available_only=available_only, limit=60)


@app.get('/business/orders')
def business_orders(query: str = '', status: str | None = None):
    from src.workspace import list_orders

    return {'orders': list_orders(query=query, status=status)}


@app.get('/business/orders/{order_id}')
def business_order(order_id: int):
    details = tools.get_order_details(order_id)
    if details.get('error'):
        raise HTTPException(404, details['error'])
    return {'order': details, 'shipment': tools.track_shipment(order_id)}


@app.get('/business/after-sales/cases', response_model=AfterSalesCaseListResponse)
def business_after_sales_cases(
    order_id: int | None = Query(None, gt=0),
    session_id: str | None = Query(None, min_length=1, max_length=64),
    status: CaseStatus | None = None,
    limit: int = Query(50, ge=1, le=200),
):
    """返回真实案件投影；不会从聊天文本推导或伪造售后状态。"""

    from src.after_sales.queries import list_cases

    with tools._session() as session:
        return AfterSalesCaseListResponse(
            cases=list(
                list_cases(
                    session,
                    order_id=order_id,
                    origin_session_id=session_id,
                    status=status,
                    limit=limit,
                )
            )
        )


@app.get('/business/after-sales/cases/{case_id}', response_model=CaseDetail)
def business_after_sales_case(case_id: str):
    """返回案件、运行关联、退款关联、证据、事件与适用政策快照。"""

    from src.after_sales import AfterSalesDomainError
    from src.after_sales.queries import get_case_detail

    try:
        with tools._session() as session:
            return get_case_detail(session, case_id)
    except AfterSalesDomainError as error:
        raise HTTPException(
            404,
            detail={"code": error.code, "message": str(error), "retryable": error.retryable},
        ) from error


@app.post('/runs/{run_id}/resume-async', status_code=202)
def resume_async(run_id: str, req: ResumeRequest):
    from src.session_control import SessionController
    try:
        runner = _product_resume_runner(req.session_id, run_id)
        return SessionController().start(req.session_id, '', resume_id=run_id, runner=runner)
    except AgentError as error:
        raise _control_error(error) from error


@app.get('/runs/{run_id}/events')
async def run_events(run_id: str, session_id: str, request: Request, after: int = 0):
    """SSE 仅传输已落库的实际事件；序号支持断线续传。"""
    import asyncio
    import json
    from fastapi.responses import StreamingResponse
    get_run(run_id, session_id)
    try:
        cursor = max(after, int(request.headers.get('last-event-id', '0')))
    except ValueError as error:
        raise HTTPException(400, '事件序号必须为整数') from error

    async def stream():
        nonlocal cursor
        while not await request.is_disconnected():
            run = RunStore().run(run_id)
            if not run or run['session_id'] != session_id:
                break
            for event in run['events']:
                if event['sequence'] > cursor:
                    cursor = event['sequence']
                    yield f"id: {cursor}\nevent: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            if run['result'] is not None or run['status'] == 'INTERRUPTED':
                break
            yield ': heartbeat\n\n'
            await asyncio.sleep(.25)
    return StreamingResponse(stream(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


class PreferenceBindingRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)


class PreferenceRequest(BaseModel):
    key: str
    value: object
    confirmed: bool = False


@app.post('/profiles', status_code=201)
def create_profile():
    from src.preferences import PreferenceStore
    return {'profile_id': PreferenceStore().create()}


@app.post('/profiles/{profile_id}/bindings')
def bind_profile(profile_id: str, req: PreferenceBindingRequest):
    from src.preferences import PreferenceStore
    try:
        PreferenceStore().bind(req.session_id, profile_id)
        return {'bound': True}
    except AgentError as error:
        raise _control_error(error) from error


@app.get('/profiles/{profile_id}/preferences')
def get_preferences(profile_id: str, query: str = ''):
    from src.preferences import PreferenceStore
    try:
        return {'preferences': PreferenceStore().list(profile_id, query)}
    except AgentError as error:
        raise _control_error(error) from error


@app.put('/profiles/{profile_id}/preferences')
def save_preference(profile_id: str, req: PreferenceRequest):
    from src.preferences import PreferenceStore
    try:
        return {'preferences': PreferenceStore().remember(profile_id, req.key, req.value, confirmed=req.confirmed)}
    except AgentError as error:
        raise _control_error(error) from error


@app.delete('/profiles/{profile_id}/preferences/{key}')
def forget_preference(profile_id: str, key: str):
    from src.preferences import PreferenceStore
    try:
        PreferenceStore().forget(profile_id, key)
        return {'forgotten': True}
    except AgentError as error:
        raise _control_error(error) from error


from src.pg_api import router as pg_router
app.include_router(pg_router)


# A production build is served by FastAPI so deployment has one process and
# one origin.  Vite's development server is still used for local hot reload.
_frontend_dist = Path(__file__).resolve().parent / 'frontend' / 'dist'
if _frontend_dist.is_dir():
    @app.get('/{frontend_path:path}', include_in_schema=False)
    def frontend_app(frontend_path: str):
        requested = (_frontend_dist / frontend_path).resolve()
        if _frontend_dist.resolve() in requested.parents and requested.is_file():
            return FileResponse(requested)
        return FileResponse(_frontend_dist / 'index.html')
