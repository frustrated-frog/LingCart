import html
import json
import time

import streamlit as st

from src.memory import ConversationMemory
from src.orchestrator import run_orchestrator


st.set_page_config(
    page_title="选序 · 智能消费助手",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    :root {
        --page: #f7f8fc;
        --surface: #ffffff;
        --text: #182235;
        --text-secondary: #718096;
        --text-tertiary: #98a2b3;
        --brand: #2563eb;
        --brand-soft: #eff6ff;
        --border: #e6eaf1;
        --success: #16a34a;
        --success-soft: #f0fdf4;
        --warning: #d97706;
        --warning-soft: #fff7ed;
        --danger: #dc2626;
        --danger-soft: #fef2f2;
        --radius: 16px;
        --input-radius: 18px;
        --shadow: 0 1px 3px rgba(24, 34, 53, .04);
    }

    html, body, [class*="css"] {
        font-family: "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
        color: var(--text);
    }
    .stApp { background: var(--page); }
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stHeader"] { display: none; }
    [data-testid="stMainBlockContainer"] {
        width: min(1320px, calc(100% - 48px));
        max-width: 1320px;
        padding: 18px 0 40px;
    }

    .topbar {
        height: 62px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 20px;
        border: 1px solid var(--border);
        border-radius: var(--radius);
        background: rgba(255, 255, 255, .96);
        box-shadow: var(--shadow);
    }
    .brand-wrap { display: flex; align-items: center; gap: 12px; }
    .brand-mark {
        width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        border-radius: 10px;
        color: #fff;
        background: var(--brand);
        font-size: 16px;
        font-weight: 800;
    }
    .brand-name { color: var(--text); font-size: 16px; font-weight: 750; }
    .brand-desc { margin-top: 1px; color: var(--text-secondary); font-size: 12px; }
    .service-state {
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--text-secondary);
        font-size: 12px;
    }
    .service-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--success);
        box-shadow: 0 0 0 4px rgba(22, 163, 74, .09);
    }

    .view-controls {
        min-height: 48px;
        display: flex;
        align-items: center;
        margin: 10px 0 14px;
    }
    [data-testid="stToggle"] label { gap: 8px; }
    [data-testid="stToggle"] p {
        color: var(--text-secondary);
        font-size: 13px;
    }

    .workspace-title {
        margin: 4px 0 3px;
        color: var(--text);
        font-size: 23px;
        font-weight: 750;
        letter-spacing: -.02em;
    }
    .workspace-copy {
        margin-bottom: 18px;
        color: var(--text-secondary);
        font-size: 14px;
    }

    /* Conversation is the visual protagonist. */
    .chat-divider {
        height: 1px;
        margin-bottom: 8px;
        background: var(--border);
    }
    .empty-state {
        max-width: 590px;
        margin: 100px auto 28px;
        text-align: center;
    }
    .empty-symbol {
        width: 48px;
        height: 48px;
        display: grid;
        place-items: center;
        margin: 0 auto 18px;
        border: 1px solid #dbeafe;
        border-radius: 14px;
        color: var(--brand);
        background: var(--brand-soft);
        font-size: 19px;
        font-weight: 800;
    }
    .empty-title {
        color: var(--text);
        font-size: 22px;
        font-weight: 750;
        letter-spacing: -.02em;
    }
    .empty-copy {
        max-width: 480px;
        margin: 9px auto 0;
        color: var(--text-secondary);
        font-size: 15px;
        line-height: 1.75;
    }
    .quick-label {
        margin: 24px 0 10px;
        color: var(--text-tertiary);
        font-size: 12px;
        text-align: center;
    }

    div[data-testid="stChatMessage"] {
        margin: 0;
        padding: 18px 6px;
        border: 0;
        border-bottom: 1px solid #f0f2f6;
        border-radius: 0;
        background: transparent;
    }
    div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        color: #26354d;
        font-size: 15.5px;
        line-height: 1.78;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label="Chat message from user"]) {
        width: fit-content;
        max-width: 76%;
        margin: 18px 0 18px auto;
        padding: 11px 15px;
        border: 1px solid #dbeafe;
        border-radius: 15px 15px 4px 15px;
        background: var(--brand-soft);
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label="Chat message from user"]) > div:first-child {
        display: none;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageContent"][aria-label="Chat message from user"]) [data-testid="stChatMessageContent"] {
        width: auto;
    }
    .message-meta {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 10px;
    }
    .agent-chip, .cache-chip {
        width: fit-content;
        padding: 3px 8px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
    }
    .agent-chip { color: #1d4ed8; background: var(--brand-soft); }
    .cache-chip { color: #9a3412; background: var(--warning-soft); }

    .business-card {
        max-width: 560px;
        margin-top: 14px;
        overflow: hidden;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: var(--surface);
        box-shadow: var(--shadow);
    }
    .business-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 13px 15px;
        border-bottom: 1px solid var(--border);
        background: #fbfcfe;
    }
    .business-title { color: var(--text); font-size: 14px; font-weight: 750; }
    .business-status {
        padding: 3px 8px;
        border-radius: 999px;
        color: #166534;
        background: var(--success-soft);
        font-size: 11px;
        font-weight: 700;
    }
    .business-body {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 13px 18px;
        padding: 14px 15px 15px;
    }
    .business-key { color: var(--text-tertiary); font-size: 11px; }
    .business-value { margin-top: 3px; color: #344054; font-size: 13px; font-weight: 650; }
    .escalation-card {
        max-width: 560px;
        display: flex;
        align-items: flex-start;
        gap: 11px;
        margin-top: 14px;
        padding: 13px 14px;
        border: 1px solid #fed7aa;
        border-radius: 13px;
        background: var(--warning-soft);
    }
    .escalation-icon {
        width: 25px;
        height: 25px;
        display: grid;
        place-items: center;
        flex: 0 0 25px;
        border-radius: 7px;
        color: #fff;
        background: var(--warning);
        font-size: 12px;
        font-weight: 800;
    }
    .escalation-title { color: #9a3412; font-size: 13px; font-weight: 750; }
    .escalation-copy { margin-top: 3px; color: #b45309; font-size: 12px; }

    [data-testid="stChatInput"] {
        position: sticky;
        bottom: 18px;
        z-index: 10;
        margin-top: 18px;
        border: 1px solid #ccd5e3;
        border-radius: var(--input-radius);
        background: rgba(255,255,255,.98);
        box-shadow: 0 10px 28px rgba(24,34,53,.10);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--brand);
        box-shadow: 0 0 0 3px rgba(37,99,235,.10), 0 10px 28px rgba(24,34,53,.10);
    }
    [data-testid="stChatInput"] textarea { font-size: 15px; }

    .stButton > button {
        min-height: 46px;
        color: #344054;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--surface);
        font-size: 13px;
        box-shadow: var(--shadow);
    }
    .stButton > button:hover {
        color: var(--brand);
        border-color: #bfdbfe;
        background: #fafcff;
    }
    .stButton > button:focus-visible { outline: 3px solid rgba(37,99,235,.15); }

    /* One continuous inspector, instead of stacked debug cards. */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--border);
        border-radius: var(--radius);
        background: var(--surface);
        box-shadow: var(--shadow);
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0 !important;
    }
    .inspector-head {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 14px;
        padding: 19px 20px 16px;
        border-bottom: 1px solid var(--border);
    }
    .inspector-title { color: var(--text); font-size: 16px; font-weight: 750; }
    .inspector-copy { margin-top: 3px; color: var(--text-secondary); font-size: 12px; }
    .status-badge {
        padding: 5px 9px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 750;
        white-space: nowrap;
    }
    .status-idle { color: #475467; background: #f2f4f7; }
    .status-success { color: #166534; background: var(--success-soft); }
    .status-escalated { color: #9a3412; background: var(--warning-soft); }
    .status-waiting { color: #9a3412; background: var(--warning-soft); }
    .status-failed { color: #991b1b; background: var(--danger-soft); }
    .inspector-section { padding: 17px 20px; border-bottom: 1px solid var(--border); }
    .inspector-section:last-child { border-bottom: 0; }
    .section-label { margin-bottom: 12px; color: #475467; font-size: 12px; font-weight: 750; }
    .overview-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
    }
    .overview-value {
        color: var(--text);
        font-family: "SFMono-Regular", Consolas, monospace;
        font-size: 15px;
        font-weight: 750;
    }
    .overview-key { margin-top: 3px; color: var(--text-tertiary); font-size: 11px; }
    .inspector-empty {
        padding: 48px 20px;
        color: var(--text-secondary);
        font-size: 13px;
        line-height: 1.7;
        text-align: center;
    }
    .timeline { position: relative; }
    .timeline-item { position: relative; padding: 0 0 18px 24px; }
    .timeline-item:last-child { padding-bottom: 0; }
    .timeline-item::before {
        content: "";
        position: absolute;
        left: 2px;
        top: 5px;
        width: 8px;
        height: 8px;
        border: 2px solid #fff;
        border-radius: 50%;
        background: #94a3b8;
        box-shadow: 0 0 0 1px #cbd5e1;
    }
    .timeline-item.active::before { background: var(--brand); box-shadow: 0 0 0 1px #93c5fd; }
    .timeline-item.warning::before { background: var(--warning); box-shadow: 0 0 0 1px #fdba74; }
    .timeline-item:not(:last-child)::after {
        content: "";
        position: absolute;
        left: 5.5px;
        top: 16px;
        bottom: -1px;
        border-left: 1px solid #dce2ea;
    }
    .timeline-title { color: #344054; font-size: 13px; font-weight: 700; }
    .timeline-copy { margin-top: 3px; color: var(--text-secondary); font-size: 12px; line-height: 1.55; }
    .session-row { display: flex; justify-content: space-between; gap: 16px; padding: 6px 0; }
    .session-key { color: var(--text-secondary); font-size: 12px; }
    .session-value { color: #344054; font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; font-weight: 700; }

    [data-testid="stExpander"] {
        margin: 0 20px 10px;
        border-color: var(--border);
        border-radius: 11px;
        background: #fbfcfe;
    }
    [data-testid="stExpander"] summary p { font-size: 12px; }
    [data-testid="stJson"] { font-size: 11px; }

    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] * { color: #e5e7eb; }
    [data-testid="stSidebar"] .stButton > button { background: #1f2937; border-color: #374151; }
    .drawer-title { padding: 8px 0 5px; font-size: 16px; font-weight: 750; }
    .drawer-copy { color: #9ca3af !important; font-size: 12px; line-height: 1.7; }
    </style>
    """,
    unsafe_allow_html=True,
)


AGENT_LABELS = {
    "order": "订单专员",
    "refund": "退款专员",
    "payment": "支付专员",
    "faq": "规则助手",
    "escalate": "人工客服",
    "intake": "任务识别",
    "runtime": "运行系统",
}

TOOL_LABELS = {
    "get_order_status": "查询订单状态",
    "get_order_details": "读取订单详情",
    "track_shipment": "查询物流进度",
    "check_refund_eligibility": "核验退款资格",
    "initiate_refund": "发起退款申请",
    "get_payment_status": "查询支付状态",
    "process_payment": "处理订单支付",
    "answer_faq": "检索服务规则",
    "escalate_to_human": "转交人工客服",
}

STATUS_LABELS = {
    "placed": "已下单",
    "shipped": "运输中",
    "delivered": "已送达",
    "cancelled": "已取消",
    "pending": "待支付",
    "paid": "已支付",
    "refunded": "已退款",
    "requested": "已申请",
    "approved": "已批准",
    "completed": "已完成",
}

FIELD_LABELS = {
    "item": "商品",
    "customer_name": "客户",
    "order_date": "下单日期",
    "amount": "金额",
    "method": "支付方式",
    "tracking_message": "物流信息",
    "eligible": "退款资格",
    "refund_id": "退款单号",
    "reason": "说明",
}

TERMINAL_LABELS = {
    "COMPLETED": "已生成可验证结果",
    "APPROVAL_REQUIRED": "确认后才会继续执行",
    "MISSING_ORDER_ID": "缺少订单号",
    "UNSUPPORTED_INTENT": "超出当前支持范围",
    "MODEL_UNAVAILABLE": "模型服务暂时不可用",
    "MODEL_TIMEOUT": "模型响应超时",
    "MODEL_RATE_LIMIT": "模型请求受限",
    "BUSINESS_RULE_REJECTED": "业务规则不允许执行",
    "BUSINESS_DATA_NOT_FOUND": "未找到对应业务数据",
    "NO_ACTION_NEEDED": "无需执行额外操作",
    "USER_REQUESTED_ESCALATION": "用户主动要求人工服务",
    "MODEL_ROUTED_ESCALATION": "已按路由结果转人工处理",
    "MODEL_FAILED_AFTER_TOOL_RESULT": "模型失败，已保留工具结果",
    "REPEATED_TOOL_CALL": "已阻止重复工具调用",
    "USER_CANCELLED_ACTION": "用户取消了待执行操作",
    "MISSING_PAYMENT_METHOD": "需要补充支付方式",
    "INVALID_PENDING_ACTION": "待确认操作已失效",
}


def terminal_label(reason: str | None) -> str:
    if not reason:
        return "等待继续"
    return TERMINAL_LABELS.get(reason, "运行已结束")


def init_state() -> None:
    defaults = {
        "messages": [],
        "memory": ConversationMemory(),
        "total_tokens": 0,
        "last_agent": None,
        "last_route": None,
        "last_trace": [],
        "last_events": [],
        "last_usage": {},
        "last_latency_ms": None,
        "last_status": None,
        "last_terminal_reason": None,
        "last_run_id": None,
        "last_error": None,
        "last_pending_action": None,
        "last_prompt": None,
        "run_count": 0,
        "developer_mode": False,
        "show_inspector": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session() -> None:
    keep = {
        "developer_mode": st.session_state.get("developer_mode", False),
        "show_inspector": st.session_state.get("show_inspector", True),
    }
    for key in list(st.session_state.keys()):
        if key not in keep:
            st.session_state.pop(key, None)
    init_state()
    st.session_state.update(keep)


def get_ticket_id(trace: list) -> str | None:
    for entry in reversed(trace):
        result = entry.get("result")
        if isinstance(result, dict) and result.get("ticket_id"):
            return str(result["ticket_id"])
    return None


def localized_reply(result: dict) -> str:
    if result.get("agent") != "escalate":
        return result["reply"]
    ticket_id = get_ticket_id(result.get("trace") or [])
    suffix = f"（工单 #{ticket_id}）" if ticket_id else ""
    return f"当前请求已转交人工客服{suffix}，处理进展会在后续通知你。"


def derive_status(result: dict) -> str:
    if result.get("status"):
        return result["status"]
    if result.get("agent") == "escalate" or result.get("route") == "ESCALATE":
        return "ESCALATED"
    return "SUCCEEDED"


def render_message_meta(agent: str | None, cached: bool) -> None:
    if not agent:
        return
    cache = '<span class="cache-chip">缓存响应</span>' if cached else ""
    label = html.escape(AGENT_LABELS.get(agent, agent))
    st.markdown(
        f'<div class="message-meta"><span class="agent-chip">{label}</span>{cache}</div>',
        unsafe_allow_html=True,
    )


def merged_business_data(trace: list) -> dict:
    merged = {}
    for entry in trace:
        result = entry.get("result")
        if isinstance(result, dict) and ("order_id" in result or "refund_id" in result):
            merged.update(result)
    return merged


def display_value(key: str, value) -> str:
    if key == "status":
        return STATUS_LABELS.get(str(value), str(value))
    if key == "eligible":
        return "可以退款" if value else "暂不可退款"
    if key == "amount" and isinstance(value, (int, float)):
        return f"¥{value:,.2f}"
    tracking = {
        "Order placed, not yet shipped.": "订单已创建，等待发货",
        "In transit.": "包裹运输中",
        "Delivered.": "包裹已送达",
        "Order was cancelled; no shipment in progress.": "订单已取消，无物流信息",
    }
    return tracking.get(str(value), str(value))


def render_business_result(trace: list) -> None:
    data = merged_business_data(trace)
    order_id = data.get("order_id")
    if order_id is None:
        return

    status = display_value("status", data.get("status", "已查询"))
    fields = []
    for key in ("item", "customer_name", "order_date", "amount", "method", "tracking_message", "eligible", "refund_id", "reason"):
        if key in data and data[key] is not None:
            fields.append(
                '<div><div class="business-key">'
                + html.escape(FIELD_LABELS[key])
                + '</div><div class="business-value">'
                + html.escape(display_value(key, data[key]))
                + "</div></div>"
            )
    body = "".join(fields) or '<div><div class="business-key">查询结果</div><div class="business-value">已获取订单数据</div></div>'
    st.markdown(
        f"""
        <div class="business-card">
            <div class="business-head"><div class="business-title">订单 #{html.escape(str(order_id))}</div><div class="business-status">{html.escape(status)}</div></div>
            <div class="business-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_escalation(trace: list) -> None:
    ticket_id = get_ticket_id(trace)
    if not ticket_id:
        return
    st.markdown(
        f"""
        <div class="escalation-card">
            <div class="escalation-icon">↗</div>
            <div><div class="escalation-title">等待人工客服处理</div><div class="escalation-copy">工单 #{html.escape(ticket_id)} · 自动流程已结束，但用户任务尚未完成</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pending_state(status: str, terminal_reason: str | None) -> None:
    if status == "WAITING_APPROVAL":
        title = "等待你的确认"
        copy = "涉及支付或退款等状态变更，系统不会在未确认时执行。"
    elif status == "WAITING_CLARIFICATION":
        title = "需要补充信息"
        copy = "补充订单号后，系统会继续处理当前任务。"
    else:
        return
    st.markdown(
        f"""
        <div class="escalation-card">
            <div class="escalation-icon">!</div>
            <div><div class="escalation-title">{title}</div><div class="escalation-copy">{copy} · {html.escape(terminal_label(terminal_reason))}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message(message: dict) -> None:
    avatar = "👤" if message["role"] == "user" else "✨"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            trace = message.get("trace") or []
            render_message_meta(message.get("agent"), message.get("cached", False))
            if message.get("status") == "ESCALATED":
                render_escalation(trace)
            elif message.get("status") in {"WAITING_APPROVAL", "WAITING_CLARIFICATION"}:
                render_business_result(trace)
                render_pending_state(message["status"], message.get("terminal_reason"))
            else:
                render_business_result(trace)


def status_presentation(status: str | None) -> tuple[str, str]:
    if status == "SUCCEEDED":
        return "已完成", "status-success"
    if status == "ESCALATED":
        return "已转人工", "status-escalated"
    if status == "WAITING_APPROVAL":
        return "等待确认", "status-waiting"
    if status == "WAITING_CLARIFICATION":
        return "等待补充", "status-waiting"
    if status == "CANCELLED":
        return "已取消", "status-idle"
    if status == "FAILED":
        return "执行失败", "status-failed"
    return "等待任务", "status-idle"


def render_timeline() -> None:
    status = st.session_state.last_status
    trace = st.session_state.last_trace
    events = st.session_state.last_events
    if not status:
        st.markdown('<div class="inspector-empty">发送问题后，这里会展示真实的路由和工具执行过程。</div>', unsafe_allow_html=True)
        return

    items = []
    for event in events:
        event_type = event.get("type")
        if event_type == "route_decision":
            source = "规则快速路径" if event.get("source") == "deterministic" else "模型路由"
            items.append(("任务识别完成", f"{source} · {event.get('route', 'UNKNOWN')}", "active"))
        elif event_type == "approval_decision":
            items.append(("已收到明确确认", "允许执行待处理的状态变更", "active"))
        elif event_type == "model_call":
            duration = event.get("duration_ms")
            copy = f"真实耗时 {duration:.0f}ms" if isinstance(duration, (int, float)) else "模型已返回"
            items.append(("模型调用", copy, "active"))
        elif event_type == "tool_call":
            tool = event.get("tool", "unknown")
            items.append((TOOL_LABELS.get(tool, tool), "工具调用已发起", "active"))
        elif event_type == "tool_result":
            tool = event.get("tool", "unknown")
            duration = event.get("duration_ms")
            copy = f"返回真实结果 · {duration:.0f}ms" if isinstance(duration, (int, float)) else "返回真实结果"
            items.append((f"{TOOL_LABELS.get(tool, tool)}完成", copy, "active"))
        elif event_type == "error":
            items.append(("执行出现错误", terminal_label(event.get("code")), "warning"))
        elif event_type == "terminal":
            terminal_status = event.get("status", status)
            kind = "warning" if terminal_status in {"ESCALATED", "WAITING_APPROVAL", "WAITING_CLARIFICATION", "FAILED"} else "active"
            items.append((status_presentation(terminal_status)[0], terminal_label(event.get("reason")), kind))

    if not items:
        agent = AGENT_LABELS.get(st.session_state.last_agent, st.session_state.last_agent or "未知")
        items = [
            ("已接收任务", "请求已进入 Agent 执行流程", "active"),
            (f"由{agent}处理", f"路由结果：{st.session_state.last_route or st.session_state.last_agent}", "active"),
        ]
        for entry in trace:
            tool = entry.get("tool", "unknown")
            items.append((TOOL_LABELS.get(tool, tool), "工具已返回真实结果", "active"))

    markup = "".join(
        f'<div class="timeline-item {kind}"><div class="timeline-title">{html.escape(title)}</div><div class="timeline-copy">{html.escape(copy)}</div></div>'
        for title, copy, kind in items
    )
    st.markdown(f'<div class="timeline">{markup}</div>', unsafe_allow_html=True)


def render_inspector() -> None:
    status = st.session_state.last_status
    status_text, status_class = status_presentation(status)
    run_id = (st.session_state.last_run_id or "—")[:12]

    with st.container(border=True):
        st.markdown(
            f"""
            <div class="inspector-head">
                <div><div class="inspector-title">Agent 执行过程</div><div class="inspector-copy">仅展示本次请求的真实事件</div></div>
                <div class="status-badge {status_class}">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not status:
            render_timeline()
            return

        latency = st.session_state.last_latency_ms
        trace = st.session_state.last_trace
        usage = st.session_state.last_usage or {}
        metrics = [
            (run_id, "运行编号"),
            (f"{latency / 1000:.2f}s" if latency is not None else "暂无数据", "实际耗时"),
            (str(len(trace)), "工具调用"),
        ]
        if usage.get("total_tokens") is not None:
            metrics.append((f'{usage.get("total_tokens") or 0:,}', "Token 消耗"))
        metrics_html = "".join(
            f'<div><div class="overview-value">{html.escape(value)}</div><div class="overview-key">{label}</div></div>'
            for value, label in metrics
        )
        st.markdown(
            f'<div class="inspector-section"><div class="section-label">运行概览</div><div class="overview-grid">{metrics_html}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="inspector-section"><div class="section-label">执行时间线</div>', unsafe_allow_html=True)
        render_timeline()
        st.markdown("</div>", unsafe_allow_html=True)

        if st.session_state.developer_mode and trace:
            st.markdown('<div class="inspector-section"><div class="section-label">工具详情</div></div>', unsafe_allow_html=True)
            for index, entry in enumerate(trace, start=1):
                tool = entry.get("tool", "unknown")
                with st.expander(f"{index}. {TOOL_LABELS.get(tool, tool)} · {tool}"):
                    st.caption("调用参数")
                    st.json(entry.get("args") or {})
                    st.caption("工具结果")
                    st.json(entry.get("result"))

        if st.session_state.developer_mode and st.session_state.last_events:
            with st.expander("完整运行事件 · Run Trace"):
                st.json(st.session_state.last_events)
        if st.session_state.developer_mode and st.session_state.last_error:
            with st.expander("终止错误"):
                st.json(st.session_state.last_error)

        context_rows = []
        last_order = st.session_state.memory.last_order_id
        if last_order is not None:
            context_rows.append(("最近订单", f"#{last_order}"))
        if len(st.session_state.messages) > 0:
            context_rows.append(("会话消息", str(len(st.session_state.messages))))
        if st.session_state.total_tokens > 0:
            context_rows.append(("累计 Token", f"{st.session_state.total_tokens:,}"))
        if context_rows:
            rows = "".join(
                f'<div class="session-row"><span class="session-key">{label}</span><span class="session-value">{html.escape(value)}</span></div>'
                for label, value in context_rows
            )
            st.markdown(
                f'<div class="inspector-section"><div class="section-label">会话信息</div>{rows}</div>',
                unsafe_allow_html=True,
            )


init_state()


with st.sidebar:
    st.markdown('<div class="drawer-title">选序 · Agent Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="drawer-copy">当前阶段：稳定 Baseline、保留真实执行轨迹，并为 Shopping Agent、Memory 和 Planning 事件预留统一展示结构。</div>', unsafe_allow_html=True)
    st.divider()
    st.button("新建会话", on_click=reset_session, use_container_width=True)


st.markdown(
    """
    <div class="topbar">
        <div class="brand-wrap"><div class="brand-mark">选</div><div><div class="brand-name">选序 · 智能消费助手</div><div class="brand-desc">购物决策与消费服务 Agent</div></div></div>
        <div class="service-state"><span class="service-dot"></span>服务正常</div>
    </div>
    """,
    unsafe_allow_html=True,
)


control_spacer, developer_control, inspector_control = st.columns([6.4, 1.8, 1.8], gap="small")
with control_spacer:
    st.markdown('<div class="view-controls"></div>', unsafe_allow_html=True)
with developer_control:
    st.toggle("开发者模式", key="developer_mode")
with inspector_control:
    st.toggle("显示执行面板", key="show_inspector")


if st.session_state.show_inspector:
    chat_col, inspector_col = st.columns([7, 3], gap="large")
else:
    chat_col = st.container()
    inspector_col = None


with chat_col:
    st.markdown('<div class="workspace-title">对话工作台</div><div class="workspace-copy">描述你的订单、支付、退款或服务规则问题。</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-divider"></div>', unsafe_allow_html=True)

    for message in st.session_state.messages:
        render_message(message)

    quick_prompt = None
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-symbol">✦</div>
                <div class="empty-title">今天想处理什么消费问题？</div>
                <div class="empty-copy">告诉我订单号和你的目标。我会调用真实业务工具完成查询，并把关键处理步骤保留在执行面板中。</div>
                <div class="quick-label">选择一个常用场景开始</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        quick_cols = st.columns(4, gap="small")
        prompts = [
            ("查询订单", "请查询订单 1004 的当前状态"),
            ("追踪物流", "请追踪订单 1002 的物流进度"),
            ("申请退款", "订单 1001 的商品损坏了，请检查能否退款"),
            ("咨询规则", "请告诉我退货和退款规则"),
        ]
        for column, (label, example) in zip(quick_cols, prompts):
            with column:
                if st.button(label, use_container_width=True):
                    quick_prompt = example

    typed_prompt = st.chat_input("继续提问，或描述新的消费需求…")
    prompt = quick_prompt or typed_prompt
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        memory: ConversationMemory = st.session_state.memory
        augmented_message = memory.context_note() + prompt

        started_at = time.perf_counter()
        with st.spinner("正在处理，请稍候…"):
            result = run_orchestrator(
                augmented_message,
                history=memory.turns,
                pending_action=memory.pending_action,
            )
        latency_ms = result.get("latency_ms") or round((time.perf_counter() - started_at) * 1000)
        reply = localized_reply(result)
        status = derive_status(result)
        trace = result.get("trace") or []
        usage = result.get("usage") or {}

        st.session_state.total_tokens += usage.get("total_tokens") or 0
        st.session_state.last_agent = result.get("agent")
        st.session_state.last_route = result.get("route")
        st.session_state.last_trace = trace
        st.session_state.last_events = result.get("events") or []
        st.session_state.last_usage = usage
        st.session_state.last_latency_ms = latency_ms
        st.session_state.last_status = status
        st.session_state.last_terminal_reason = result.get("terminal_reason")
        st.session_state.last_run_id = result.get("run_id")
        st.session_state.last_error = result.get("error")
        st.session_state.last_pending_action = result.get("pending_action")
        st.session_state.last_prompt = prompt
        st.session_state.run_count += 1

        memory.add_turn(prompt, reply)
        memory.update_last_order_id(trace)
        memory.update_pending_action(result)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": reply,
                "agent": result.get("agent"),
                "trace": trace,
                "status": status,
                "terminal_reason": result.get("terminal_reason"),
                "pending_action": result.get("pending_action"),
                "cached": result.get("cached", False),
            }
        )
        st.rerun()


if inspector_col is not None:
    with inspector_col:
        render_inspector()
