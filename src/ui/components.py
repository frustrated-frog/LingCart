import html
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st


ORDER_LABELS = {"Wireless Headphones": "无线耳机", "Mechanical Keyboard": "机械键盘", "USB-C Hub": "USB-C 扩展坞",
                "Standing Desk": "升降桌", "Desk Lamp": "台灯", "Bluetooth Speaker": "蓝牙音箱",
                "Laptop Stand": "笔记本支架", "Webcam 1080p": "1080p 摄像头", "Ergonomic Mouse": "人体工学鼠标",
                "Noise-Cancelling Earbuds": "降噪耳塞"}
STATUSES = {"placed": "已下单", "shipped": "运输中", "delivered": "已送达", "cancelled": "已取消",
            "pending": "待支付", "paid": "已支付", "refunded": "已退款", "none": "暂无退款",
            "requested": "退款申请中", "approved": "退款已批准", "completed": "退款完成",
            "SUCCEEDED": "已完成", "FAILED": "执行失败", "ESCALATED": "已转人工", "WAITING_APPROVAL": "等待确认",
            "WAITING_CLARIFICATION": "需要补充信息", "CANCELLED": "已取消", "PARTIAL": "部分完成",
            "INTERRUPTED": "等待恢复", "RUNNING": "执行中", "REUSED": "复用已保存结果", "SKIPPED": "已跳过", "REJECTED": "无需执行"}
TOOLS = {"get_order_status": "查询订单状态", "get_order_details": "读取订单详情", "track_shipment": "查询物流进度",
         "check_refund_eligibility": "核验退款资格", "initiate_refund": "发起退款申请", "get_payment_status": "查询支付状态",
         "process_payment": "处理支付", "answer_faq": "查询服务规则", "escalate_to_human": "转交人工客服",
         "search_products": "搜索商品目录", "compare_products": "对比商品参数"}
AGENTS = {"order": "订单助手", "refund": "售后助手", "payment": "支付助手", "faq": "规则助手",
          "shopping": "选购助手", "intake": "任务识别", "runtime": "运行系统", "escalate": "人工客服", "orchestrator": "协作助手"}


def esc(value) -> str:
    return html.escape(str(value))


def status_badge(status: str) -> str:
    kind = "ok" if status in {"SUCCEEDED", "delivered", "paid", "completed"} else (
        "bad" if status == "FAILED" else "warn" if status in {"WAITING_APPROVAL", "WAITING_CLARIFICATION", "ESCALATED", "pending", "requested"} else "")
    return f'<span class="ws-status {kind}">{esc(STATUSES.get(status, status))}</span>'


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="ws-breadcrumb"><span>消费工作空间 / {esc(title)}</span><span class="ws-demo">演示数据 · 本地业务环境</span></div>'
                f'<div class="ws-title">{esc(title)}</div><div class="ws-subtitle">{esc(subtitle)}</div>', unsafe_allow_html=True)


def order_summary(order: dict) -> None:
    name = ORDER_LABELS.get(order["item"], order["item"])
    st.markdown(f'<div class="ws-order-heading"><div class="ws-order"><div class="ws-order-icon">↗</div><div>'
                f'<div class="ws-order-name">{esc(name)}</div><div class="ws-order-meta">订单 #{order["order_id"]} · ¥{order["amount"]:.2f}</div>'
                f'</div></div>{status_badge(order["status"])}</div>', unsafe_allow_html=True)


def product_visual(category: str) -> None:
    if category == "耳机":
        shapes = '<path d="M126 119V84a74 74 0 0 1 148 0v35" fill="none" stroke="#7382a5" stroke-width="18"/>' \
                 '<rect x="111" y="87" width="39" height="79" rx="17" fill="#344665"/><rect x="250" y="87" width="39" height="79" rx="17" fill="#344665"/>' \
                 '<rect x="140" y="100" width="12" height="55" rx="6" fill="#9aa9c5"/><rect x="248" y="100" width="12" height="55" rx="6" fill="#9aa9c5"/>'
    elif category == "键盘":
        keys = ''.join(f'<rect x="{100+x*23}" y="{70+y*23}" width="18" height="17" rx="3" fill="{("#adc0ee" if y == 0 else "#e8edf7")}"/>' for y in range(4) for x in range(9))
        shapes = '<rect x="88" y="58" width="224" height="116" rx="13" fill="#596b90"/>' + keys + '<rect x="168" y="139" width="82" height="17" rx="3" fill="#e8edf7"/>'
    else:
        shapes = '<path d="M271 105h38q30 0 30-24V48" fill="none" stroke="#8d9cb9" stroke-width="8"/>' \
                 '<rect x="90" y="70" width="183" height="73" rx="17" fill="#71829f"/>' \
                 '<rect x="110" y="112" width="24" height="10" rx="2" fill="#2b3853"/><rect x="148" y="112" width="24" height="10" rx="2" fill="#2b3853"/>' \
                 '<rect x="190" y="112" width="39" height="10" rx="2" fill="#2b3853"/><rect x="246" y="98" width="7" height="27" rx="2" fill="#2b3853"/>' \
                 '<rect x="329" y="26" width="20" height="27" rx="4" fill="#526481"/>'
    st.markdown(f'<div class="ws-product-visual"><svg viewBox="0 0 400 200" role="img" aria-label="{esc(category)}商品示意图">{shapes}</svg></div>', unsafe_allow_html=True)


def product_summary(product: dict, visual: bool = True) -> None:
    if visual:
        product_visual(product["category"])
    st.markdown(f'<div class="ws-category">{esc(product["category"])} · 商品示意图</div>'
                f'<div class="ws-product-name">{esc(product["name"])}</div>'
                f'<div class="ws-product-desc">{esc(product["summary"])}</div>'
                f'<div class="ws-price">¥{product["price"]:.2f}</div>'
                f'<div class="ws-product-id">{esc(product["product_id"])} · 库存 {product["stock"]}</div>', unsafe_allow_html=True)


def local_time(value: str) -> str:
    date = datetime.fromisoformat(value)
    if date.tzinfo is not None:
        date = date.astimezone(ZoneInfo("Asia/Shanghai"))
    return date.strftime("%m-%d %H:%M")


def timeline(items: list[tuple[str, str]]) -> None:
    st.markdown('<ul class="ws-timeline">' + ''.join(
        f'<li>{esc(title)}<div class="ws-timeline-time">{esc(detail)}</div></li>' for title, detail in items) + '</ul>', unsafe_allow_html=True)


def render_run(result: dict, detailed: bool = False) -> None:
    status = result.get("status", "UNKNOWN")
    st.markdown(f'<div class="ws-run-heading"><span>{esc(AGENTS.get(result.get("agent"), "消费助手"))}</span>{status_badge(status)}</div>', unsafe_allow_html=True)
    st.caption(f"运行 {result.get('run_id', '—')[:12]} · 终止原因 {result.get('terminal_reason', '—')}")
    if detailed:
        cols = st.columns(3)
        latency = result.get("latency_ms")
        cols[0].metric("实际耗时", f"{latency / 1000:.2f}s" if latency is not None else "暂无数据")
        calls = [event for event in result.get("events", []) if event.get("type") == "tool_call"]
        cols[1].metric("工具调用", len(calls))
        token_count = (result.get("usage") or {}).get("total_tokens")
        cols[2].metric("Token", str(token_count) if token_count is not None else "暂无数据")
        plan = result.get("plan")
        if plan:
            st.markdown("#### 任务计划与完成情况")
            st.dataframe([{"步骤": s["step_id"], "负责助手": AGENTS.get(s["specialist"], s["specialist"]),
                           "操作": TOOLS.get(s["tool"], {"prepare_payment": "准备支付确认", "prepare_refund": "准备退款确认"}.get(s["tool"], s["tool"])),
                           "前置步骤": "、".join(s["depends_on"]) or "—", "结果": STATUSES.get(s["status"], s["status"])}
                          for s in plan["steps"]], hide_index=True, width="stretch")
        if result.get("budget"):
            with st.expander("运行预算"):
                st.json(result["budget"])
    items = []
    for event in result.get("events", []):
        kind = event.get("type")
        if kind == "route_decision":
            items.append(("识别任务与执行路径", "规则快速路径" if event.get("source") == "deterministic" else "模型路由"))
        elif kind == "tool_call":
            items.append((TOOLS.get(event.get("tool"), event.get("tool", "工具调用")), "已发起调用"))
        elif kind == "tool_result":
            items.append(("工具返回", event.get("status", "已返回")))
        elif kind == "model_call":
            items.append(("模型调用", f"{event.get('duration_ms', 0):.0f}ms"))
        elif kind == "error":
            items.append(("执行错误", event.get("message", event.get("code", "未知错误"))))
        elif kind == "terminal":
            items.append((STATUSES.get(event.get("status"), event.get("status", "已结束")), event.get("reason", "")))
        elif kind == "approval_decision":
            items.append(("收到操作确认", event.get("decision", "")))
        elif kind == "verification":
            items.append(("核验任务结果", "全部目标已完成" if event.get("all_goals_satisfied") else "仍有待完成或待确认的步骤"))
    if items:
        timeline(items)
    if detailed:
        for index, entry in enumerate(result.get("trace", [])):
            with st.expander(f"{index + 1}. {TOOLS.get(entry.get('tool'), entry.get('tool', '工具'))}"):
                st.caption("参数")
                st.json(entry.get("args", {}))
                st.caption("结果")
                st.json(entry.get("result"))
        with st.expander("完整运行事件"):
            st.json(result.get("events", []))
        if result.get("error"):
            st.error(result["error"].get("message", str(result["error"])))


def business_results(result: dict) -> None:
    for entry in result.get("trace", []):
        output = entry.get("result")
        if not isinstance(output, dict):
            continue
        if output.get("products"):
            products = output["products"]
            if entry.get("tool") == "compare_products":
                comparison_table(products)
            else:
                st.dataframe([{"商品": p["name"], "编号": p["product_id"], "价格（元）": p["price"], "规格": p["summary"], "库存": p["stock"]} for p in products], hide_index=True, width="stretch")
        elif entry.get("tool") == "track_shipment" and output.get("events"):
            with st.expander("物流时间线"):
                timeline([(event["description"], local_time(event["timestamp"])) for event in output["events"]])
        elif output.get("item"):
            name = ORDER_LABELS.get(output["item"], output["item"])
            st.caption(f"订单 #{output['order_id']} · {name} · ¥{output['amount']:.2f} · {STATUSES.get(output['status'], output['status'])}")
        elif output.get("ticket_id"):
            st.info(f"人工处理工单 #{output['ticket_id']}")


def comparison_table(products: list[dict]) -> None:
    fields = list(dict.fromkeys(key for product in products for key in product["specs"]))
    rows = [{"对比项": field, **{f"{p['name']} ({p['product_id']})": (
        f"¥{p['price']:.2f}" if field == "价格" else str(p['stock']) if field == "库存" else str(p['specs'].get(field, "—")))
        for p in products}} for field in ["价格", "库存", *fields]]
    st.dataframe(rows, hide_index=True, width="stretch")
