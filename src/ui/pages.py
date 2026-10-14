"""Connected pages backed by business records and recorded runs."""
import uuid

import streamlit as st

from src import tools
from src.memory import ConversationMemory
from src.durable import load_conversation, run_conversation, resume_run
from src.errors import AgentError
from src.ui.components import AGENTS, STATUSES, business_results, comparison_table, esc, local_time, order_summary, page_header, product_summary, render_run, status_badge, timeline
from src.workspace import list_orders, list_runs


def init_state():
    defaults = dict(memory=ConversationMemory(), messages=[], conversation_id=uuid.uuid4().hex,
                    queued_prompt=None, compare_ids=[], product_page=0, catalog_filter=None,
                    selected_run=None, archive_error=None)
    first_load = "conversation_id" not in st.session_state
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if first_load:
        st.session_state.conversation_id = st.query_params.get("conversation", st.session_state.conversation_id)
        restore_conversation()
    st.query_params["conversation"] = st.session_state.conversation_id


def new_conversation():
    st.session_state.memory = ConversationMemory()
    st.session_state.messages = []
    st.session_state.conversation_id = uuid.uuid4().hex
    st.session_state.queued_prompt = None
    st.query_params["conversation"] = st.session_state.conversation_id
    st.session_state.active_run_id = None


def ask_assistant(prompt):
    st.session_state.queued_prompt = prompt
    st.switch_page(st.session_state.workspace_pages["home"])


def view_run(run_id):
    st.session_state.selected_run = run_id
    st.switch_page(st.session_state.workspace_pages["lab"])


def restore_conversation():
    saved = load_conversation(st.session_state.conversation_id)
    st.session_state.memory = saved["memory"]
    st.session_state.messages = saved["messages"]
    st.session_state.active_run_id = saved["active_run_id"]


def execute_message(prompt):
    pending = st.session_state.memory.pending_action
    with st.spinner("正在处理你的需求…"):
        try:
            run_conversation(prompt, st.session_state.conversation_id,
                             action_id=(pending or {}).get("action_id"))
            st.session_state.archive_error = None
        except Exception as exc:
            st.session_state.archive_error = f"任务中断：{exc}。请恢复当前任务后继续。"
    restore_conversation()


def approval_controls():
    pending = st.session_state.memory.pending_action
    if not pending:
        return
    with st.container(border=True, key="ws_panel_approval"):
        title = "支付" if pending["type"] == "PROCESS_PAYMENT" else "退款"
        st.markdown(f"### 确认订单 #{pending['order_id']} 的{title}操作")
        st.caption("此操作只修改本地演示数据库，确认后才会执行。")
        method = None
        if pending["type"] == "PROCESS_PAYMENT":
            if pending.get("amount") is not None:
                st.write(f"待支付金额：¥{pending['amount']:.2f}")
            else:
                st.caption("支付金额以订单数据库为准，执行时会重新核验支付状态。")
            method = st.selectbox("支付方式", ["card", "paypal"], format_func=lambda v: {"card": "银行卡", "paypal": "PayPal"}[v], key="approval_method")
        cols = st.columns(2)
        if cols[0].button(f"确认{title}", type="primary", width="stretch", key="approve_action"):
            execute_message(f"确认使用 {method} 支付" if method else "确认继续")
            st.rerun()
        if cols[1].button("取消操作", width="stretch", key="cancel_action"):
            execute_message("取消，先不要执行")
            st.rerun()


def home_page():
    queued = st.session_state.pop("queued_prompt", None)
    if queued:
        execute_message(queued)
        st.rerun()
    page_header("工作台", "从选购到售后，让每一个消费需求都有下一步。")
    if st.session_state.get("active_run_id"):
        st.warning("上一项任务尚未结束，可以从已保存的执行步骤继续。")
        if st.button("恢复任务", key="resume_run", type="primary"):
            try:
                resume_run(st.session_state.conversation_id, st.session_state.active_run_id)
                st.session_state.archive_error = None
            except AgentError as exc:
                st.session_state.archive_error = str(exc)
            restore_conversation()
            st.rerun()
        if st.session_state.archive_error:
            st.error(st.session_state.archive_error)
        return
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        if not st.session_state.messages:
            with st.container(border=True, key="ws_panel_composer"):
                st.markdown("### 告诉助手你的目标")
                with st.form("start_request", clear_on_submit=True, border=False):
                    prompt = st.text_area("消费需求", placeholder="例如：推荐一款通勤降噪耳机，预算 500 元以内", label_visibility="collapsed", height=110)
                    sent = st.form_submit_button("交给助手", type="primary", width="stretch")
                if sent and prompt.strip():
                    execute_message(prompt.strip())
                    st.rerun()
                quick = st.columns(3)
                for col, label, text in zip(quick, ["按预算选购", "查物流", "检查退款资格"], ["推荐降噪耳机，预算 500 元以内", "查询订单 1002 的状态和物流", "查询订单 1001 是否可以退款"]):
                    if col.button(label, width="stretch", key=f"quick_{label}"):
                        execute_message(text)
                        st.rerun()
                st.caption("支持商品搜索、参数对比，以及订单、支付和售后查询。")
            with st.container(border=True, key="ws_panel_active_orders"):
                st.markdown("### 继续处理订单")
                for index, order in enumerate([o for o in list_orders() if o["status"] in {"placed", "shipped"}][:2]):
                    if index:
                        st.divider()
                    order_summary(order)
                    if st.button("查看状态与物流", key=f"home_order_{order['order_id']}", width="stretch"):
                        execute_message(f"查询订单 {order['order_id']} 的状态和物流")
                        st.rerun()
        else:
            with st.container(height=540, autoscroll=True, key="conversation"):
                for index, message in enumerate(st.session_state.messages):
                    with st.chat_message(message["role"], avatar=":material/person:" if message["role"] == "user" else ":material/auto_awesome:"):
                        st.markdown(message["content"])
                        if message["role"] == "assistant":
                            result = message["result"]
                            st.caption(f"{AGENTS.get(result.get('agent'), '消费助手')} · {STATUSES.get(result.get('status'), result.get('status'))}")
                            business_results(result)
                            with st.expander("查看执行过程"):
                                render_run(result)
                                if st.button("在 Agent Lab 中打开", key=f"run_detail_{index}"):
                                    view_run(result["run_id"])
            approval_controls()
            prompt = st.chat_input("继续提问，或补充你的需求…", key="workspace_chat")
            if prompt:
                execute_message(prompt)
                st.rerun()
    with right:
        with st.container(border=True, key="ws_panel_scenarios"):
            st.markdown("### 值得探索的选购场景")
            scenarios = [("通勤时听得更清楚", "降噪、佩戴方式与续航", "推荐适合通勤的降噪耳机，预算 500 元以内"),
                         ("打造舒适的办公桌", "低噪键盘与多设备连接", "推荐静音三模键盘，预算 400 元以内"),
                         ("给笔记本扩展接口", "视频输出与便携办公", "推荐 4K 扩展坞，预算 300 元以内")]
            for index, (title, subtitle, text) in enumerate(scenarios):
                if index:
                    st.divider()
                st.markdown(f'<div class="ws-prompt-title">{esc(title)}</div>', unsafe_allow_html=True)
                st.caption(subtitle)
                if st.button("让助手帮我选", key=f"scenario_{index}", width="stretch"):
                    execute_message(text)
                    st.rerun()
        with st.container(border=True, key="ws_panel_recent"):
            st.markdown("### 最近的任务")
            runs = list_runs(limit=2)
            if not runs:
                st.caption("处理一次需求后，结果会保存在这里，方便继续查看。")
            for run in runs:
                st.write(run["request"])
                st.caption(f"{local_time(run['created_at'])} · {STATUSES.get(run['result'].get('status'), '未知状态')}")
                if st.button("查看执行记录", key=f"home_run_{run['run_id']}", width="stretch"):
                    view_run(run["run_id"])
            st.page_link(st.session_state.workspace_pages["tasks"], label="全部任务记录", icon=":material/history:")
    if st.session_state.archive_error:
        st.warning("本次回复已保留，但任务记录保存失败。请检查本地数据库；不要重复提交已确认的支付或退款。")


def toggle_comparison(product_id):
    selected = list(st.session_state.compare_ids)
    if st.session_state[f"compare_{product_id}"]:
        if product_id not in selected:
            if len(selected) >= 3:
                st.session_state[f"compare_{product_id}"] = False
                st.toast("最多选择 3 款商品进行对比。")
                return
            selected.append(product_id)
    elif product_id in selected:
        selected.remove(product_id)
    st.session_state.compare_ids = selected


def clear_comparison():
    st.session_state.compare_ids = []
    for key in list(st.session_state):
        if key.startswith("compare_") and key != "compare_ids":
            st.session_state[key] = False


def products_page():
    page_header("商品发现", "数码与办公演示商品，先明确需求，再比较取舍。")
    filters = st.columns([2, 1, 1], gap="medium")
    query = filters[0].text_input("搜索商品", placeholder="商品名、编号或参数", key="catalog_query")
    category = filters[1].selectbox("品类", ["全部", "耳机", "键盘", "扩展坞"], key="catalog_category")
    budget = filters[2].number_input("预算上限（元，0 为不限）", min_value=0.0, value=1000.0, step=100.0, key="catalog_budget")
    stock_only = st.checkbox("只看有货商品", value=True, key="catalog_stock")
    signature = (query, category, budget, stock_only)
    if st.session_state.catalog_filter != signature:
        st.session_state.catalog_filter = signature
        st.session_state.product_page = 0
    output = tools.search_products(query=query, category=None if category == "全部" else category, max_price=budget or None, available_only=stock_only, limit=60)
    products = output["products"]
    st.caption(f"找到 {output['total']} 款商品 · 演示品牌、价格与规格，不是实时商品行情")
    selected = st.session_state.compare_ids
    if selected:
        with st.container(border=True, key="ws_panel_comparison"):
            st.markdown("### 商品对比")
            st.caption("已选：" + "、".join(selected))
            st.button("清空对比", key="clear_comparison", on_click=clear_comparison)
            if len(selected) >= 2:
                compared = tools.compare_products(selected)
                if compared.get("error"):
                    st.warning(compared["error"])
                else:
                    comparison_table(compared["products"])
                    if st.button("让助手对比这些商品", type="primary", key="ask_compare"):
                        ask_assistant("对比商品 " + " 和 ".join(selected))
            else:
                st.caption("再选一款商品即可比较。")
    if not products:
        st.info("没有符合条件的商品。可以清空关键词、调整品类或提高预算。")
        return
    last_page = (len(products) - 1) // 6
    page = min(st.session_state.product_page, last_page)
    for offset in range(0, 6, 3):
        columns = st.columns(3, gap="medium")
        for col, product in zip(columns, products[page * 6 + offset:page * 6 + offset + 3]):
            with col, st.container(border=True, key=f"ws_panel_product_{product['product_id']}"):
                product_summary(product)
                with st.expander("商品参数"):
                    for key, value in product["specs"].items():
                        st.write(f"**{key}**：{value}")
                st.checkbox("加入对比", value=product["product_id"] in selected, key=f"compare_{product['product_id']}", on_change=toggle_comparison, args=(product["product_id"],))
                if st.button("找同类商品", key=f"similar_{product['product_id']}", width="stretch"):
                    ask_assistant(f"推荐{product['category']}，预算 {product['price']:.2f} 元以内")
    pagination = st.columns([1, 2, 1], vertical_alignment="center")
    if pagination[0].button("上一页", disabled=page == 0, width="stretch"):
        st.session_state.product_page = page - 1
        st.rerun()
    pagination[1].caption(f"第 {page + 1} / {last_page + 1} 页 · 每页 6 款")
    if pagination[2].button("下一页", disabled=page == last_page, width="stretch"):
        st.session_state.product_page = page + 1
        st.rerun()


def orders_page():
    page_header("订单与售后", "查看订单、跟踪配送，或从具体订单发起售后咨询。")
    filters = st.columns([2, 1])
    query = filters[0].text_input("搜索订单", placeholder="订单号、商品名称或演示客户", key="order_query")
    status = filters[1].selectbox("订单状态", ["全部", "placed", "shipped", "delivered", "cancelled"], format_func=lambda v: STATUSES.get(v, v), key="order_status")
    orders = list_orders(query=query, status=None if status == "全部" else status)
    st.caption(f"{len(orders)} 条订单 · 此工作空间展示全部种子订单，未接入真实客户账户")
    if not orders:
        st.info("没有找到匹配的订单，请调整搜索条件。")
    for order in orders:
        with st.container(border=True, key=f"ws_panel_order_{order['order_id']}"):
            order_summary(order)
            st.caption(f"下单日期 {order['order_date']} · 支付：{STATUSES.get(order['payment_status'], '暂无记录')} · 售后：{STATUSES.get(order['refund_status'], '暂无记录')}")
            actions = st.columns(3)
            if actions[0].button("问助手 · 状态与物流", key=f"track_{order['order_id']}", width="stretch"):
                ask_assistant(f"查询订单 {order['order_id']} 的状态和物流")
            if actions[1].button("检查退款资格", key=f"refund_{order['order_id']}", width="stretch"):
                ask_assistant(f"查询订单 {order['order_id']} 是否可以退款")
            label = "准备支付" if order["payment_status"] == "pending" else "申请退款"
            if actions[2].button(label, key=f"action_{order['order_id']}", width="stretch", disabled=order["status"] == "cancelled"):
                ask_assistant(f"帮我支付订单 {order['order_id']}" if label == "准备支付" else f"订单 {order['order_id']} 商品有问题，我想退款")
            with st.expander("订单详情与物流时间线"):
                details = tools.get_order_details(order["order_id"])
                st.write(f"演示客户：{details['customer_name']}")
                if order["refund_reason"]:
                    st.write(f"售后原因：{order['refund_reason']}")
                shipment = tools.track_shipment(order["order_id"])
                if shipment.get("events"):
                    timeline([(e["description"], local_time(e["timestamp"])) for e in shipment["events"]])
                else:
                    st.caption("暂无已记录的物流节点。")


def tasks_page():
    page_header("任务记录", "每一次处理都有记录，查看结果或重新核验当前业务状态。")
    runs = list_runs()
    status = st.selectbox("处理状态", ["全部", "SUCCEEDED", "WAITING_APPROVAL", "WAITING_CLARIFICATION", "FAILED", "ESCALATED", "CANCELLED"], format_func=lambda v: STATUSES.get(v, v), key="task_status")
    filtered = [run for run in runs if status == "全部" or run["result"].get("status") == status]
    st.caption("演示工作空间最近 100 次真实执行记录；历史中的等待状态表示当次运行状态。")
    if not filtered:
        st.info("暂无匹配的任务。到工作台处理一次需求后，记录会自动出现在这里。")
        st.page_link(st.session_state.workspace_pages["home"], label="前往工作台", icon=":material/arrow_forward:")
    for run in filtered:
        result = run["result"]
        with st.container(border=True, key=f"ws_panel_task_{run['run_id']}"):
            st.markdown(f'<div class="ws-run-heading"><span>{esc(run["request"])}</span>{status_badge(result.get("status", "UNKNOWN"))}</div>', unsafe_allow_html=True)
            st.caption(f"{local_time(run['created_at'])} · {AGENTS.get(result.get('agent'), '消费助手')} · {run['run_id'][:12]}")
            st.markdown(result.get("reply", "暂无回复"))
            cols = st.columns(2)
            if cols[0].button("查看执行记录", key=f"task_lab_{run['run_id']}", width="stretch"):
                view_run(run["run_id"])
            if cols[1].button("重新核验这项需求", key=f"task_retry_{run['run_id']}", width="stretch"):
                prompt = run["request"]
                order_id = (result.get("pending_action") or {}).get("order_id")
                if order_id:
                    prompt = f"查询订单 {order_id} 的状态和支付状态，再判断是否可以退款"
                if prompt.strip() in {"确认", "确认继续", "取消，先不要执行"} or prompt.startswith("确认使用"):
                    st.info("这是一条历史确认记录，请从订单页面重新发起业务请求。")
                else:
                    ask_assistant(prompt)


def lab_page():
    page_header("Agent Lab", "查看真实执行路径、工具参数、运行结果与终止原因。")
    runs = list_runs()
    if not runs:
        st.info("尚无执行记录。先在工作台运行订单查询或商品搜索，再到这里查看。")
        st.page_link(st.session_state.workspace_pages["home"], label="前往工作台", icon=":material/arrow_forward:")
        return
    ids = [run["run_id"] for run in runs]
    selected = st.session_state.selected_run
    chosen = st.selectbox("选择任务", ids, index=ids.index(selected) if selected in ids else 0,
                          format_func=lambda value: next(f"{local_time(r['created_at'])} · {r['request']}" for r in runs if r['run_id'] == value))
    st.session_state.selected_run = chosen
    run = next(r for r in runs if r["run_id"] == chosen)
    st.caption(run["request"])
    with st.container(border=True, key="ws_panel_lab"):
        render_run(run["result"], detailed=True)
    with st.expander("任务理解结果"):
        st.json(run["result"].get("intake", {}))
