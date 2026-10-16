"""LingCart Agent engineering console and executable consumer sandbox."""
from pathlib import Path

import streamlit as st

from src.ui.engineering_pages import console_page, datasets_page, experiments_page, mechanisms_page, runs_page
from src.ui.pages import home_page, init_state, new_conversation, orders_page, products_page, tasks_page, preferences_page
from src.ui.styles import apply_styles

st.set_page_config(page_title="LingCart · Agent Engineering", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
apply_styles()
init_state()
st.logo(str(Path(__file__).parent / "assets" / "brand.svg"), size="large")
pages = {
    "console": st.Page(console_page, title="Agent 控制台", icon=":material/space_dashboard:", default=True),
    "runs": st.Page(runs_page, title="运行检查", icon=":material/route:", url_path="runs"),
    "experiments": st.Page(experiments_page, title="实验契约", icon=":material/science:", url_path="experiments"),
    "mechanisms": st.Page(mechanisms_page, title="机制注册表", icon=":material/account_tree:", url_path="mechanisms"),
    "datasets": st.Page(datasets_page, title="数据集覆盖", icon=":material/dataset:", url_path="datasets"),
    "home": st.Page(home_page, title="消费工作台", icon=":material/storefront:", url_path="sandbox"),
    "products": st.Page(products_page, title="商品发现", icon=":material/search:", url_path="products"),
    "orders": st.Page(orders_page, title="订单与售后", icon=":material/package_2:", url_path="orders"),
    "tasks": st.Page(tasks_page, title="任务记录", icon=":material/history:", url_path="tasks"),
    "preferences": st.Page(preferences_page, title="我的偏好", icon=":material/tune:", url_path="preferences"),
}
st.session_state.workspace_pages = pages
selected_page = st.navigation({
    "Agent Engineering": [pages[k] for k in ("console", "runs", "experiments", "mechanisms", "datasets")],
    "业务沙盒": [pages[k] for k in ("home", "products", "orders", "tasks", "preferences")],
}, position="sidebar", expanded=True)
with st.sidebar:
    st.markdown('<div class="eng-sidebar-status"><i></i><span>LOCAL RUNTIME</span><b>CONNECTED</b></div>', unsafe_allow_html=True)
    st.button("新建业务对话", icon=":material/add:", on_click=new_conversation, width="stretch", disabled=bool(st.session_state.memory.pending_action or st.session_state.get('active_run_id')))
    if st.session_state.memory.pending_action:
        st.caption("当前有待确认操作，请先在业务沙盒确认或取消。")
        st.page_link(pages["home"], label="返回待确认操作", icon=":material/arrow_forward:")
    st.markdown('<div class="eng-sidebar-foot"><span>LINGCART / LAB BUILD</span><p>本地数据 · 可追溯执行 · 桌面工程界面</p></div>', unsafe_allow_html=True)
selected_page.run()
