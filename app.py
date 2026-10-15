"""选序 consumer workspace with native sidebar navigation."""
from pathlib import Path

import streamlit as st

from src.ui.pages import home_page, init_state, lab_page, new_conversation, orders_page, products_page, tasks_page, preferences_page
from src.ui.styles import apply_styles

st.set_page_config(page_title="选序 · 消费工作空间", page_icon="🛍️", layout="wide", initial_sidebar_state="auto")
apply_styles()
init_state()
st.logo(str(Path(__file__).parent / "assets" / "brand.svg"), size="large")
pages = {
    "home": st.Page(home_page, title="工作台", icon=":material/home:", default=True),
    "products": st.Page(products_page, title="商品发现", icon=":material/search:", url_path="products"),
    "orders": st.Page(orders_page, title="订单与售后", icon=":material/package_2:", url_path="orders"),
    "tasks": st.Page(tasks_page, title="任务记录", icon=":material/history:", url_path="tasks"),
    "lab": st.Page(lab_page, title="Agent Lab", icon=":material/science:", url_path="lab"),
    "preferences": st.Page(preferences_page, title="我的偏好", icon=":material/tune:", url_path="preferences"),
}
st.session_state.workspace_pages = pages
selected_page = st.navigation({"消费工作空间": [pages[k] for k in ("home", "products", "orders", "tasks", "preferences")],
                               "开发与评测": [pages["lab"]]}, position="sidebar", expanded=True)
with st.sidebar:
    st.button("新建对话", icon=":material/add:", on_click=new_conversation, width="stretch", disabled=bool(st.session_state.memory.pending_action or st.session_state.get('active_run_id')))
    if st.session_state.memory.pending_action:
        st.caption("当前有待确认操作，请先在工作台确认或取消。")
        st.page_link(pages["home"], label="返回待确认操作", icon=":material/arrow_forward:")
    st.caption("数码 · 桌面办公")
    st.caption("本地演示数据，支付与退款只修改演示数据库。")
selected_page.run()
