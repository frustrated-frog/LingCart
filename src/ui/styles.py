import streamlit as st


def apply_styles() -> None:
    st.markdown("""
<style>
:root { --ws-bg:#f5f7fb; --ws-surface:#fff; --ws-text:#202c40; --ws-secondary:#63718a;
 --ws-line:#e5eaf1; --ws-brand:#315efa; --ws-soft:#eef3ff; --ws-ok:#24765b; --ws-warn:#9a641c; }
html,body,.stApp { font-family:"PingFang SC","Microsoft YaHei",sans-serif; color:var(--ws-text); }
.stApp, [data-testid="stMain"] { background:var(--ws-bg); }
[data-testid="stHeader"] { background:transparent; height:3rem; }
#MainMenu,footer { visibility:hidden; }
[data-testid="stMainBlockContainer"] { max-width:1320px; padding:2.2rem 2.4rem 3rem; }
[data-testid="stSidebar"] { background:var(--ws-surface); border-right:1px solid var(--ws-line); min-width:228px; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top:12px; }
[data-testid="stSidebarNav"] a { border-radius:8px; margin:4px 0; padding:9px 12px; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background:var(--ws-soft); color:var(--ws-brand); }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color:var(--ws-brand); font-weight:600; }
[data-testid="stSidebarNav"] span { font-size:14px; }
[data-testid="stSidebarNavSeparator"] { margin:18px 0; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top:20px; }
h1,h2,h3 { color:var(--ws-text); }
[data-testid="stMarkdownContainer"] h3 { font-size:16px; letter-spacing:0; font-weight:650; padding-top:0; }
[data-testid="stMarkdownContainer"] p { font-size:14px; line-height:1.65; }
[data-testid="stCaptionContainer"] p { color:var(--ws-secondary); font-size:12px; }
.ws-brand { display:flex; align-items:center; gap:10px; padding:0 0 15px; }
.ws-logo { display:grid; place-items:center; width:34px; height:34px; border-radius:10px; background:var(--ws-brand); color:white; font-size:17px; font-weight:700; }
.ws-brand-name { font-size:19px; font-weight:700; }
.ws-brand-note { color:var(--ws-secondary); font-size:11px; margin-top:2px; }
.ws-breadcrumb { color:var(--ws-secondary); font-size:12px; margin-bottom:18px; display:flex; align-items:center; justify-content:space-between; gap:8px; }
.ws-demo { font-size:11px; padding:3px 8px; color:var(--ws-secondary); border:1px solid var(--ws-line); border-radius:5px; background:var(--ws-surface); }
.ws-title { font-size:28px; font-weight:700; letter-spacing:-.7px; margin:0 0 6px; }
.ws-subtitle { color:var(--ws-secondary); font-size:14px; margin-bottom:24px; }
[data-testid="stVerticalBlock"][class*="st-key-ws_panel_"] { border-color:var(--ws-line)!important; border-radius:12px!important; background:var(--ws-surface); padding:20px; gap:.75rem; }
[data-testid="stMarkdownContainer"] hr { margin:12px 0; border-color:var(--ws-line); }
[data-testid="stButton"] button, [data-testid="stPageLink"] a { border-radius:8px; font-size:13px; min-height:38px; box-shadow:none; }
[data-testid="stButton"] button[kind="secondary"] { background:var(--ws-surface); border-color:var(--ws-line); color:var(--ws-text); }
[data-testid="stButton"] button[kind="secondary"]:hover { border-color:var(--ws-brand); color:var(--ws-brand); background:var(--ws-soft); }
[data-testid="stButton"] button[kind="primary"] { background:var(--ws-brand); border-color:var(--ws-brand); color:white; }
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea { background:var(--ws-surface); color:var(--ws-text); }
[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"]>div { border-color:var(--ws-line); background:var(--ws-surface); }
[data-testid="stChatInput"] { background:var(--ws-surface); border:1px solid #d7dfeb; border-radius:12px; box-shadow:0 4px 16px rgba(32,44,64,.04); }
[data-testid="stChatInput"] textarea { color:var(--ws-text); background:var(--ws-surface); font-size:15px; }
[data-testid="stChatInput"] textarea::placeholder { color:var(--ws-secondary); }
[data-testid="stChatMessage"] { background:var(--ws-surface); padding:16px; border-radius:12px; border:1px solid var(--ws-line); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:var(--ws-soft); border-color:#dce5ff; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { font-size:15px; }
.ws-section-title { font-size:15px; font-weight:650; margin-bottom:4px; }
.ws-order { display:flex; align-items:center; gap:12px; }
.ws-order-icon { width:40px; height:40px; border-radius:9px; display:grid; place-items:center; color:var(--ws-brand); background:var(--ws-soft); flex-shrink:0; font-size:19px; }
.ws-order-name { font-size:14px; font-weight:600; }
.ws-order-meta { font-size:12px; color:var(--ws-secondary); margin-top:4px; }
.ws-status { display:inline-block; color:var(--ws-secondary); background:#f1f4f8; padding:3px 8px; border-radius:5px; font-size:11px; white-space:nowrap; }
.ws-status.ok { background:#eaf6f0; color:var(--ws-ok); }
.ws-status.warn { background:#fff4df; color:var(--ws-warn); }
.ws-status.bad { background:#fff0ee; color:#b63b32; }
.ws-order-heading { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:8px; }
.ws-price { font-size:22px; font-weight:650; font-variant-numeric:tabular-nums; margin:8px 0 4px; }
.ws-product-name { font-size:16px; font-weight:650; margin:10px 0 5px; }
.ws-product-desc { color:var(--ws-secondary); font-size:12px; min-height:38px; }
.ws-product-visual { background:#f6f8fd; border-radius:8px; height:134px; display:flex; align-items:center; justify-content:center; }
.ws-product-visual svg { width:100%; height:124px; }
.ws-category { font-size:11px; color:var(--ws-secondary); margin-top:12px; }
.ws-product-id { font-size:11px; color:var(--ws-secondary); font-family:ui-monospace,monospace; }
.ws-timeline { list-style:none; border-left:2px solid var(--ws-line); padding:0 0 0 18px; margin:12px 0 8px 6px; }
.ws-timeline li { position:relative; padding-bottom:19px; font-size:13px; }
.ws-timeline li:last-child { padding-bottom:0; }
.ws-timeline li:before { content:""; width:8px; height:8px; border-radius:50%; background:var(--ws-brand); position:absolute; left:-23px; top:6px; }
.ws-timeline-time { color:var(--ws-secondary); font-size:11px; margin-top:3px; }
.ws-run-heading { display:flex; align-items:center; justify-content:space-between; gap:12px; font-size:15px; font-weight:600; }
.ws-result-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin:12px 0; }
.ws-field-label { font-size:11px; color:var(--ws-secondary); }
.ws-field-value { font-size:13px; font-weight:500; margin-top:3px; }
.ws-prompt-title { font-size:14px; font-weight:600; margin-bottom:4px; }
[data-testid="stExpander"] { border-color:var(--ws-line); border-radius:9px; background:var(--ws-surface); }
[data-testid="stMetricValue"] { font-size:23px; font-weight:600; }
[data-testid="stBottomBlockContainer"] { padding-left:2.4rem; padding-right:2.4rem; background:var(--ws-bg); }
@media(max-width:900px) { [data-testid="stMainBlockContainer"] { padding:2rem 1.25rem; } .ws-title { font-size:25px; } }
@media(min-width:641px) { [data-testid="stSidebar"] { width:230px!important; min-width:230px!important; max-width:230px!important; } }
@media(max-width:640px) {
 [data-testid="stMainBlockContainer"] { padding:2.8rem 1rem 2rem; }
 [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
 [data-testid="stColumn"] { min-width:min(100%,260px)!important; flex:1 1 100%!important; width:100%!important; }
 .ws-title { font-size:23px; } .ws-result-grid { grid-template-columns:1fr; }
 .ws-order-heading { flex-wrap:wrap; }
 [data-testid="stBottomBlockContainer"] { padding-left:1rem; padding-right:1rem; }
}
</style>
""", unsafe_allow_html=True)
