"""Visual system for the desktop Agent engineering console."""
import streamlit as st


def apply_styles() -> None:
    st.markdown("""
<style>
:root {
  --canvas:#f3f5f8; --surface:#ffffff; --surface-2:#eef1f5; --ink:#172033; --ink-2:#33405a;
  --muted:#69758a; --line:#dce1e8; --line-strong:#bec7d4; --blue:#356ae6; --blue-soft:#eaf0ff;
  --violet:#7856d8; --violet-soft:#f0ecff; --teal:#16877a; --teal-soft:#e5f5f1;
  --amber:#c87816; --amber-soft:#fff3df; --red:#bc4b45; --red-soft:#ffebe9; --nav:#131b2b;
  --mono:"JetBrains Mono","SFMono-Regular",Consolas,"Liberation Mono",monospace;
  --sans:"IBM Plex Sans","Noto Sans SC","PingFang SC","Microsoft YaHei",sans-serif;
}
html,body,.stApp { min-width:1180px; font-family:var(--sans); color:var(--ink); }
.stApp,[data-testid="stMain"] {
  background-color:var(--canvas);
  background-image:linear-gradient(rgba(23,32,51,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(23,32,51,.025) 1px,transparent 1px);
  background-size:24px 24px;
}
[data-testid="stHeader"] { background:transparent; height:2.2rem; }
#MainMenu,footer { visibility:hidden; }
[data-testid="stMainBlockContainer"] { max-width:1500px; padding:2.1rem 3rem 4rem; }
[data-testid="stBottomBlockContainer"] { padding-left:3rem; padding-right:3rem; background:var(--canvas); }

/* Instrument housing: sidebar */
[data-testid="stSidebar"] { background:var(--nav); border-right:1px solid #28344a; width:260px!important; min-width:260px!important; max-width:260px!important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top:12px; }
[data-testid="stSidebar"] [data-testid="stSidebarNav"] { padding-top:8px; }
[data-testid="stSidebarNav"] a { border-radius:4px; margin:2px 0; padding:8px 11px; color:#aeb8ca; }
[data-testid="stSidebarNav"] a:hover { background:#1d2940; color:#fff; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background:#253451; color:#fff; box-shadow:inset 2px 0 0 #78a0ff; }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color:#fff; font-weight:600; }
[data-testid="stSidebarNav"] span { font-size:13px; }
[data-testid="stSidebarNav"] [data-testid="stMarkdownContainer"] p { color:#75829a; font-family:var(--mono); font-size:10px; letter-spacing:.09em; text-transform:uppercase; }
[data-testid="stSidebarNavSeparator"] { margin:15px 0; border-color:#2a3447; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top:16px; }
[data-testid="stSidebar"] [data-testid="stButton"] button { background:#1d2940; color:#eaf0fb; border-color:#34425c; }
[data-testid="stSidebar"] [data-testid="stButton"] button:hover { background:#253451; border-color:#6d8cc7; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color:#8692a8; }
.eng-sidebar-status { display:grid; grid-template-columns:8px 1fr auto; gap:8px; align-items:center; color:#8190aa; font:9px/1 var(--mono); letter-spacing:.08em; margin:1px 0 12px; }
.eng-sidebar-status i { width:7px; height:7px; background:#45c1a5; box-shadow:0 0 0 3px rgba(69,193,165,.12); }
.eng-sidebar-status b { color:#d6dfef; font-weight:500; }
.eng-sidebar-foot { border-top:1px solid #2a3447; margin-top:18px; padding-top:14px; }
.eng-sidebar-foot span { color:#738099; font:9px/1.2 var(--mono); letter-spacing:.09em; }
.eng-sidebar-foot p { color:#9aa6bb!important; font-size:11px!important; line-height:1.55!important; margin-top:7px!important; }

/* Global Streamlit controls */
h1,h2,h3,h4 { color:var(--ink); }
[data-testid="stMarkdownContainer"] h3 { font-size:16px; letter-spacing:-.01em; font-weight:650; padding-top:0; }
[data-testid="stMarkdownContainer"] p { font-size:14px; line-height:1.67; }
[data-testid="stCaptionContainer"] p { color:var(--muted); font-size:12px; }
[data-testid="stMarkdownContainer"] hr { margin:12px 0; border-color:var(--line); }
[data-testid="stButton"] button,[data-testid="stPageLink"] a { border-radius:4px; font-size:12px; min-height:36px; box-shadow:none; font-weight:550; }
[data-testid="stButton"] button[kind="secondary"] { background:var(--surface); border-color:var(--line-strong); color:var(--ink); }
[data-testid="stButton"] button[kind="secondary"]:hover { border-color:var(--blue); color:var(--blue); background:var(--blue-soft); }
[data-testid="stButton"] button[kind="primary"] { background:var(--blue); border-color:var(--blue); color:#fff; }
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea { background:var(--surface); color:var(--ink); }
[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"]>div { border-color:var(--line-strong); background:var(--surface); border-radius:4px; }
[data-baseweb="select"] span { font-size:13px; }
[data-testid="stExpander"] { border-color:var(--line); border-radius:4px; background:rgba(255,255,255,.82); }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:22px; border-bottom:1px solid var(--line); }
[data-testid="stTabs"] [data-baseweb="tab"] { height:42px; padding:0 2px; font-size:13px; }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--blue); }
[data-testid="stDataFrame"] { border:1px solid var(--line); }
[data-testid="stJson"] { font-family:var(--mono); font-size:11px; }
[data-testid="stAlert"] { border-radius:4px; }
code { font-family:var(--mono); }

/* Engineering page frame */
.eng-header { padding:2px 0 22px; border-bottom:1px solid var(--line-strong); margin-bottom:18px; }
.eng-eyebrow { display:flex; align-items:center; justify-content:space-between; font:10px/1.2 var(--mono); letter-spacing:.11em; color:var(--blue); text-transform:uppercase; }
.eng-live-marker { color:var(--muted); display:flex; align-items:center; gap:8px; }
.eng-live-marker i { width:7px; height:7px; background:var(--teal); display:inline-block; }
.eng-title { font-size:34px; line-height:1.12; font-weight:680; letter-spacing:-1.2px; margin-top:18px; }
.eng-subtitle { color:var(--muted); font-size:14px; max-width:820px; margin-top:9px; line-height:1.65; }
.eng-section-heading { display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid var(--line-strong); padding:0 0 9px; margin:24px 0 12px; color:var(--ink); font-size:14px; font-weight:650; }
.eng-section-heading div { display:flex; align-items:center; gap:9px; }
.eng-section-heading b { font:10px/1 var(--mono); color:var(--blue); border:1px solid #bfcdf4; padding:4px 5px; }
.eng-section-heading span { color:var(--muted); font:9px/1 var(--mono); letter-spacing:.08em; }

/* Source-backed telemetry strip */
.eng-telemetry { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid var(--line-strong); background:var(--surface); margin-bottom:18px; }
.eng-telemetry-cell { padding:15px 17px 13px; border-right:1px solid var(--line); position:relative; min-width:0; }
.eng-telemetry-cell:last-child { border-right:0; }
.eng-telemetry-cell:before { content:""; position:absolute; left:-1px; top:-1px; right:-1px; height:3px; background:var(--ink); }
.eng-telemetry-cell.blue:before { background:var(--blue); }.eng-telemetry-cell.teal:before { background:var(--teal); }
.eng-telemetry-cell.violet:before { background:var(--violet); }.eng-telemetry-cell.amber:before { background:var(--amber); }
.eng-telemetry-label { color:var(--muted); font:9px/1.3 var(--mono); letter-spacing:.08em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.eng-telemetry-value { font:600 22px/1.15 var(--mono); letter-spacing:-.04em; margin-top:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.eng-telemetry-note { color:var(--muted); font-size:11px; margin-top:6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

/* Console composition */
.eng-console-hero { display:grid; grid-template-columns:1.25fr 1fr; gap:48px; background:var(--ink); color:#f3f6fb; min-height:210px; padding:29px 32px; position:relative; overflow:hidden; }
.eng-console-hero:after { content:""; position:absolute; right:-45px; bottom:-120px; width:320px; height:320px; border:1px solid rgba(119,151,215,.25); transform:rotate(45deg); }
.eng-hero-index { color:#8ea1c1; font:9px/1 var(--mono); letter-spacing:.12em; }
.eng-hero-title { font-size:29px; line-height:1.3; letter-spacing:-.8px; font-weight:590; margin-top:31px; }
.eng-hero-copy { border-left:1px solid #41506a; padding-left:27px; position:relative; z-index:1; }
.eng-hero-copy p { color:#bec9da!important; margin:18px 0 28px!important; font-size:13px!important; }
.eng-hero-rule { display:grid; grid-template-columns:auto 1fr; gap:7px 13px; }
.eng-hero-rule span { color:#7283a2; font:9px/1.4 var(--mono); letter-spacing:.08em; }
.eng-hero-rule b { color:#eef3fb; font-size:12px; font-weight:500; }
[data-testid="stVerticalBlock"][class*="st-key-eng_panel_"],
[data-testid="stVerticalBlock"][class*="st-key-eng_research_"],
[data-testid="stVerticalBlock"][class*="st-key-eng_change_"] { border:1px solid var(--line)!important; border-radius:2px!important; background:rgba(255,255,255,.9); padding:18px; gap:.7rem; }
[data-testid="stVerticalBlock"][class*="st-key-eng_console_run_"] { border-bottom:1px solid var(--line); padding:4px 0 10px; }
.eng-panel-lead { display:flex; align-items:flex-start; justify-content:space-between; gap:15px; }
.eng-panel-lead span,.eng-candidate-kicker { color:var(--blue); font:9px/1 var(--mono); letter-spacing:.09em; }
.eng-panel-lead h3,.eng-candidate-kicker+h3 { margin:8px 0 0; font-size:17px; }
.eng-hash { color:var(--muted); font:9px/1.4 var(--mono); }
.eng-change-callout { border-left:3px solid var(--violet); background:var(--violet-soft); padding:10px 12px; margin:7px 0; display:flex; justify-content:space-between; }
.eng-change-callout span { color:#6a589b; font:9px/1.4 var(--mono); }.eng-change-callout b { font-size:12px; color:#5134a6; }
.eng-gateway-state { display:flex; align-items:center; gap:12px; padding-bottom:11px; border-bottom:1px solid var(--line); }
.eng-gateway-state i { width:9px; height:9px; background:var(--amber); box-shadow:0 0 0 4px var(--amber-soft); }
.eng-gateway-state.ok i { background:var(--teal); box-shadow:0 0 0 4px var(--teal-soft); }
.eng-gateway-state div { display:flex; justify-content:space-between; flex:1; align-items:center; }
.eng-gateway-state span,.eng-provider span { color:var(--muted); font:9px/1 var(--mono); letter-spacing:.06em; }
.eng-gateway-state b { font:11px/1 var(--mono); }.eng-provider { display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid var(--line); }
.eng-provider b { color:var(--teal); font:10px/1 var(--mono); }
.eng-empty-inline { color:var(--muted); border:1px dashed var(--line-strong); padding:13px; text-align:center; font:10px/1.3 var(--mono); letter-spacing:.07em; }

/* Mechanism fingerprint rail */
.eng-fingerprint-head { display:flex; justify-content:space-between; margin:15px 0 6px; color:var(--muted); font:9px/1 var(--mono); letter-spacing:.05em; }
.eng-mechanism-rail { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); border:1px solid var(--line-strong); }
.eng-mechanism { min-width:0; min-height:109px; padding:10px; border-right:1px solid var(--line); position:relative; background:#fbfcfd; }
.eng-mechanism:last-child { border-right:0; }
.eng-mechanism.active:before,.eng-mechanism.changed:before { content:""; position:absolute; left:-1px; top:-1px; right:-1px; height:3px; background:var(--blue); }
.eng-mechanism.changed { background:var(--violet-soft); }.eng-mechanism.changed:before { background:var(--violet); }
.eng-mechanism.empty { background:repeating-linear-gradient(-45deg,#f7f8fa,#f7f8fa 5px,#f0f2f5 5px,#f0f2f5 6px); }
.eng-slot-index { color:#9aa5b4; font:9px/1 var(--mono); }.eng-slot-label { color:var(--muted); font-size:10px; margin-top:11px; }
.eng-slot-id { font:600 11px/1.35 var(--mono); margin-top:7px; overflow-wrap:anywhere; }.eng-slot-version { color:var(--muted); font:9px/1 var(--mono); margin-top:7px; }
.eng-chip-row { display:flex; flex-wrap:wrap; gap:5px; margin-top:10px; }
.eng-chip { border:1px solid var(--line); background:var(--surface-2); color:var(--ink-2); padding:4px 6px; font:9px/1 var(--mono); }
.eng-compare-label { display:inline-block; padding:4px 6px; background:var(--blue-soft); color:var(--blue); font:9px/1 var(--mono); letter-spacing:.07em; }
.eng-compare-label.candidate { background:var(--violet-soft); color:var(--violet); }

/* Run inspector */
.eng-run-row { padding:8px 4px 5px 0; }.eng-run-status-line { display:flex; align-items:center; gap:9px; }
.eng-mono { color:var(--blue); font:10px/1 var(--mono); }.eng-run-request { color:var(--ink); font-size:13px; font-weight:600; margin-top:8px; }
.eng-run-meta { color:var(--muted); font:9px/1.4 var(--mono); margin-top:5px; }
.eng-run-prompt { background:var(--surface); border-left:3px solid var(--ink); border-top:1px solid var(--line); border-right:1px solid var(--line); border-bottom:1px solid var(--line); padding:13px 15px; font-size:13px; font-weight:550; }
.eng-spine { display:flex; overflow-x:auto; padding:23px 6px 18px; background:rgba(255,255,255,.72); border:1px solid var(--line); border-top:0; }
.eng-spine-node { min-width:132px; flex:1; position:relative; padding:18px 10px 0 0; }
.eng-spine-node:before { content:""; position:absolute; left:9px; right:-1px; top:5px; height:1px; background:var(--line-strong); }
.eng-spine-node:last-child:before { right:calc(100% - 9px); }
.eng-spine-dot { position:absolute; left:4px; top:0; width:11px; height:11px; background:var(--blue); border:3px solid #dce6ff; box-sizing:border-box; z-index:1; }
.eng-spine-node.terminal .eng-spine-dot { background:var(--teal); border-color:#d4eee9; }.eng-spine-node.error .eng-spine-dot { background:var(--red); border-color:#f7d8d5; }
.eng-spine-seq { color:var(--muted); font:8px/1 var(--mono); letter-spacing:.05em; }.eng-spine-label { font-size:10px; font-weight:600; margin-top:7px; max-width:120px; }
.eng-definition-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid var(--line); background:var(--surface); margin:4px 0 12px; }
.eng-definition-grid>div { min-width:0; padding:11px 12px; border-right:1px solid var(--line); }.eng-definition-grid>div:last-child { border-right:0; }
.eng-definition-grid span { display:block; color:var(--muted); font:8px/1 var(--mono); letter-spacing:.07em; }
.eng-definition-grid b { display:block; font:550 11px/1.35 var(--mono); margin-top:7px; overflow-wrap:anywhere; }

/* Dataset and experiment surfaces */
.eng-coverage { border:1px solid var(--line); background:var(--surface); padding:10px 14px; }
.eng-coverage-row { padding:8px 0; border-bottom:1px solid var(--line); }.eng-coverage-row:last-child { border-bottom:0; }
.eng-coverage-row>div:first-child { display:flex; justify-content:space-between; font-size:11px; }.eng-coverage-row b { font:10px/1 var(--mono); }
.eng-coverage-track { height:3px; background:var(--surface-2); margin:7px 0 5px; }.eng-coverage-track i { display:block; height:3px; background:var(--blue); }
.eng-coverage-row code { color:var(--muted); font-size:8px; background:transparent; padding:0; }
.eng-experiment-pipeline { display:flex; align-items:center; border:1px solid var(--line-strong); background:var(--surface); padding:17px; margin:17px 0 4px; }
.eng-experiment-pipeline>div { flex:1; min-height:72px; }.eng-experiment-pipeline>i { width:34px; height:1px; background:var(--line-strong); margin:0 10px; }
.eng-experiment-pipeline span { color:#9ba5b4; font:9px/1 var(--mono); }.eng-experiment-pipeline b { display:block; font:600 11px/1 var(--mono); margin-top:10px; }
.eng-experiment-pipeline p { color:var(--muted)!important; font-size:10px!important; margin:7px 0 0!important; }
.eng-experiment-pipeline .done span,.eng-experiment-pipeline .done b { color:var(--teal); }.eng-experiment-pipeline .current span,.eng-experiment-pipeline .current b { color:var(--violet); }
.eng-hypothesis { border-left:3px solid var(--violet); background:var(--violet-soft); padding:13px 16px; margin:12px 0; }
.eng-hypothesis span { color:var(--violet); font:9px/1 var(--mono); letter-spacing:.08em; }.eng-hypothesis p { margin:8px 0 0!important; color:#3f3560!important; }
.eng-kernel { height:100%; min-height:152px; border:1px solid var(--line); background:rgba(255,255,255,.84); padding:14px; box-sizing:border-box; }
.eng-kernel span { color:var(--blue); font:8px/1.35 var(--mono); overflow-wrap:anywhere; }.eng-kernel b { display:block; font-size:13px; margin-top:11px; }
.eng-kernel p { color:var(--muted)!important; font:8px/1.7 var(--mono)!important; margin-top:14px!important; overflow-wrap:anywhere; }

/* Consumer sandbox keeps the same system, with quieter business components. */
.ws-breadcrumb { color:var(--muted); font-size:11px; margin-bottom:16px; display:flex; justify-content:space-between; }
.ws-demo { font:9px/1 var(--mono); padding:4px 7px; border:1px solid var(--line-strong); background:var(--surface); }
.ws-title { font-size:30px; font-weight:680; letter-spacing:-.8px; }.ws-subtitle { color:var(--muted); font-size:14px; margin:6px 0 22px; }
[data-testid="stVerticalBlock"][class*="st-key-ws_panel_"] { border-color:var(--line)!important; border-radius:4px!important; background:var(--surface); padding:19px; gap:.75rem; }
[data-testid="stChatInput"] { background:var(--surface); border:1px solid var(--line-strong); border-radius:4px; box-shadow:0 5px 18px rgba(23,32,51,.04); }
[data-testid="stChatMessage"] { background:var(--surface); padding:16px; border-radius:4px; border:1px solid var(--line); }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:var(--blue-soft); border-color:#cad7f7; }
.ws-order { display:flex; align-items:center; gap:12px; }.ws-order-icon { width:38px; height:38px; display:grid; place-items:center; color:var(--blue); background:var(--blue-soft); }
.ws-order-name { font-size:14px; font-weight:600; }.ws-order-meta,.ws-category,.ws-product-id { font-size:11px; color:var(--muted); margin-top:4px; }
.ws-product-id { font-family:var(--mono); }.ws-order-heading,.ws-run-heading { display:flex; align-items:center; justify-content:space-between; gap:12px; }
.ws-status { display:inline-block; color:var(--muted); background:var(--surface-2); padding:3px 7px; border-radius:2px; font:9px/1.2 var(--mono); white-space:nowrap; }
.ws-status.ok { background:var(--teal-soft); color:var(--teal); }.ws-status.warn { background:var(--amber-soft); color:#9b5c12; }.ws-status.bad { background:var(--red-soft); color:var(--red); }
.ws-price { font-size:22px; font-weight:650; font-variant-numeric:tabular-nums; margin:8px 0 4px; }.ws-product-name { font-size:16px; font-weight:650; margin:10px 0 5px; }
.ws-product-desc { color:var(--muted); font-size:12px; min-height:38px; }.ws-product-visual { background:var(--surface-2); height:134px; display:flex; align-items:center; justify-content:center; }
.ws-product-visual svg { width:100%; height:124px; }.ws-timeline { list-style:none; border-left:1px solid var(--line-strong); padding:0 0 0 18px; margin:12px 0 8px 6px; }
.ws-timeline li { position:relative; padding-bottom:19px; font-size:13px; }.ws-timeline li:last-child { padding-bottom:0; }.ws-timeline li:before { content:""; width:7px; height:7px; background:var(--blue); position:absolute; left:-22px; top:6px; }
.ws-timeline-time { color:var(--muted); font-size:11px; margin-top:3px; }.ws-run-heading { font-size:15px; font-weight:600; }.ws-prompt-title { font-size:14px; font-weight:600; margin-bottom:4px; }
.ws-result-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; margin:12px 0; }.ws-field-label { font-size:11px; color:var(--muted); }.ws-field-value { font-size:13px; font-weight:500; margin-top:3px; }
</style>
""", unsafe_allow_html=True)
