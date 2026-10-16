"""Desktop engineering surfaces for operating and studying the Agent runtime."""
from __future__ import annotations

import streamlit as st

from src.persistence import RunStore
from src.session_control import ControlStore
from src.ui.components import (
    AGENTS,
    STATUSES,
    TOOLS,
    definition_grid,
    engineering_header,
    esc,
    event_label,
    execution_spine,
    local_time,
    mechanism_rail,
    section_heading,
    status_badge,
    telemetry_strip,
)
from src.ui.engineering_data import (
    CATEGORY_LABELS,
    RESEARCH_STATUS_LABELS,
    SLOT_LABELS,
    SLOT_ORDER,
    dataset_catalog,
    engineering_overview,
    experiment_catalog,
    profile_catalog,
    run_catalog,
)


def _page_link(key: str, label: str, icon: str) -> None:
    pages = st.session_state.get("workspace_pages", {})
    if key in pages:
        st.page_link(pages[key], label=label, icon=icon)


def _profile(profiles: list[dict], profile_id: str, *, fallback: int = 0) -> dict:
    return next((item for item in profiles if item["profile_id"] == profile_id), profiles[fallback])


def _capability_chips(capabilities: list[str]) -> str:
    return '<div class="eng-chip-row">' + "".join(
        f'<span class="eng-chip">{esc(item)}</span>' for item in capabilities
    ) + '</div>'


def _run_summary_row(run: dict) -> None:
    profile_id = run["profile"]["profile_id"]
    st.markdown(
        f'<div class="eng-run-row"><div class="eng-run-status-line">'
        f'<span class="eng-mono">{esc(run["run_id"][:10])}</span>{status_badge(run["status"])}</div>'
        f'<div class="eng-run-request">{esc(run["request"])}</div>'
        f'<div class="eng-run-meta">{esc(local_time(run["created_at"]))} · {esc(profile_id)} · '
        f'{run["event_count"]} events · {run["tool_calls"]} tools</div></div>',
        unsafe_allow_html=True,
    )


def console_page() -> None:
    overview = engineering_overview()
    profiles = overview["profiles"]
    counts = overview["counts"]
    model = overview["model"]
    engineering_header(
        "AGENT CONSOLE / 00",
        "Agent 工程控制台",
        "观察运行、比较机制、验证假设。这里展示的是当前代码和本地数据的真实状态。",
        marker="RUNTIME CONNECTED",
    )

    production = _profile(profiles, "adaptive-v1")
    candidate = next((item for item in profiles if item["research"]["status"] == "candidate"), None)
    model_state = "READY" if model.get("ready") else "NOT READY"
    model_note = "已实时验证" if model.get("live_verified") else (
        "配置已加载" if model.get("configured") else "等待配置"
    )
    telemetry_strip([
        ("HARNESS PROFILES", str(counts["profiles"]), f'{counts["candidate_profiles"]} 个候选', "blue"),
        ("EVALUATION CASES", str(counts["dataset_cases"]), f'{counts["dataset_turns"]} turns', "teal"),
        ("RECORDED RUNS", str(counts["runs"]), f'{counts["events"]} events', "ink"),
        ("MODEL GATEWAY", model_state, model_note, "violet" if model.get("ready") else "amber"),
    ])

    st.markdown(
        '<div class="eng-console-hero">'
        '<div><div class="eng-hero-index">THESIS / 01</div>'
        '<div class="eng-hero-title">让 Agent 的每次进步<br>都能被隔离、复现与证伪。</div></div>'
        '<div class="eng-hero-copy">'
        '<p>稳定内核负责状态、安全边界与恢复；Harness Profile 负责装配可替换机制；'
        '实验契约只允许一个清晰变量进入比较。</p>'
        '<div class="eng-hero-rule"><span>PRODUCT</span><b>稳定交付</b><span>RESEARCH</span><b>受控演化</b></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.72, 1], gap="large")
    with left:
        section_heading("01", "当前自主链路", "PROFILE FINGERPRINT")
        with st.container(key="eng_panel_current_profile"):
            st.markdown(
                f'<div class="eng-panel-lead"><div><span>PRODUCTION BASELINE</span>'
                f'<h3>{esc(production["display_name"])}</h3></div>'
                f'<div class="eng-hash">SHA {esc(production["execution_sha256"][:12])}</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(production["description"])
            mechanism_rail(production)
            st.markdown(_capability_chips(production["capabilities"]), unsafe_allow_html=True)
        section_heading("02", "最近执行", "LATEST RUN ARCHIVE")
        if not overview["runs"]:
            st.info("还没有可检查的运行。业务沙盒产生的下一次任务会自动进入这里。")
        for run in overview["runs"][:4]:
            with st.container(key=f'eng_console_run_{run["run_id"]}'):
                row, action = st.columns([5, 1], vertical_alignment="center")
                with row:
                    _run_summary_row(run)
                if action.button("检查", key=f'eng_open_run_{run["run_id"]}', width="stretch"):
                    st.session_state.selected_run = run["run_id"]
                    if "runs" in st.session_state.get("workspace_pages", {}):
                        st.switch_page(st.session_state.workspace_pages["runs"])
        _page_link("runs", "打开完整 Run Inspector", ":material/arrow_forward:")

    with right:
        section_heading("03", "实验队列", "ONE VARIABLE AT A TIME")
        with st.container(key="eng_panel_experiment"):
            if candidate:
                st.markdown(
                    f'<div class="eng-candidate-kicker">CANDIDATE · '
                    f'{esc(candidate["profile_id"])}</div><h3>{esc(candidate["display_name"])}</h3>',
                    unsafe_allow_html=True,
                )
                st.write(candidate["research"].get("hypothesis") or "尚未声明假设")
                changed = [slot for slot in SLOT_ORDER
                           if candidate["mechanisms"].get(slot) != production["mechanisms"].get(slot)]
                st.markdown(
                    '<div class="eng-change-callout"><span>CHANGED SLOT</span>'
                    f'<b>{esc(" / ".join(SLOT_LABELS[item] for item in changed) or "跨 Profile 对照")}</b></div>',
                    unsafe_allow_html=True,
                )
                _page_link("experiments", "查看实验契约", ":material/science:")
            else:
                st.caption("Registry 中没有候选 Profile。")

        section_heading("04", "模型接入", "SANITIZED STATUS")
        with st.container(key="eng_panel_model"):
            state_class = "ok" if model.get("ready") else "warn"
            st.markdown(
                f'<div class="eng-gateway-state {state_class}"><i></i><div>'
                f'<span>MODEL GATEWAY</span><b>{esc(model_state)}</b></div></div>',
                unsafe_allow_html=True,
            )
            st.caption(model.get("message", "未读取到模型网关状态"))
            providers = model.get("providers", [])
            if providers:
                for provider in providers:
                    st.markdown(
                        f'<div class="eng-provider"><span>{esc(provider.get("provider", "unknown"))}'
                        f' / {esc(provider.get("model", "unknown"))}</span>'
                        f'<b>{esc(provider.get("health", "UNKNOWN"))}</b></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown('<div class="eng-empty-inline">NO PROVIDER CONFIGURED</div>', unsafe_allow_html=True)

    section_heading("05", "稳定内核边界", "CHANGES REQUIRE COMPATIBILITY REVIEW")
    kernel_columns = st.columns(len(overview["kernel"]), gap="small")
    for column, boundary in zip(kernel_columns, overview["kernel"]):
        with column:
            paths = "<br>".join(esc(path) for path in boundary["paths"])
            st.markdown(
                f'<div class="eng-kernel"><span>{esc(boundary["boundary_id"])}</span>'
                f'<b>{esc(boundary["label"])}</b><p>{paths}</p></div>',
                unsafe_allow_html=True,
            )


def runs_page() -> None:
    runs = run_catalog()
    engineering_header(
        "LIVE RUNS / 01",
        "Run Inspector",
        "沿执行主脊检查一次任务：配置身份、事件顺序、证据状态、工具调用和终止原因。",
    )
    if not runs:
        st.info("尚无运行记录。请先在业务沙盒执行一个消费任务。")
        _page_link("home", "前往消费工作台", ":material/storefront:")
        return

    statuses = sorted({run["status"] for run in runs})
    profiles = sorted({run["profile"]["profile_id"] for run in runs})
    filters = st.columns([1, 1, 2.2], gap="medium")
    selected_status = filters[0].selectbox(
        "状态", ["ALL", *statuses], format_func=lambda value: "全部状态" if value == "ALL" else STATUSES.get(value, value)
    )
    selected_profile = filters[1].selectbox(
        "Profile", ["ALL", *profiles], format_func=lambda value: "全部 Profile" if value == "ALL" else value
    )
    filtered = [run for run in runs if (selected_status == "ALL" or run["status"] == selected_status)
                and (selected_profile == "ALL" or run["profile"]["profile_id"] == selected_profile)]
    if not filtered:
        st.warning("当前筛选没有运行记录。")
        return
    ids = [run["run_id"] for run in filtered]
    selected = st.session_state.get("selected_run")
    chosen = filters[2].selectbox(
        "运行", ids, index=ids.index(selected) if selected in ids else 0,
        format_func=lambda value: next(
            f'{local_time(item["created_at"])} · {item["request"]}' for item in filtered if item["run_id"] == value
        ),
    )
    st.session_state.selected_run = chosen
    run = next(item for item in filtered if item["run_id"] == chosen)
    result = run["result"]
    events = result.get("events", [])

    telemetry_strip([
        ("RUN STATUS", STATUSES.get(run["status"], run["status"]), result.get("terminal_reason") or "尚未终止", "teal"),
        ("HARNESS PROFILE", run["profile"]["profile_id"], "已记录" if not run["profile"].get("legacy_profile_reference") else "历史运行未记录", "violet"),
        ("TOOL / MODEL", f'{run["tool_calls"]} / {run["model_calls"]}', f'{run["event_count"]} total events', "blue"),
        ("TOKEN USAGE", str(run["tokens"]) if run["tokens"] is not None else "N/A", "Provider 未报告时不伪造", "ink"),
    ])
    section_heading("01", "Execution Spine", f'{run["run_id"][:12]} / {local_time(run["created_at"])}')
    st.markdown(f'<div class="eng-run-prompt">{esc(run["request"])}</div>', unsafe_allow_html=True)
    execution_spine(events)

    if events:
        event_indexes = list(range(len(events)))
        selected_event = st.selectbox(
            "事件探针", event_indexes,
            format_func=lambda index: f'{events[index].get("sequence", index + 1):02d} · {event_label(events[index])[0]} · {event_label(events[index])[1]}',
        )
        with st.container(key="eng_panel_event_probe"):
            event = events[selected_event]
            code, label = event_label(event)
            definition_grid([
                ("EVENT TYPE", event.get("type", "event")),
                ("SEMANTIC", f"{code} / {label}"),
                ("TIMESTAMP", event.get("timestamp", "—")),
                ("SEQUENCE", str(event.get("sequence", selected_event + 1))),
            ])
            st.json(event)

    saved = RunStore().run(chosen)
    tabs = st.tabs(["结果与计划", "上下文与证据", "工具账本", "原始记录"])
    with tabs[0]:
        section_heading("A", "最终结果")
        st.markdown(result.get("reply") or "该运行尚未生成最终回复。")
        plan = result.get("plan") or {}
        if plan.get("steps"):
            st.dataframe([
                {
                    "step": step.get("step_id"),
                    "specialist": AGENTS.get(step.get("specialist"), step.get("specialist")),
                    "operation": TOOLS.get(step.get("tool"), step.get("tool")),
                    "depends_on": ", ".join(step.get("depends_on", [])) or "—",
                    "status": STATUSES.get(step.get("status"), step.get("status")),
                }
                for step in plan["steps"]
            ], hide_index=True, width="stretch")
        else:
            st.caption("该路径没有记录结构化计划。")
        if result.get("error"):
            st.error(result["error"].get("message", str(result["error"])))
    with tabs[1]:
        checkpoint = (saved or {}).get("checkpoint", {})
        adaptive = checkpoint.get("adaptive", {})
        definition_grid([
            ("GOAL REVISION", str(adaptive.get("binding_revision", 0))),
            ("EVIDENCE READ", str(len(adaptive.get("read_evidence_refs", [])))),
            ("OBJECTIVES", str(len(adaptive.get("objective_ledger", [])))),
            ("CONTEXT", "RECORDED" if checkpoint.get("context") else "NOT RECORDED"),
        ])
        context_tab, evidence_tab, audit_tab = st.tabs(["结构化上下文", "目标与证据", "独立审查"])
        with context_tab:
            st.json(checkpoint.get("context", {}))
        with evidence_tab:
            st.json({
                "goal": adaptive.get("goal"),
                "objective_ledger": adaptive.get("objective_ledger", []),
                "constraint_bindings": adaptive.get("constraint_bindings", {}),
                "working_memory": adaptive.get("working_memory"),
                "read_evidence_refs": adaptive.get("read_evidence_refs", []),
            })
        with audit_tab:
            st.json({
                "request_audit": result.get("request_audit"),
                "binding_audit": result.get("binding_audit"),
                "completion_audit": result.get("completion_audit"),
                "verification": plan.get("verification"),
            })
    with tabs[2]:
        trace = result.get("trace", [])
        outcomes = RunStore().outcomes(chosen) if saved else []
        definition_grid([
            ("TRACE ENTRIES", str(len(trace))),
            ("DURABLE LEDGER", str(len(outcomes))),
        ])
        if not trace and not outcomes:
            st.caption("该运行没有工具调用。")
        for index, entry in enumerate(trace):
            with st.expander(f'{index + 1:02d} / {TOOLS.get(entry.get("tool"), entry.get("tool", "tool"))}'):
                args_col, result_col = st.columns(2)
                args_col.caption("ARGS")
                args_col.json(entry.get("args", {}))
                result_col.caption("RESULT")
                result_col.json(entry.get("result"))
        if outcomes:
            with st.expander("持久化 Tool Ledger"):
                st.json(outcomes)
    with tabs[3]:
        st.json({
            "run_id": chosen,
            "harness": run["profile"],
            "events": events,
            "budget": result.get("budget"),
            "usage": result.get("usage"),
            "controls": ControlStore().commands(saved["session_id"], chosen) if saved else [],
        })


def mechanisms_page() -> None:
    profiles = profile_catalog()
    engineering_header(
        "MECHANISM REGISTRY / 02",
        "Mechanism Registry",
        "Profile 是可执行配置，不是标签。比较槽位、源码指纹和研究状态，确认究竟改变了什么。",
    )
    if not profiles:
        st.error("Registry 中没有可执行 Profile。")
        return
    ids = [item["profile_id"] for item in profiles]
    controls = st.columns(2)
    baseline_id = controls[0].selectbox(
        "Reference Profile", ids, index=ids.index("adaptive-v1") if "adaptive-v1" in ids else 0,
    )
    candidate_default = "adaptive-objective-context-v1"
    candidate_id = controls[1].selectbox(
        "Comparison Profile", ids, index=ids.index(candidate_default) if candidate_default in ids else min(1, len(ids) - 1),
    )
    baseline = _profile(profiles, baseline_id)
    candidate = _profile(profiles, candidate_id)
    section_heading("01", "机制指纹对照", "CHANGED SLOTS ARE HIGHLIGHTED")
    left, right = st.columns(2, gap="large")
    with left, st.container(key="eng_panel_profile_reference"):
        st.markdown(f'<div class="eng-compare-label">REFERENCE</div><h3>{esc(baseline["display_name"])}</h3>', unsafe_allow_html=True)
        mechanism_rail(baseline)
    with right, st.container(key="eng_panel_profile_candidate"):
        st.markdown(f'<div class="eng-compare-label candidate">COMPARISON</div><h3>{esc(candidate["display_name"])}</h3>', unsafe_allow_html=True)
        mechanism_rail(candidate, baseline)

    rows = []
    for slot in SLOT_ORDER:
        before = baseline["mechanisms"].get(slot)
        after = candidate["mechanisms"].get(slot)
        before_id = f'{before["id"]}@{before["version"]}' if before else "unbound"
        after_id = f'{after["id"]}@{after["version"]}' if after else "unbound"
        rows.append({"slot": slot, "槽位": SLOT_LABELS[slot], "Reference": before_id,
                     "Comparison": after_id, "变化": "CHANGED" if before != after else "UNCHANGED"})
    st.dataframe(rows, hide_index=True, width="stretch")

    detail_tabs = st.tabs(["研究声明", "能力边界", "可复现来源", "全部 Profile"])
    with detail_tabs[0]:
        for label, profile in (("REFERENCE", baseline), ("COMPARISON", candidate)):
            research = profile["research"]
            with st.container(key=f'eng_research_{label.lower()}'):
                definition_grid([
                    (label, profile["profile_id"]),
                    ("STATUS", RESEARCH_STATUS_LABELS.get(research["status"], research["status"])),
                    ("SOURCE", research["source_type"]),
                    ("PAPER ID", research.get("paper_id") or "NATIVE"),
                ])
                st.write(research.get("hypothesis") or "基线配置不主张实验增益。")
    with detail_tabs[1]:
        st.markdown("**Reference capabilities**")
        st.markdown(_capability_chips(baseline["capabilities"]), unsafe_allow_html=True)
        st.markdown("**Comparison capabilities**")
        st.markdown(_capability_chips(candidate["capabilities"]), unsafe_allow_html=True)
    with detail_tabs[2]:
        for profile in (baseline, candidate):
            with st.expander(f'{profile["profile_id"]} · execution {profile["execution_sha256"][:12]}'):
                st.json({
                    "profile_sha256": profile["fingerprint"],
                    "execution_sha256": profile["execution_sha256"],
                    "source_revision": profile["source_revision"],
                    "mechanisms": profile["resolved_mechanisms"],
                })
    with detail_tabs[3]:
        for profile in profiles:
            with st.expander(f'{profile["profile_id"]} · {profile["display_name"]}'):
                st.write(profile["description"])
                mechanism_rail(profile)


def datasets_page() -> None:
    datasets = dataset_catalog()
    engineering_header(
        "DATASET COVERAGE / 03",
        "Dataset Coverage",
        "从任务类别、数据分组和断言深度观察评测底座，避免用少量顺利案例代替能力证据。",
    )
    telemetry_strip([
        ("DATASETS", str(len(datasets)), "固定 / 真实模型 / 自主专项", "blue"),
        ("TOTAL CASES", str(sum(item["case_count"] for item in datasets)), "来自版本化 JSON", "teal"),
        ("TOTAL TURNS", str(sum(item["turn_count"] for item in datasets)), "含多轮任务", "ink"),
        ("TEST SPLIT", str(sum(item["splits"].get("test", 0) for item in datasets)), "未参与调参", "violet"),
    ])
    selected_id = st.selectbox(
        "任务集", [item["dataset_id"] for item in datasets],
        format_func=lambda value: next(f'{item["display_name"]} · {item["case_count"]} cases' for item in datasets if item["dataset_id"] == value),
    )
    dataset = next(item for item in datasets if item["dataset_id"] == selected_id)
    section_heading("01", dataset["display_name"], dataset["path"])
    definition_grid([
        ("DATASET ID", dataset["dataset_id"]),
        ("SUITE", dataset["suite"]),
        ("EXECUTION", dataset["execution_source"]),
        ("SPLITS", " · ".join(f"{key}:{value}" for key, value in dataset["splits"].items())),
    ])
    st.caption(dataset["description"])

    left, right = st.columns([1.05, 1.7], gap="large")
    with left:
        section_heading("02", "类别覆盖", "CASE DISTRIBUTION")
        maximum = max(dataset["categories"].values(), default=1)
        bars = []
        for category, count in sorted(dataset["categories"].items(), key=lambda item: (-item[1], item[0])):
            width = max(8, round(count / maximum * 100))
            bars.append(
                f'<div class="eng-coverage-row"><div><span>{esc(CATEGORY_LABELS.get(category, category))}</span>'
                f'<b>{count:02d}</b></div><div class="eng-coverage-track"><i style="width:{width}%"></i></div>'
                f'<code>{esc(category)}</code></div>'
            )
        st.markdown(f'<div class="eng-coverage">{"".join(bars)}</div>', unsafe_allow_html=True)
    with right:
        section_heading("03", "任务清单", "SELECT A CASE TO INSPECT")
        rows = [{
            "case": case["id"],
            "category": case.get("category", "—"),
            "split": case.get("split", "fixed"),
            "turns": len(case.get("turns", [])),
            "description": case.get("description", ""),
        } for case in dataset["cases"]]
        st.dataframe(rows, hide_index=True, width="stretch", height=380)

    case_ids = [case["id"] for case in dataset["cases"]]
    chosen = st.selectbox(
        "Case Inspector", case_ids,
        format_func=lambda value: next(f'{case["id"]} · {case.get("description", "")}' for case in dataset["cases"] if case["id"] == value),
    )
    case = next(item for item in dataset["cases"] if item["id"] == chosen)
    with st.container(key="eng_panel_case_inspector"):
        definition_grid([
            ("CASE ID", case["id"]),
            ("CATEGORY", case.get("category", "—")),
            ("SPLIT", case.get("split", "fixed")),
            ("LANGUAGE", case.get("language", "—")),
        ])
        for index, turn in enumerate(case.get("turns", []), start=1):
            st.markdown(f'**Turn {index:02d}** · {turn.get("message", "")}' )
            expectation = turn.get("expect", {})
            st.caption(
                "状态：" + ", ".join(expectation.get("statuses", [])) +
                " · 工具：" + (", ".join(tool.get("name", "") for tool in expectation.get("tools", [])) or "无强制工具")
            )
        with st.expander("完整机器断言"):
            st.json(case)


def experiments_page() -> None:
    experiments = experiment_catalog()
    profiles = profile_catalog()
    engineering_header(
        "EXPERIMENTS / 04",
        "Experiment Contracts",
        "先冻结基线、变量和指标，再执行候选。没有可比报告时，界面明确保持“不能下结论”。",
        marker="RESEARCH MODE",
    )
    if not experiments:
        st.info("尚无结构化实验契约。")
        return
    experiment_id = st.selectbox("实验", [item["experiment_id"] for item in experiments])
    experiment = next(item for item in experiments if item["experiment_id"] == experiment_id)
    baseline = _profile(profiles, experiment["baseline_profile"])
    candidate = _profile(profiles, experiment["candidate_profile"])
    report_ready = experiment["status"] == "COMPLETED"
    conclusion = "REPORT READY" if report_ready else "NO COMPARABLE REPORT"
    telemetry_strip([
        ("CONTRACT", "VALID", experiment["path"], "teal"),
        ("VARIABLES", str(len(experiment["changes"])), "契约声明的变化", "violet"),
        ("TEST REPEATS", str(experiment["repeats"]), f'{experiment["evaluation_split"]} split', "blue"),
        ("CONCLUSION", conclusion, "可以下结论" if report_ready else "禁止声称增益", "teal" if report_ready else "amber"),
    ])
    st.markdown(
        '<div class="eng-experiment-pipeline">'
        '<div class="done"><span>01</span><b>FREEZE</b><p>冻结 Profile 与数据集</p></div>'
        '<i></i><div class="done"><span>02</span><b>DECLARE</b><p>声明唯一变化与指标</p></div>'
        '<i></i><div class="current"><span>03</span><b>RUN</b><p>dev 调试 / test 结论</p></div>'
        '<i></i><div><span>04</span><b>GATE</b><p>正确率、安全、成本</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    section_heading("01", "实验变量", "BASELINE → CANDIDATE")
    left, right = st.columns(2, gap="large")
    with left, st.container(key="eng_panel_experiment_baseline"):
        st.markdown(f'<div class="eng-compare-label">BASELINE</div><h3>{esc(baseline["display_name"])}</h3>', unsafe_allow_html=True)
        mechanism_rail(baseline)
    with right, st.container(key="eng_panel_experiment_candidate"):
        st.markdown(f'<div class="eng-compare-label candidate">CANDIDATE</div><h3>{esc(candidate["display_name"])}</h3>', unsafe_allow_html=True)
        mechanism_rail(candidate, baseline)
    st.markdown('<div class="eng-hypothesis"><span>HYPOTHESIS</span>'
                f'<p>{esc(candidate["research"].get("hypothesis") or "未声明")}</p></div>', unsafe_allow_html=True)

    detail, metrics = st.columns([1.05, 1.4], gap="large")
    with detail:
        section_heading("02", "变更白名单")
        for change in experiment["changes"]:
            with st.container(key=f'eng_change_{change["kind"]}_{change.get("slot") or "runtime"}'):
                definition_grid([
                    ("KIND", change["kind"]),
                    ("SLOT", change.get("slot") or "—"),
                    ("IMPLEMENTATION", change.get("implementation_id") or "—"),
                ])
                st.code("\n".join(change["paths"]), language=None)
    with metrics:
        section_heading("03", "Gate 指标")
        rows = []
        for kind, rules in (("PRIMARY", experiment["primary_metrics"]), ("GUARD", experiment["guard_metrics"])):
            for rule in rules:
                rows.append({
                    "kind": kind,
                    "metric": rule["name"],
                    "objective": rule["objective"],
                    "max_regression": rule["max_regression"],
                    "min_improvement": rule["min_improvement"],
                })
        st.dataframe(rows, hide_index=True, width="stretch")
    if report_ready:
        section_heading("04", "实验报告", experiment["report_path"])
        st.json((experiment["report"] or {}).get("metrics", {}))
    else:
        st.warning("当前仓库没有同时绑定该实验契约的冻结基线与候选报告，因此不能声称候选机制更好。")
        st.caption("下一步应分别在 test split 上运行 baseline 与 candidate，保持模型配置、依赖锁、重复次数和业务环境一致。")
