from src.trace_projector import project_trace


def test_trace_projector_builds_one_stable_timeline():
    events = [
        {"run_id": "run-1", "sequence": 1, "timestamp": "t1", "type": "model_call", "status": "SUCCEEDED"},
        {"run_id": "run-1", "sequence": 2, "timestamp": "t2", "type": "tool_call", "tool": "lookup"},
        {"run_id": "run-1", "sequence": 3, "timestamp": "t3", "type": "tool_result", "tool": "lookup", "status": "SUCCEEDED"},
        {"run_id": "run-1", "sequence": 4, "timestamp": "t4", "type": "terminal", "status": "SUCCEEDED", "reason": "COMPLETED"},
    ]

    projected = project_trace(events)

    assert projected["run_id"] == "run-1"
    assert projected["integrity"]["sequential"] is True
    assert projected["integrity"]["single_run"] is True
    assert projected["integrity"]["single_terminal"] is True
    assert projected["integrity"]["tool_pairs"] is True
    assert projected["integrity"]["protocol_pairs"] is None
    assert projected["integrity"]["complete"] is None
    assert projected["integrity"]["event_count"] == 4
    assert projected["summary"]["tool_calls"] == 1
    assert projected["summary"]["tool_results"] == 1
    assert projected["summary"]["terminal_reason"] == "COMPLETED"
    assert [item["phase"] for item in projected["timeline"]] == ["MODEL", "TOOL", "TOOL", "TERMINAL"]


def test_trace_projector_reports_broken_integrity_without_hiding_events():
    projected = project_trace([
        {"run_id": "one", "sequence": 2, "type": "error"},
        {"run_id": "two", "sequence": 4, "type": "terminal"},
    ])

    assert projected["run_id"] is None
    assert projected["integrity"]["sequential"] is False
    assert projected["integrity"]["single_run"] is False
    assert len(projected["timeline"]) == 2


def test_trace_projector_exposes_harness_selection_as_a_first_class_phase():
    projected = project_trace([
        {"run_id": "one", "sequence": 1, "type": "harness_profile_selected", "profile_id": "baseline-v1"},
        {"run_id": "one", "sequence": 2, "type": "terminal", "status": "SUCCEEDED", "reason": "COMPLETED"},
    ])
    assert projected["timeline"][0]["phase"] == "HARNESS"
    assert projected["timeline"][0]["label"] == "已选择 Harness Profile"


def test_trace_projector_surfaces_liveplan_monitor_advisor_and_intervention_evidence():
    projected = project_trace([
        {"run_id": "one", "sequence": 1, "type": "trajectory_monitor_evaluated"},
        {"run_id": "one", "sequence": 2, "type": "trajectory_drift_detected",
         "drift_id": "drift", "signal_codes": ["ACTION_OSCILLATION"]},
        {"run_id": "one", "sequence": 3, "type": "model_call", "purpose": "liveplan_advisor"},
        {"run_id": "one", "sequence": 4, "type": "corrective_advice_generated"},
        {"run_id": "one", "sequence": 5, "type": "intervention_applied",
         "intervention_id": "intervention", "outcome": "ACTIVATED"},
        {"run_id": "one", "sequence": 6, "type": "intervention_applied",
         "intervention_id": "intervention", "outcome": "ACTION_REJECTED"},
        {"run_id": "one", "sequence": 7, "type": "terminal", "status": "PARTIAL", "reason": "MODEL_STOPPED"},
    ])

    assert [item["phase"] for item in projected["timeline"][:6]] == [
        "MONITOR", "MONITOR", "MODEL", "ADVISOR", "INTERVENTION", "INTERVENTION",
    ]
    assert projected["summary"]["drift_detections"] == 1
    assert projected["summary"]["advisor_calls"] == 1
    assert projected["summary"]["interventions"] == 1
    assert projected["summary"]["action_rejections"] == 1


def test_trace_projector_surfaces_policy_aware_case_execution_phases():
    projected = project_trace([
        {"run_id": "case-run", "sequence": 1, "type": "case_state_projected"},
        {"run_id": "case-run", "sequence": 2, "type": "case_obligation_graph_solved"},
        {"run_id": "case-run", "sequence": 3, "type": "case_obligation_verified"},
        {"run_id": "case-run", "sequence": 4, "type": "case_autonomous_action_selected"},
        {"run_id": "case-run", "sequence": 5, "type": "case_autonomous_action_completed"},
        {"run_id": "case-run", "sequence": 6, "type": "case_execution_paused"},
        {"run_id": "case-run", "sequence": 7, "type": "terminal", "status": "WAITING_APPROVAL"},
    ])

    assert [item["phase"] for item in projected["timeline"][:6]] == [
        "CONTEXT", "PLAN", "VERIFY", "PLAN", "TOOL", "GUARD",
    ]
    assert [item["label"] for item in projected["timeline"][:6]] == [
        "已投影政策约束下的案件状态",
        "已构建并求解政策义务图",
        "已核验下一项政策义务",
        "已选择受限自主动作",
        "受限自主动作已产生版本进展",
        "政策感知执行已在边界处暂停",
    ]
