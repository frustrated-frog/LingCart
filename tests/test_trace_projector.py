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
