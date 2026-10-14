import pytest

from src.errors import AgentError
from src.runtime_state import advance_runtime_snapshot, new_runtime_snapshot, restore_runtime_snapshot


def test_runtime_snapshot_is_versioned_checksummed_and_advances():
    snapshot = new_runtime_snapshot("run-1", conversation={"topic": "order"}, history=[])

    assert snapshot["schema_version"] == 2
    assert snapshot["state_version"] == 1
    assert snapshot["run_id"] == "run-1"
    assert len(snapshot["checksum"]) == 64

    advanced = advance_runtime_snapshot(snapshot, "run-1", current_node="MODEL_CALL")

    assert advanced["state_version"] == 2
    assert advanced["current_node"] == "MODEL_CALL"
    assert restore_runtime_snapshot(advanced, "run-1") == advanced


def test_runtime_snapshot_rejects_tampering_and_unknown_schema():
    snapshot = new_runtime_snapshot("run-2", conversation={}, history=[])
    snapshot["conversation"] = {"tampered": True}

    with pytest.raises(AgentError) as corrupted:
        restore_runtime_snapshot(snapshot, "run-2")
    assert corrupted.value.code == "CHECKPOINT_CORRUPT"

    unsupported = new_runtime_snapshot("run-2", conversation={}, history=[])
    unsupported["schema_version"] = 99
    with pytest.raises(AgentError) as schema_error:
        restore_runtime_snapshot(unsupported, "run-2")
    assert schema_error.value.code == "CHECKPOINT_SCHEMA_UNSUPPORTED"


def test_terminal_runtime_snapshot_cannot_return_to_execution():
    snapshot = new_runtime_snapshot("run-3", conversation={}, history=[])
    terminal = advance_runtime_snapshot(snapshot, "run-3", current_node="TERMINAL", status="SUCCEEDED")

    with pytest.raises(AgentError) as transition_error:
        advance_runtime_snapshot(terminal, "run-3", current_node="MODEL_CALL", status="RUNNING")
    assert transition_error.value.code == "INVALID_STATE_TRANSITION"
