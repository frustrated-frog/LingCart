"""Versioned, checksummed runtime snapshot protocol."""
from __future__ import annotations

import hashlib
import json

from src.errors import AgentError


CURRENT_RUNTIME_SCHEMA_VERSION = 2


def _checksum(snapshot: dict) -> str:
    payload = {key: value for key, value in snapshot.items() if key != "checksum"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sealed(snapshot: dict) -> dict:
    result = dict(snapshot)
    result["checksum"] = _checksum(result)
    return result


def new_runtime_snapshot(run_id: str, *, conversation: dict, history: list) -> dict:
    return _sealed({
        "schema_version": CURRENT_RUNTIME_SCHEMA_VERSION,
        "state_version": 1,
        "run_id": run_id,
        "status": "RUNNING",
        "current_node": "INTAKE",
        "phase": "INTAKE",
        "terminal_reason": None,
        "conversation": conversation,
        "history": history,
        "budget": {},
        "models": {},
        "sequence": 0,
    })


def restore_runtime_snapshot(payload: dict | None, run_id: str) -> dict:
    """Validate the stored state and migrate pre-versioned V1 snapshots."""

    if not isinstance(payload, dict):
        raise AgentError("CHECKPOINT_CORRUPT", "任务检查点不是有效对象", retryable=False)
    snapshot = dict(payload)
    schema_version = snapshot.get("schema_version")
    if schema_version is None:
        # V1 used phase/sequence but had no explicit protocol version.
        snapshot.update(
            schema_version=CURRENT_RUNTIME_SCHEMA_VERSION,
            state_version=max(1, int(snapshot.get("sequence", 0)) + 1),
            run_id=run_id,
            status="COMPLETED" if snapshot.get("phase") == "TERMINAL" else "RUNNING",
            current_node=snapshot.get("phase", "INTAKE"),
            terminal_reason=snapshot.get("terminal_reason"),
        )
        return _sealed(snapshot)
    if schema_version != CURRENT_RUNTIME_SCHEMA_VERSION:
        raise AgentError(
            "CHECKPOINT_SCHEMA_UNSUPPORTED",
            f"任务检查点版本 {schema_version} 与当前运行时不兼容",
            retryable=False,
        )
    if snapshot.get("run_id") != run_id:
        raise AgentError("CHECKPOINT_RUN_MISMATCH", "任务检查点与运行编号不匹配", retryable=False)
    state_version = snapshot.get("state_version")
    if isinstance(state_version, bool) or not isinstance(state_version, int) or state_version < 1:
        raise AgentError("CHECKPOINT_CORRUPT", "任务检查点状态版本无效", retryable=False)
    checksum = snapshot.get("checksum")
    if not checksum or checksum != _checksum(snapshot):
        raise AgentError("CHECKPOINT_CORRUPT", "任务检查点完整性校验失败", retryable=False)
    return snapshot


def seal_runtime_snapshot(payload: dict, run_id: str) -> dict:
    """Seal an in-memory working snapshot after the runtime has mutated it."""

    if not isinstance(payload, dict):
        raise AgentError("CHECKPOINT_CORRUPT", "任务检查点不是有效对象", retryable=False)
    snapshot = dict(payload)
    if snapshot.get("schema_version") != CURRENT_RUNTIME_SCHEMA_VERSION:
        raise AgentError("CHECKPOINT_SCHEMA_UNSUPPORTED", "任务检查点版本与当前运行时不兼容", retryable=False)
    if snapshot.get("run_id") != run_id:
        raise AgentError("CHECKPOINT_RUN_MISMATCH", "任务检查点与运行编号不匹配", retryable=False)
    state_version = snapshot.get("state_version")
    if isinstance(state_version, bool) or not isinstance(state_version, int) or state_version < 1:
        raise AgentError("CHECKPOINT_CORRUPT", "任务检查点状态版本无效", retryable=False)
    return _sealed(snapshot)


def advance_runtime_snapshot(snapshot: dict, run_id: str, **updates) -> dict:
    current = restore_runtime_snapshot(snapshot, run_id)
    if current.get("current_node") == "TERMINAL" and updates.get("current_node", "TERMINAL") != "TERMINAL":
        raise AgentError("INVALID_STATE_TRANSITION", "已结束任务不能回到执行状态", retryable=False)
    result = {**current, **updates}
    result["schema_version"] = CURRENT_RUNTIME_SCHEMA_VERSION
    result["state_version"] = current["state_version"] + 1
    result["run_id"] = run_id
    result["phase"] = result.get("current_node", result.get("phase", "INTAKE"))
    return _sealed(result)
