"""Crash-safe append-only journal for paid evaluation executions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import BinaryIO

from eval.semantic_contracts import canonical_json, fingerprint


JOURNAL_SCHEMA_VERSION = 1


class JournalError(ValueError):
    pass


class EvaluationJournal:
    """Single-writer JSONL hash chain.

    A complete line is a committed record and is never repaired.  Only an
    unterminated final byte tail may be discarded after the file lock is held.
    """

    def __init__(self, path: Path, *, resume: bool):
        self.path = Path(path)
        self.resume = resume
        self._file: BinaryIO | None = None
        self.records: list[dict] = []
        self._discarded_tail: bytes | None = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        self._file = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._file.close()
            self._file = None
            raise JournalError("评测 Journal 正被另一个进程占用") from exc
        self._load()
        if self.records and not self.resume:
            self.close()
            raise JournalError("Journal 已存在；必须显式使用 resume，不能覆盖付费样本")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self) -> None:
        if self._file is None:
            return
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        finally:
            self._file.close()
            self._file = None

    def _load(self) -> None:
        assert self._file is not None
        self._file.seek(0)
        content = self._file.read()
        if content and not content.endswith(b"\n"):
            boundary = content.rfind(b"\n") + 1
            self._discarded_tail = content[boundary:]
            self._file.seek(boundary)
            self._file.truncate()
            self._sync()
            content = content[:boundary]
        previous = None
        for index, raw in enumerate(content.splitlines(), start=1):
            try:
                record = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise JournalError(f"Journal 第 {index} 行不是有效 JSON，拒绝跳过已提交记录") from exc
            if not isinstance(record, dict):
                raise JournalError(f"Journal 第 {index} 行必须是对象")
            supplied_hash = record.get("record_sha256")
            unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
            if supplied_hash != fingerprint(unsigned):
                raise JournalError(f"Journal 第 {index} 行哈希校验失败")
            if record.get("journal_schema_version") != JOURNAL_SCHEMA_VERSION:
                raise JournalError(f"Journal 第 {index} 行 schema 不受支持")
            if record.get("sequence") != index:
                raise JournalError(f"Journal 第 {index} 行 sequence 不连续")
            if record.get("prev_record_sha256") != previous:
                raise JournalError(f"Journal 第 {index} 行哈希链断裂")
            previous = supplied_hash
            self.records.append(record)
        if self.records:
            evaluation_ids = {record.get("evaluation_id") for record in self.records}
            if len(evaluation_ids) != 1 or None in evaluation_ids:
                raise JournalError("Journal 包含多个或缺失 evaluation_id")
            if self.records[0].get("record_type") != "RUN_STARTED":
                raise JournalError("Journal 首条记录必须是 RUN_STARTED")
            finalized = [record for record in self.records if record.get("record_type") == "RUN_FINALIZED"]
            if len(finalized) > 1 or (finalized and self.records[-1] is not finalized[0]):
                raise JournalError("RUN_FINALIZED 必须唯一且位于 Journal 末尾")
            self._validate_sample_lifecycle()
        self._file.seek(0, os.SEEK_END)

    def _validate_sample_lifecycle(self) -> None:
        started, terminal = set(), set()
        for record in self.records:
            sample_id = record.get("sample_id")
            if record["record_type"] == "SAMPLE_STARTED":
                if not sample_id or sample_id in started:
                    raise JournalError("同一样本只能出现一次 SAMPLE_STARTED")
                started.add(sample_id)
            elif record["record_type"] == "SAMPLE_TERMINAL":
                if not sample_id or sample_id not in started or sample_id in terminal:
                    raise JournalError("SAMPLE_TERMINAL 缺少唯一的先行 SAMPLE_STARTED")
                terminal.add(sample_id)

    def _sync(self) -> None:
        assert self._file is not None
        self._file.flush()
        os.fsync(self._file.fileno())

    @property
    def evaluation_id(self) -> str | None:
        return self.records[0]["evaluation_id"] if self.records else None

    @property
    def finalized(self) -> bool:
        return bool(self.records and self.records[-1]["record_type"] == "RUN_FINALIZED")

    def append(self, record_type: str, payload: dict, *, sample_id: str | None = None) -> dict:
        if self._file is None:
            raise JournalError("Journal 尚未打开")
        if self.finalized:
            raise JournalError("已完成的 Journal 不可追加")
        if self.records and record_type == "RUN_STARTED":
            raise JournalError("RUN_STARTED 只能写入一次")
        if not self.records and record_type != "RUN_STARTED":
            raise JournalError("RUN_STARTED 必须先于其它记录")
        evaluation_id = payload.get("evaluation_id") if record_type == "RUN_STARTED" else self.evaluation_id
        if not isinstance(evaluation_id, str) or not evaluation_id:
            raise JournalError("Journal record 缺少 evaluation_id")
        if record_type == "SAMPLE_TERMINAL" and sample_id in self.terminal_sample_ids:
            raise JournalError("同一样本不能覆盖或追加第二个终态")
        unsigned = {
            "journal_schema_version": JOURNAL_SCHEMA_VERSION,
            "sequence": len(self.records) + 1,
            "record_type": record_type,
            "evaluation_id": evaluation_id,
            "sample_id": sample_id,
            "prev_record_sha256": self.records[-1]["record_sha256"] if self.records else None,
            "payload": payload,
        }
        record = {**unsigned, "record_sha256": fingerprint(unsigned)}
        encoded = (canonical_json(record) + "\n").encode("utf-8")
        self._file.write(encoded)
        self._sync()
        self.records.append(record)
        return record

    def start(self, *, evaluation_id: str, run_identity_sha256: str, manifest: dict,
              sample_plan: list[dict]) -> dict:
        if self.records:
            self.require_identity(run_identity_sha256)
            return self.records[0]
        record = self.append("RUN_STARTED", {
            "evaluation_id": evaluation_id,
            "run_identity_sha256": run_identity_sha256,
            "manifest": manifest,
            "sample_plan": sample_plan,
        })
        if self._discarded_tail:
            self.append("JOURNAL_RECOVERED", {
                "discarded_bytes": len(self._discarded_tail),
                "discarded_sha256": hashlib.sha256(self._discarded_tail).hexdigest(),
            })
            self._discarded_tail = None
        return record

    def require_identity(self, run_identity_sha256: str) -> None:
        saved = (self.records[0].get("payload") or {}).get("run_identity_sha256")
        if saved != run_identity_sha256:
            raise JournalError("续跑身份已变化：数据集、阶段、Profile、模型配置、预算或源码不一致")
        if self._discarded_tail:
            self.append("JOURNAL_RECOVERED", {
                "discarded_bytes": len(self._discarded_tail),
                "discarded_sha256": hashlib.sha256(self._discarded_tail).hexdigest(),
            })
            self._discarded_tail = None

    @property
    def terminal_sample_ids(self) -> set[str]:
        return {
            record["sample_id"] for record in self.records
            if record["record_type"] == "SAMPLE_TERMINAL"
        }

    def terminal_samples(self) -> dict[str, dict]:
        return {
            record["sample_id"]: deepcopy_json(record["payload"]["sample"])
            for record in self.records
            if record["record_type"] == "SAMPLE_TERMINAL"
        }

    def orphaned_samples(self) -> dict[str, dict]:
        terminals = self.terminal_sample_ids
        return {
            record["sample_id"]: deepcopy_json(record["payload"])
            for record in self.records
            if record["record_type"] == "SAMPLE_STARTED" and record["sample_id"] not in terminals
        }


def deepcopy_json(value: object):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def write_exclusive_json(path: Path, payload: dict) -> str:
    """Durably create a final report without exists/write TOCTOU."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        # The file may contain a partial report, but it is never overwritten.
        # Its absence from RUN_FINALIZED keeps it from becoming valid evidence.
        raise
    return hashlib.sha256(encoded).hexdigest()
