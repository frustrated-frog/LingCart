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
            prepared = [record for record in self.records if record.get("record_type") == "REPORT_PREPARED"]
            if len(prepared) > 1:
                raise JournalError("REPORT_PREPARED 只能出现一次")
            if prepared:
                prepared_index = self.records.index(prepared[0])
                trailing = [record["record_type"] for record in self.records[prepared_index + 1:]]
                if trailing not in ([], ["RUN_FINALIZED"]):
                    raise JournalError("REPORT_PREPARED 之后只能提交 RUN_FINALIZED")
                if finalized and prepared_index != len(self.records) - 2:
                    raise JournalError("RUN_FINALIZED 必须紧跟 REPORT_PREPARED")
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

    @property
    def report_prepared(self) -> bool:
        return any(record["record_type"] == "REPORT_PREPARED" for record in self.records)

    @property
    def started_at(self) -> str | None:
        if not self.records:
            return None
        return (self.records[0].get("payload") or {}).get("started_at")

    @property
    def prepared_report(self) -> dict | None:
        record = next(
            (item for item in self.records if item["record_type"] == "REPORT_PREPARED"),
            None,
        )
        return deepcopy_json(record["payload"]) if record else None

    def append(self, record_type: str, payload: dict, *, sample_id: str | None = None) -> dict:
        if self._file is None:
            raise JournalError("Journal 尚未打开")
        if self.finalized:
            raise JournalError("已完成的 Journal 不可追加")
        if self.records and record_type == "RUN_STARTED":
            raise JournalError("RUN_STARTED 只能写入一次")
        if not self.records and record_type != "RUN_STARTED":
            raise JournalError("RUN_STARTED 必须先于其它记录")
        if self.report_prepared and record_type != "RUN_FINALIZED":
            raise JournalError("报告准备完成后只能提交 RUN_FINALIZED")
        if record_type == "RUN_FINALIZED" and not self.report_prepared:
            raise JournalError("RUN_FINALIZED 必须在 REPORT_PREPARED 和报告发布之后提交")
        if record_type == "REPORT_PREPARED" and self.report_prepared:
            raise JournalError("REPORT_PREPARED 只能写入一次")
        if record_type == "REPORT_PREPARED":
            expected_ids = {
                item.get("sample_id")
                for item in ((self.records[0].get("payload") or {}).get("sample_plan") or [])
            }
            if expected_ids != self.terminal_sample_ids:
                raise JournalError("完整样本网格提交前不能准备最终报告")
            output = Path(payload.get("output_path", ""))
            prepared_path = Path(payload.get("prepared_path", ""))
            digest = payload.get("report_sha256")
            if (
                not output.is_absolute()
                or not prepared_path.is_absolute()
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise JournalError("REPORT_PREPARED 缺少规范的报告路径或哈希")
            _require_report_copy(output, prepared_path, digest)
        if record_type == "RUN_FINALIZED":
            prepared = self.prepared_report or {}
            output = Path(prepared.get("output_path", ""))
            digest = prepared.get("report_sha256")
            if (
                payload.get("output_path") != str(output)
                or payload.get("report_sha256") != digest
                or not output.is_file()
                or _file_sha256(output) != digest
            ):
                raise JournalError("最终报告尚未可靠发布，拒绝 RUN_FINALIZED")
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
            "started_at": _utc_now(),
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

    def prepare_report(self, output_path: Path, report: dict, *, finalization: dict) -> dict:
        """Durably stage an immutable report before recording its prepared state.

        The deterministic staging file is fsynced before ``REPORT_PREPARED`` is
        appended.  A crash on either side of that append is therefore
        recoverable without re-running paid samples.
        """

        output = Path(output_path).resolve()
        prepared_path = _prepared_path(output, self.evaluation_id)
        encoded = _json_bytes(report)
        report_sha256 = hashlib.sha256(encoded).hexdigest()
        expected = {
            "output_path": str(output),
            "prepared_path": str(prepared_path),
            "report_sha256": report_sha256,
            "report_size_bytes": len(encoded),
            "finalization": deepcopy_json(finalization),
        }
        existing = self.prepared_report
        if existing is not None:
            if existing != expected:
                raise JournalError("续跑生成的最终报告与已准备报告不一致")
            _require_report_copy(output, prepared_path, report_sha256)
            return existing

        if output.exists():
            raise FileExistsError(f"最终报告已存在但 Journal 尚未准备发布：{output}")
        if prepared_path.exists():
            if _file_sha256(prepared_path) != report_sha256:
                prepared_path.unlink()
                _sync_directory(prepared_path.parent)
                _write_exclusive_bytes(prepared_path, encoded)
        else:
            _write_exclusive_bytes(prepared_path, encoded)
        self.append("REPORT_PREPARED", expected)
        return expected

    def publish_prepared_report(self, output_path: Path) -> str:
        """Atomically publish the staged report without overwriting evidence."""

        prepared = self.prepared_report
        if prepared is None:
            raise JournalError("报告尚未进入 REPORT_PREPARED")
        output = Path(output_path).resolve()
        if prepared.get("output_path") != str(output):
            raise JournalError("续跑输出路径与 REPORT_PREPARED 不一致")
        prepared_path = Path(prepared["prepared_path"])
        report_sha256 = prepared["report_sha256"]
        if output.exists():
            if not output.is_file() or _file_sha256(output) != report_sha256:
                raise JournalError("已存在的最终报告与 Journal 证据不一致")
            return report_sha256
        if not prepared_path.is_file() or _file_sha256(prepared_path) != report_sha256:
            raise JournalError("报告准备文件缺失或哈希不一致，拒绝伪造恢复")
        output.parent.mkdir(parents=True, exist_ok=True)
        os.link(prepared_path, output)
        _sync_directory(output.parent)
        return report_sha256

    def finalize_report(self, output_path: Path) -> dict:
        """Finalize only after the published bytes are durably observable."""

        if self.finalized:
            self.load_published_report(output_path)
            return deepcopy_json(self.records[-1]["payload"])
        prepared = self.prepared_report
        if prepared is None:
            raise JournalError("没有可完成的 REPORT_PREPARED")
        report_sha256 = self.publish_prepared_report(output_path)
        payload = {
            **deepcopy_json(prepared.get("finalization") or {}),
            "report_sha256": report_sha256,
            "output_path": prepared["output_path"],
        }
        record = self.append("RUN_FINALIZED", payload)
        prepared_path = Path(prepared["prepared_path"])
        if prepared_path.exists():
            prepared_path.unlink()
            _sync_directory(prepared_path.parent)
        return deepcopy_json(record["payload"])

    def load_published_report(self, output_path: Path) -> dict:
        """Return a finalized report only when bytes and Journal agree."""

        if not self.finalized:
            raise JournalError("Journal 尚未完成")
        payload = self.records[-1]["payload"]
        output = Path(output_path).resolve()
        if payload.get("output_path") != str(output):
            raise JournalError("最终报告路径与 Journal 不一致")
        if not output.is_file() or _file_sha256(output) != payload.get("report_sha256"):
            raise JournalError("最终报告缺失或哈希与 Journal 不一致")
        try:
            return json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JournalError("最终报告不是有效 JSON") from exc


def deepcopy_json(value: object):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _prepared_path(output: Path, evaluation_id: str | None) -> Path:
    if not evaluation_id:
        raise JournalError("Journal 缺少 evaluation_id")
    return output.with_name(f".{output.name}.{evaluation_id}.prepared")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive_bytes(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(path.parent)
    except BaseException:
        # A partial, exclusively named staging file is evidence of an
        # interrupted prepare.  The next resume verifies its hash and fails
        # closed instead of silently replacing it.
        raise


def _require_report_copy(output: Path, prepared_path: Path, report_sha256: str) -> None:
    candidates = [path for path in (output, prepared_path) if path.is_file()]
    if not candidates or any(_file_sha256(path) != report_sha256 for path in candidates):
        raise JournalError("REPORT_PREPARED 缺少可验证的报告副本")


def write_exclusive_json(path: Path, payload: dict) -> str:
    """Durably create a final report without exists/write TOCTOU."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _json_bytes(payload)
    _write_exclusive_bytes(path, encoded)
    return hashlib.sha256(encoded).hexdigest()
