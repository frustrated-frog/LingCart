import json
from pathlib import Path

import pytest

from eval.eval_journal import EvaluationJournal, JournalError, write_exclusive_json


def _start(journal, identity="a" * 64):
    journal.start(
        evaluation_id="evaluation-1",
        run_identity_sha256=identity,
        manifest={"stage_id": "single-smoke"},
        sample_plan=[{"sample_id": "sample-1"}],
    )


def test_journal_fsyncs_a_terminal_sample_and_resume_never_duplicates_it(tmp_path):
    path = tmp_path / "semantic.journal.jsonl"
    with EvaluationJournal(path, resume=False) as journal:
        _start(journal)
        journal.append("SAMPLE_STARTED", {"case_id": "clock"}, sample_id="sample-1")
        journal.append("SAMPLE_TERMINAL", {
            "outcome": "COMPLETED", "sample": {"case_id": "clock", "passed": True}
        }, sample_id="sample-1")

    with EvaluationJournal(path, resume=True) as journal:
        journal.require_identity("a" * 64)
        assert journal.terminal_sample_ids == {"sample-1"}
        assert journal.terminal_samples()["sample-1"]["passed"] is True
        with pytest.raises(JournalError, match="第二个终态"):
            journal.append("SAMPLE_TERMINAL", {
                "outcome": "COMPLETED", "sample": {"case_id": "clock", "passed": True}
            }, sample_id="sample-1")


def test_resume_rejects_any_run_identity_drift_before_new_records(tmp_path):
    path = tmp_path / "semantic.journal.jsonl"
    with EvaluationJournal(path, resume=False) as journal:
        _start(journal)

    with EvaluationJournal(path, resume=True) as journal:
        before = path.read_bytes()
        with pytest.raises(JournalError, match="续跑身份已变化"):
            journal.require_identity("b" * 64)
        assert path.read_bytes() == before


def test_only_unterminated_tail_is_recovered_and_audited(tmp_path):
    path = tmp_path / "semantic.journal.jsonl"
    with EvaluationJournal(path, resume=False) as journal:
        _start(journal)
    with path.open("ab") as stream:
        stream.write(b'{"partial":')

    with EvaluationJournal(path, resume=True) as journal:
        journal.require_identity("a" * 64)
        assert journal.records[-1]["record_type"] == "JOURNAL_RECOVERED"
        assert journal.records[-1]["payload"]["discarded_bytes"] == len(b'{"partial":')

    assert path.read_bytes().endswith(b"\n")


def test_complete_corrupt_record_is_never_skipped_or_repaired(tmp_path):
    path = tmp_path / "semantic.journal.jsonl"
    with EvaluationJournal(path, resume=False) as journal:
        _start(journal)
    with path.open("ab") as stream:
        stream.write(b'{"complete":"but invalid"}\n')

    with pytest.raises(JournalError, match="哈希校验失败"):
        with EvaluationJournal(path, resume=True):
            pass


def test_orphaned_started_sample_is_visible_for_fail_closed_recovery(tmp_path):
    path = tmp_path / "semantic.journal.jsonl"
    with EvaluationJournal(path, resume=False) as journal:
        _start(journal)
        journal.append("SAMPLE_STARTED", {"case_id": "clock", "mode": "auto"}, sample_id="sample-1")

    with EvaluationJournal(path, resume=True) as journal:
        assert journal.orphaned_samples() == {
            "sample-1": {"case_id": "clock", "mode": "auto"}
        }


def test_final_report_uses_exclusive_create(tmp_path):
    output = tmp_path / "report.json"
    digest = write_exclusive_json(output, {"schema_version": 2, "passed": True})

    assert len(digest) == 64
    assert json.loads(output.read_text())["passed"] is True
    with pytest.raises(FileExistsError):
        write_exclusive_json(output, {"schema_version": 2, "passed": False})


def _complete_sample(journal):
    journal.append("SAMPLE_STARTED", {"case_id": "clock"}, sample_id="sample-1")
    journal.append("SAMPLE_TERMINAL", {
        "outcome": "COMPLETED", "sample": {"case_id": "clock", "passed": True},
    }, sample_id="sample-1")


def test_report_prepare_crash_resumes_publish_without_rewriting_samples(tmp_path):
    journal_path = tmp_path / "semantic.journal.jsonl"
    output = tmp_path / "report.json"
    report = {"schema_version": 2, "created_at": "frozen", "passed": True}
    finalization = {"gate": {"passed": True}, "sample_terminal_sha256": ["sample"]}

    with EvaluationJournal(journal_path, resume=False) as journal:
        _start(journal)
        _complete_sample(journal)
        prepared = journal.prepare_report(output, report, finalization=finalization)
        assert journal.report_prepared is True
        assert output.exists() is False
        assert Path(prepared["prepared_path"]).is_file()

    with EvaluationJournal(journal_path, resume=True) as journal:
        journal.prepare_report(output, report, finalization=finalization)
        journal.publish_prepared_report(output)
        finalized = journal.finalize_report(output)
        assert finalized["gate"]["passed"] is True
        assert journal.load_published_report(output) == report
        assert len([record for record in journal.records
                    if record["record_type"] == "SAMPLE_TERMINAL"]) == 1


def test_published_report_crash_resumes_finalization_idempotently(tmp_path):
    journal_path = tmp_path / "semantic.journal.jsonl"
    output = tmp_path / "report.json"
    report = {"schema_version": 2, "created_at": "frozen", "passed": True}

    with EvaluationJournal(journal_path, resume=False) as journal:
        _start(journal)
        _complete_sample(journal)
        journal.prepare_report(output, report, finalization={"gate": {"passed": True}})
        journal.publish_prepared_report(output)
        assert output.is_file()
        assert journal.finalized is False

    with EvaluationJournal(journal_path, resume=True) as journal:
        journal.publish_prepared_report(output)
        journal.finalize_report(output)
        assert journal.finalized is True

    with EvaluationJournal(journal_path, resume=True) as journal:
        assert journal.finalize_report(output)["gate"]["passed"] is True
        assert journal.load_published_report(output) == report


def test_finalization_rejects_missing_or_tampered_published_report(tmp_path):
    journal_path = tmp_path / "semantic.journal.jsonl"
    output = tmp_path / "report.json"
    report = {"schema_version": 2, "created_at": "frozen", "passed": True}

    with EvaluationJournal(journal_path, resume=False) as journal:
        _start(journal)
        _complete_sample(journal)
        journal.prepare_report(output, report, finalization={"gate": {"passed": True}})
        with pytest.raises(JournalError, match="可靠发布"):
            journal.append("RUN_FINALIZED", {
                "report_sha256": journal.prepared_report["report_sha256"],
                "output_path": str(output.resolve()),
            })
        journal.publish_prepared_report(output)
        output.write_text('{"tampered":true}\n', encoding="utf-8")
        with pytest.raises(JournalError, match="不一致"):
            journal.finalize_report(output)
