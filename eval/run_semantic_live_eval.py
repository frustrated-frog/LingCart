"""Stage-bound, resumable live evaluation for semantic understanding.

The command is dry-run by default.  A paid run requires ``--live``, a frozen
stage, an exclusive output path and (for later stages) verified prerequisite
receipts.  Every sample is fsynced to a hash-chained journal before another
provider request can begin.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import uuid
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(__file__).with_name("semantic_cases.json")
ACCEPTANCE_SPEC = ROOT / "research/experiments/semantic-understanding-v1/acceptance-spec.json"
DEFAULT_STAGE = "single-smoke"
MODES = ("auto", "adaptive")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_journal import EvaluationJournal, JournalError, write_exclusive_json
from eval.semantic_contracts import (
    AcceptanceSpec,
    AcceptanceStage,
    GateReceipt,
    SemanticCase,
    SemanticDataset,
    StageBudget,
    StageExecution,
    StageSelection,
    canonical_json,
    fingerprint,
    load_acceptance_spec_payload,
    load_dataset_payload,
)
from eval.semantic_oracle import apply_mode_consistency, score_semantic_result


def load_dataset(path: Path = DATASET) -> SemanticDataset:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取语义数据集：{path}") from exc
    return load_dataset_payload(payload)


def load_cases(path: Path = DATASET) -> dict:
    """Compatibility helper returning the validated persisted representation."""

    return load_dataset(path).model_dump(mode="json")


def load_acceptance_spec(path: Path = ACCEPTANCE_SPEC) -> AcceptanceSpec:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取语义验收契约：{path}") from exc
    return load_acceptance_spec_payload(payload)


def validate_dataset_binding(dataset: SemanticDataset, spec: AcceptanceSpec, dataset_path: Path) -> None:
    reference = spec.dataset
    try:
        relative = str(Path(dataset_path).resolve().relative_to(ROOT))
    except ValueError:
        relative = str(Path(dataset_path).resolve())
    if (
        relative != reference.path
        or dataset.dataset_id != reference.dataset_id
        or dataset.dataset_version != reference.dataset_version
        or len(dataset.cases) != reference.case_count
        or fingerprint(dataset.model_dump(mode="json")) != reference.sha256
    ):
        raise ValueError("数据集身份、版本、数量或指纹与 acceptance-spec 不一致")


def _dataset_object(dataset: SemanticDataset | dict) -> SemanticDataset:
    return dataset if isinstance(dataset, SemanticDataset) else load_dataset_payload(dataset)


def select_cases(
    dataset: SemanticDataset | dict,
    *,
    case_ids: list[str] | tuple[str, ...] | None = None,
    groups: list[str] | tuple[str, ...] | None = None,
    max_cases: int | None = None,
    all_cases: bool = False,
) -> list[dict]:
    model = _dataset_object(dataset)
    cases = list(model.cases)
    if case_ids:
        index = {case.id: case for case in cases}
        missing = [case_id for case_id in case_ids if case_id not in index]
        if missing:
            raise ValueError("未知语义评测案例：" + ", ".join(missing))
        cases = [index[case_id] for case_id in dict.fromkeys(case_ids)]
    elif groups:
        requested = set(groups)
        available = {case.risk_group for case in cases}
        missing = sorted(requested - available)
        if missing:
            raise ValueError("未知语义风险组：" + ", ".join(missing))
        cases = [case for case in cases if case.risk_group in requested]
    elif not all_cases:
        cases = cases[: max_cases or 1]
    if max_cases is not None:
        if isinstance(max_cases, bool) or max_cases < 1:
            raise ValueError("--max-cases 必须是正整数")
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("至少需要选择一个语义评测案例")
    return [case.model_dump(mode="json") for case in cases]


def planned_executions(cases: list[dict], *, repeats: int, modes: tuple[str, ...]) -> int:
    if isinstance(repeats, bool) or repeats < 1:
        raise ValueError("--repeats 必须是正整数")
    return len(cases) * repeats * len(modes)


def _history_messages(case: SemanticCase) -> list:
    from langchain_core.messages import AIMessage, HumanMessage

    factories = {"human": HumanMessage, "assistant": AIMessage}
    return [factories[item.role](content=item.content) for item in case.history]


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True,
    )
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def _tracked_worktree_clean() -> bool:
    completed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def _file_manifest(paths: list[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths)) if path.exists() and path.is_file()
    }


def _source_manifest() -> dict:
    declared = [
        ROOT / "eval/run_semantic_live_eval.py",
        ROOT / "eval/semantic_contracts.py",
        ROOT / "eval/semantic_oracle.py",
        ROOT / "eval/eval_journal.py",
        ROOT / "src/dialogue_understanding.py",
        ROOT / "src/capability_registry.py",
        ROOT / "src/orchestrator.py",
        ROOT / "src/adaptive.py",
        ROOT / "src/runtime.py",
        ROOT / "src/trace_projector.py",
        *list((ROOT / "configs/harness").glob("*.json")),
    ]
    files = _file_manifest(declared)
    dependency_lock = ROOT / "requirements.lock"
    return {
        "source_revision": _git_commit(),
        "source_files_sha256": files,
        "source_tree_sha256": fingerprint(files),
        "evaluator_protocol_sha256": fingerprint({
            name: digest for name, digest in files.items() if name.startswith("eval/")
        }),
        "dependency_lock_sha256": (
            hashlib.sha256(dependency_lock.read_bytes()).hexdigest() if dependency_lock.exists() else None
        ),
        "python_runtime": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "business_environment": "local_demo_sqlite",
    }


def _safe_model_configuration(readiness: dict) -> dict:
    allowed = {
        "provider", "model", "slot", "custom_base_url", "thinking_mode",
        "reasoning_effort", "strict_tools", "timeout_seconds", "max_attempts",
    }
    providers = [
        {key: provider.get(key) for key in sorted(allowed) if provider.get(key) is not None}
        for provider in readiness.get("providers") or []
        if isinstance(provider, dict)
    ]
    payload = {"providers": providers}
    return {**payload, "model_config_sha256": fingerprint(payload)}


def _runtime_default(mode: str):
    from src.runtime import BudgetLimits

    return BudgetLimits.adaptive() if mode == "adaptive" else BudgetLimits()


def effective_model_call_budget(mode: str, *, stage_cap: int, cli_cap: int | None = None) -> dict:
    for label, value in (("stage", stage_cap), ("cli", cli_cap)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            raise ValueError(f"{label} 模型调用上限必须是正整数")
    runtime_default = _runtime_default(mode).max_model_calls
    effective = min(runtime_default, stage_cap, cli_cap if cli_cap is not None else math.inf)
    return {
        "runtime_default": runtime_default,
        "stage_cap": stage_cap,
        "cli_cap": cli_cap,
        "effective": int(effective),
    }


def _runner_provenance(runner, *, mode: str, injected: bool) -> dict:
    candidate = getattr(runner, "harness_runner", runner)
    if hasattr(candidate, "provenance"):
        return candidate.provenance()
    return {
        "profile_id": f"injected-{mode}-runner" if injected else f"unresolved-{mode}",
        "profile_version": "test-only" if injected else "unknown",
        "profile_sha256": None,
        "execution_sha256": None,
    }


def _sample_id(case_id: str, repeat: int, execution: StageExecution) -> str:
    return fingerprint({
        "case_id": case_id,
        "repeat": repeat,
        "mode": execution.mode,
        "profile_id": execution.profile_id,
    })


def _build_sample_plan(cases: list[SemanticCase], repeats: int,
                       executions: tuple[StageExecution, ...], budgets: dict[str, dict]) -> list[dict]:
    return [
        {
            "sample_id": _sample_id(case.id, repeat, execution),
            "case_id": case.id,
            "case_sha256": fingerprint(case.model_dump(mode="json")),
            "risk_group": case.risk_group,
            "repeat": repeat,
            "mode": execution.mode,
            "profile_id": execution.profile_id,
            "effective_model_calls": budgets[execution.mode]["effective"],
        }
        for case in cases
        for repeat in range(1, repeats + 1)
        for execution in executions
    ]


def _usage_total(result: dict) -> int | None:
    total = (result.get("usage") or {}).get("total_tokens")
    return total if isinstance(total, int) and not isinstance(total, bool) else None


def _safe_result(result: dict) -> dict:
    keys = (
        "run_id", "status", "terminal_reason", "execution_mode", "route", "intake",
        "understanding", "plan", "events", "budget", "usage", "harness", "error",
    )
    payload = json.loads(json.dumps(
        {key: result.get(key) for key in keys if key in result},
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ))
    # Reports are evidence artifacts, not provider debug logs.  Keep stable
    # machine codes while dropping free-form exception strings and credential
    # shaped fields at every nesting level.
    sensitive = {
        "api_key", "apikey", "authorization", "proxy_authorization", "headers",
        "cookie", "set_cookie", "access_token", "refresh_token", "secret",
    }

    def redact(value, *, error_context=False):
        if isinstance(value, list):
            return [redact(item, error_context=error_context) for item in value]
        if not isinstance(value, dict):
            return value
        is_error = error_context or value.get("type") == "error"
        cleaned = {}
        for key, item in value.items():
            normalized = key.casefold().replace("-", "_")
            if normalized in sensitive or any(
                token in normalized for token in ("api_key", "auth_token", "credential", "password", "client_secret")
            ):
                continue
            if is_error and key in {"message", "detail", "traceback", "raw_error"}:
                continue
            cleaned[key] = redact(item, error_context=is_error or key == "error")
        return cleaned

    return redact(payload)


def _exception_sample(plan: dict, exc: Exception, latency_ms: float, *, outcome: str = "EXCEPTION") -> dict:
    code = getattr(exc, "code", type(exc).__name__)
    retryable = bool(getattr(exc, "retryable", False))
    return {
        **plan,
        "outcome": outcome,
        "passed": False,
        "failure_codes": [outcome],
        "failures": [{"code": outcome, "message": "评测样本未产生可验证终态"}],
        "goal_recall": 0.0,
        "false_fast_path": False,
        "model_calls": None,
        "usage": None,
        "latency_ms": round(latency_ms, 2),
        "exception": {
            "phase": "RUNNER" if outcome == "EXCEPTION" else "RECOVERY",
            "code": str(code)[:120],
            "type": type(exc).__name__,
            "retryable": retryable,
            "safe_message": "模型或运行时执行失败；原始异常未写入，避免泄露凭据",
        },
        "raw_result": {},
    }


def score_result(case: dict | SemanticCase, result: dict, intake=None, *, mode: str = "auto") -> dict:
    del intake
    model = case if isinstance(case, SemanticCase) else SemanticCase.model_validate_json(canonical_json(case))
    return score_semantic_result(model, result, mode=mode).payload


def _summarize(samples: list[dict], consistency: list[bool]) -> dict:
    latencies = [sample["latency_ms"] for sample in samples if isinstance(sample.get("latency_ms"), (int, float))]
    calls = [sample.get("model_calls") for sample in samples]
    tokens = [_usage_total(sample.get("raw_result") or {}) for sample in samples]
    return {
        "executions": len(samples),
        "pass_rate": sum(bool(sample.get("passed")) for sample in samples) / len(samples) if samples else None,
        "goal_recall": statistics.mean(sample.get("goal_recall", 0) for sample in samples) if samples else None,
        "false_fast_path_rate": (
            sum(bool(sample.get("false_fast_path")) for sample in samples) / len(samples) if samples else None
        ),
        "auto_adaptive_consistency": statistics.mean(consistency) if consistency else None,
        "model_calls": sum(calls) if calls and all(isinstance(item, int) for item in calls) else None,
        "total_tokens": sum(tokens) if tokens and all(isinstance(item, int) for item in tokens) else None,
        "latency_ms": {
            "mean": statistics.mean(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
    }


def _prepare_execution(
    dataset: SemanticDataset,
    cases: list[SemanticCase],
    stage: AcceptanceStage,
    *,
    runners: dict[str, Callable] | None,
    readiness: dict,
    cli_model_cap: int | None,
    spec: AcceptanceSpec | None,
) -> tuple[dict[str, Callable], dict, list[dict], dict]:
    injected = runners is not None
    resolved: dict[str, Callable] = dict(runners or {})
    if not injected:
        from src.harness import get_runner

        for execution in stage.executions:
            resolved[execution.mode] = get_runner(execution.profile_id)
    missing = {execution.mode for execution in stage.executions} - set(resolved)
    if missing:
        raise ValueError("缺少执行器：" + ", ".join(sorted(missing)))
    profiles = {}
    for execution in stage.executions:
        runner = resolved[execution.mode]
        actual_mode = getattr(runner, "execution_mode", execution.mode)
        if actual_mode != execution.mode:
            raise ValueError(f"Profile {execution.profile_id} 的执行模式不是 {execution.mode}")
        provenance = _runner_provenance(runner, mode=execution.mode, injected=injected)
        if not injected and provenance.get("profile_id") != execution.profile_id:
            raise ValueError(f"阶段要求 {execution.profile_id}，实际解析为 {provenance.get('profile_id')}")
        profiles[execution.mode] = provenance
    budgets = {
        execution.mode: effective_model_call_budget(
            execution.mode,
            stage_cap=stage.budget.max_model_calls_per_run,
            cli_cap=cli_model_cap,
        )
        for execution in stage.executions
    }
    plan = _build_sample_plan(cases, stage.repeats, stage.executions, budgets)
    source = _source_manifest()
    manifest = {
        "git_commit": _git_commit(),
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "dataset_sha256": fingerprint(dataset.model_dump(mode="json")),
        "case_sha256": {case.id: fingerprint(case.model_dump(mode="json")) for case in cases},
        "acceptance_spec_sha256": fingerprint(spec.model_dump(mode="json")) if spec else None,
        "stage_id": stage.id,
        "stage_contract_sha256": fingerprint(stage.model_dump(mode="json")),
        "gate_type": stage.gate_type,
        "claim_scope": stage.claim_scope,
        "selected_case_ids": [case.id for case in cases],
        "risk_groups": sorted({case.risk_group for case in cases}),
        "repeats": stage.repeats,
        "executions": [item.model_dump(mode="json") for item in stage.executions],
        "profiles": profiles,
        "model_call_budgets": budgets,
        "model_configuration": _safe_model_configuration(readiness),
        "oracle_protocol": (
            spec.oracle_protocol.model_dump(mode="json") if spec else {"id": "semantic-goal-oracle", "version": 1}
        ),
        **source,
    }
    identity_fields = {
        key: manifest[key] for key in (
            "dataset_sha256", "acceptance_spec_sha256", "stage_contract_sha256", "profiles",
            "model_call_budgets", "model_configuration", "oracle_protocol", "source_tree_sha256",
            "dependency_lock_sha256", "python_runtime", "business_environment",
        )
    }
    identity_fields["sample_plan"] = plan
    manifest["run_identity_sha256"] = fingerprint(identity_fields)
    return resolved, manifest, plan, budgets


def evaluate(
    dataset: SemanticDataset | dict,
    *,
    cases: list[dict] | list[SemanticCase],
    repeats: int,
    modes: tuple[str, ...],
    max_model_calls_per_run: int,
    readiness: dict,
    runners: dict[str, Callable] | None = None,
    sleep_seconds: float = 0,
    stage: AcceptanceStage | None = None,
    spec: AcceptanceSpec | None = None,
    journal: EvaluationJournal | None = None,
    stop_on_exception: bool = True,
) -> dict:
    """Execute an approved plan; tests inject non-network runners."""

    dataset_model = _dataset_object(dataset)
    case_models = [
        item if isinstance(item, SemanticCase) else SemanticCase.model_validate_json(canonical_json(item))
        for item in cases
    ]
    if stage is None:
        executions = tuple(StageExecution(mode=mode, profile_id=(
            "adaptive-v1" if mode == "adaptive" else "baseline-v1"
        )) for mode in modes)
        stage = AcceptanceStage(
            id="ad-hoc-smoke",
            gate_type="smoke",
            claim_scope="selected_cases_only",
            selection=StageSelection(case_ids=tuple(case.id for case in case_models), all_cases=False),
            executions=executions,
            repeats=repeats,
            budget=StageBudget(
                max_executions=len(case_models) * repeats * len(executions),
                max_model_calls_per_run=max_model_calls_per_run,
            ),
            requires=(),
        )
    elif repeats != stage.repeats or modes != tuple(item.mode for item in stage.executions):
        raise ValueError("执行次数或模式与冻结阶段不一致")
    resolved, manifest, sample_plan, budgets = _prepare_execution(
        dataset_model,
        case_models,
        stage,
        runners=runners,
        readiness=readiness,
        cli_model_cap=max_model_calls_per_run,
        spec=spec,
    )
    execution_source = "test_harness" if runners is not None else "real_model"
    existing: dict[str, dict] = {}
    if journal is not None:
        journal.start(
            evaluation_id=journal.evaluation_id or uuid.uuid4().hex,
            run_identity_sha256=manifest["run_identity_sha256"],
            manifest=manifest,
            sample_plan=sample_plan,
        )
        journal.require_identity(manifest["run_identity_sha256"])
        existing = journal.terminal_samples()
        for sample_id, started in journal.orphaned_samples().items():
            plan = next((item for item in sample_plan if item["sample_id"] == sample_id), started)
            sample = _exception_sample(
                plan,
                JournalError("样本已开始但没有可证明安全恢复的 Runtime handle"),
                0,
                outcome="RECOVERY_BLOCKED",
            )
            journal.append("SAMPLE_TERMINAL", {"outcome": "RECOVERY_BLOCKED", "sample": sample},
                           sample_id=sample_id)
            existing[sample_id] = sample
    case_index = {case.id: case for case in case_models}
    execution_index = {item.mode: item for item in stage.executions}
    samples: list[dict] = []
    interrupted = False
    for plan in sample_plan:
        sample_id = plan["sample_id"]
        if sample_id in existing:
            samples.append(existing[sample_id])
            continue
        case = case_index[plan["case_id"]]
        execution = execution_index[plan["mode"]]
        if journal is not None:
            journal.append("SAMPLE_STARTED", {
                **plan,
                "started_at": datetime.now(UTC).isoformat(),
                "effective_budget": budgets[execution.mode],
            }, sample_id=sample_id)
        started_at = time.perf_counter()
        try:
            runner = resolved[execution.mode]
            history = _history_messages(case)
            base = _runtime_default(execution.mode)
            limits = replace(base, max_model_calls=budgets[execution.mode]["effective"])
            result = runner(case.message, history=history or None, limits=limits)
            score = score_semantic_result(case, result, mode=execution.mode).payload
            budget = result.get("budget") or {}
            sample = {
                **plan,
                "outcome": "COMPLETED",
                **score,
                "status": result.get("status"),
                "terminal_reason": result.get("terminal_reason"),
                "usage": result.get("usage"),
                "model_calls": (budget.get("consumed") or {}).get("model_calls"),
                "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "model_providers": [
                    {key: event.get(key) for key in ("purpose", "provider", "model", "provider_slot")
                     if event.get(key) is not None}
                    for event in result.get("events") or []
                    if event.get("type") == "model_call" and event.get("status") == "SUCCEEDED"
                ],
                "raw_result": _safe_result(result),
            }
            outcome = "COMPLETED"
        except Exception as exc:
            sample = _exception_sample(plan, exc, (time.perf_counter() - started_at) * 1000)
            outcome = "EXCEPTION"
            interrupted = stop_on_exception
        samples.append(sample)
        if journal is not None:
            journal.append("SAMPLE_TERMINAL", {"outcome": outcome, "sample": sample}, sample_id=sample_id)
        print(
            f"{'PASS' if sample.get('passed') else 'FAIL'} {case.id} "
            f"repeat={plan['repeat']} mode={execution.mode}: "
            f"{', '.join(sample.get('failure_codes') or []) or 'ok'}"
        )
        if interrupted:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    expected_ids = {item["sample_id"] for item in sample_plan}
    actual_ids = {item["sample_id"] for item in samples}
    complete = actual_ids == expected_ids
    consistency = apply_mode_consistency(samples) if complete else []
    report = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "COMPLETED" if complete else "INTERRUPTED",
        "execution_source": execution_source,
        "manifest": manifest,
        "sample_plan": sample_plan,
        "metrics": _summarize(samples, consistency),
        "samples": samples,
        "journal": {
            "enabled": journal is not None,
            "path": str(journal.path) if journal is not None else None,
            "hash_chain_valid": True if journal is not None else None,
            "samples_committed": len(samples),
            "expected_samples": len(sample_plan),
        },
    }
    return report


def _stage_cases(dataset: SemanticDataset, stage: AcceptanceStage) -> list[SemanticCase]:
    if stage.selection.all_cases:
        return list(dataset.cases)
    index = {case.id: case for case in dataset.cases}
    missing = [case_id for case_id in stage.selection.case_ids if case_id not in index]
    if missing:
        raise ValueError("阶段引用了未知案例：" + ", ".join(missing))
    return [index[case_id] for case_id in stage.selection.case_ids]


def _gate_failure(code: str, message: str, **details) -> dict:
    return {"code": code, "message": message, **details}


def _gate_report(report: dict, dataset: SemanticDataset, stage: AcceptanceStage,
                 gate_type: str, claim_scope: str) -> GateReceipt:
    failures: list[dict] = []
    checked_payload = {key: value for key, value in report.items() if key != "gate"}
    report_sha = fingerprint(checked_payload)
    if report.get("status") != "COMPLETED":
        failures.append(_gate_failure("REPORT_INCOMPLETE", "评测没有完成冻结样本网格"))
    if report.get("execution_source") != "real_model":
        failures.append(_gate_failure("NOT_LIVE_PROVIDER", "门禁拒绝测试替身或手工 execution_source 声明"))
    manifest = report.get("manifest") or {}
    if manifest.get("dataset_sha256") != fingerprint(dataset.model_dump(mode="json")):
        failures.append(_gate_failure("DATASET_IDENTITY_MISMATCH", "报告数据集指纹无法重算"))
    if manifest.get("stage_contract_sha256") != fingerprint(stage.model_dump(mode="json")):
        failures.append(_gate_failure("STAGE_IDENTITY_MISMATCH", "报告阶段契约指纹不一致"))
    if not manifest.get("source_tree_sha256") or not manifest.get("evaluator_protocol_sha256"):
        failures.append(_gate_failure("SOURCE_IDENTITY_MISSING", "报告缺少执行源码或 Oracle 指纹"))
    profiles = manifest.get("profiles") or {}
    if not profiles or any(
        not isinstance(profile, dict)
        or not profile.get("profile_sha256")
        or not profile.get("execution_sha256")
        for profile in profiles.values()
    ):
        failures.append(_gate_failure("PROFILE_IDENTITY_MISSING", "报告缺少可重算的 Harness Profile 执行身份"))
    cases = _stage_cases(dataset, stage)
    budgets = manifest.get("model_call_budgets") or {}
    expected = {
        (case.id, repeat, execution.mode, execution.profile_id)
        for case in cases
        for repeat in range(1, stage.repeats + 1)
        for execution in stage.executions
    }
    samples = report.get("samples") or []
    actual = [
        (sample.get("case_id"), sample.get("repeat"), sample.get("mode"), sample.get("profile_id"))
        for sample in samples
    ]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        failures.append(_gate_failure("SAMPLE_GRID_INCOMPLETE", "样本缺失、重复或模式/Profile 错配"))
    case_index = {case.id: case for case in cases}
    configured_models = {
        (item.get("provider"), item.get("model"))
        for item in (manifest.get("model_configuration") or {}).get("providers") or []
    }
    rescored = []
    for sample in samples:
        case = case_index.get(sample.get("case_id"))
        raw = sample.get("raw_result") or {}
        if case is None or sample.get("outcome") != "COMPLETED" or not raw:
            failures.append(_gate_failure("SAMPLE_NOT_COMPLETED", "样本异常或恢复受阻",
                                          sample_id=sample.get("sample_id")))
            continue
        fresh = score_semantic_result(case, raw, mode=sample.get("mode") or "auto").payload
        rescored_sample = {**sample, **fresh}
        rescored.append(rescored_sample)
        if not fresh["passed"]:
            failures.append(_gate_failure("ORACLE_REJECTED", "目标级 Oracle 拒绝样本",
                                          sample_id=sample.get("sample_id"),
                                          failure_codes=fresh["failure_codes"]))
        mode = sample.get("mode")
        budget = budgets.get(mode) or {}
        effective = budget.get("effective")
        returned = ((raw.get("budget") or {}).get("limits") or {}).get("max_model_calls")
        consumed = ((raw.get("budget") or {}).get("consumed") or {}).get("model_calls")
        if returned != effective or not isinstance(consumed, int) or consumed > effective:
            failures.append(_gate_failure("BUDGET_EVIDENCE_MISMATCH", "Runtime 预算与冻结 effective limit 不一致",
                                          sample_id=sample.get("sample_id")))
        attempts = sum(
            event.get("type") in {"model_started", "model_retry"}
            for event in raw.get("events") or []
        )
        if consumed != attempts:
            failures.append(_gate_failure("MODEL_ATTEMPT_COUNT_MISMATCH",
                                          "模型开始/重试事件与预算计数不一致",
                                          sample_id=sample.get("sample_id")))
        if case.execution_policy.semantic_path == "required":
            calls = [event for event in raw.get("events") or [] if event.get("type") == "model_call"
                     and event.get("status") == "SUCCEEDED" and event.get("purpose") == "dialogue_understanding"
                     and event.get("provider") and event.get("model")]
            if not calls:
                failures.append(_gate_failure("PROVIDER_EVIDENCE_MISSING", "语义案例缺少真实 Provider 身份证据",
                                              sample_id=sample.get("sample_id")))
            elif configured_models and any(
                (event.get("provider"), event.get("model")) not in configured_models for event in calls
            ):
                failures.append(_gate_failure("PROVIDER_IDENTITY_MISMATCH",
                                              "样本实际 Provider/Model 不在冻结模型配置中",
                                              sample_id=sample.get("sample_id")))
    consistency = apply_mode_consistency(rescored)
    if len(stage.executions) == 2 and (not consistency or not all(consistency)):
        failures.append(_gate_failure("AUTO_ADAPTIVE_SEMANTIC_MISMATCH",
                                      "Auto/Adaptive 目标、实体、结果或终态投影不一致"))
    journal = report.get("journal") or {}
    if journal.get("enabled") and (
        journal.get("hash_chain_valid") is not True
        or journal.get("samples_committed") != journal.get("expected_samples")
    ):
        failures.append(_gate_failure("JOURNAL_INCOMPLETE", "Journal 哈希链或样本提交数量不完整"))
    codes = tuple(dict.fromkeys(item["code"] for item in failures))
    return GateReceipt(
        gate_type=gate_type,
        stage_id=stage.id,
        claim_scope=claim_scope,
        checked_report_sha256=report_sha,
        passed=not failures,
        failure_codes=codes,
        failures=tuple(failures),
    )


class SmokeGate:
    @staticmethod
    def evaluate(report: dict, dataset: SemanticDataset, stage: AcceptanceStage) -> GateReceipt:
        if stage.gate_type != "smoke":
            raise ValueError("SmokeGate 只能检查 smoke 阶段")
        return _gate_report(report, dataset, stage, "SMOKE", "selected_cases_only")


class AcceptanceGate:
    @staticmethod
    def evaluate(report: dict, dataset: SemanticDataset, stage: AcceptanceStage) -> GateReceipt:
        if stage.gate_type != "acceptance":
            raise ValueError("AcceptanceGate 不能把 Smoke 扩大为验收结论")
        return _gate_report(report, dataset, stage, "ACCEPTANCE", "frozen_stage")


class ExperimentGate:
    @staticmethod
    def evaluate(baseline: dict, candidate: dict, *, stage_id: str = "paired-experiment") -> GateReceipt:
        failures = []
        for label, report in (("baseline", baseline), ("candidate", candidate)):
            gate = report.get("gate") or {}
            if gate.get("gate_type") != "ACCEPTANCE" or gate.get("passed") is not True:
                failures.append(_gate_failure("INPUT_NOT_ACCEPTANCE", f"{label} 不是通过的 Acceptance 报告"))
        left, right = baseline.get("manifest") or {}, candidate.get("manifest") or {}
        for key in (
            "dataset_sha256", "acceptance_spec_sha256", "stage_contract_sha256",
            "model_configuration", "model_call_budgets", "dependency_lock_sha256",
            "python_runtime", "business_environment", "evaluator_protocol_sha256",
        ):
            if left.get(key) != right.get(key):
                failures.append(_gate_failure("EXPERIMENT_IDENTITY_MISMATCH",
                                              f"实验配对字段不一致：{key}", field=key))
        def grid(report):
            return {(sample.get("case_sha256"), sample.get("repeat"), sample.get("mode"))
                    for sample in report.get("samples") or []}
        if grid(baseline) != grid(candidate):
            failures.append(_gate_failure("EXPERIMENT_SAMPLE_PAIRING_MISMATCH", "基线与候选样本无法一一配对"))
        checked = fingerprint({"baseline": baseline, "candidate": candidate})
        return GateReceipt(
            gate_type="EXPERIMENT",
            stage_id=stage_id,
            claim_scope="paired_experiment",
            checked_report_sha256=checked,
            passed=not failures,
            failure_codes=tuple(dict.fromkeys(item["code"] for item in failures)),
            failures=tuple(failures),
        )


def evaluate_gate(report: dict, dataset: SemanticDataset | None = None,
                  stage: AcceptanceStage | None = None) -> list[str]:
    """Compatibility facade returning human-readable failures."""

    if dataset is None or stage is None:
        gate = report.get("gate") or {}
        if gate:
            return [item.get("message", item.get("code", "门禁失败")) for item in gate.get("failures") or []]
        return ["缺少数据集与阶段，无法从原始样本重算 Gate"]
    receipt = (SmokeGate if stage.gate_type == "smoke" else AcceptanceGate).evaluate(report, dataset, stage)
    return [item.get("message", item.get("code", "门禁失败")) for item in receipt.failures]


def _ad_hoc_stage(args, cases: list[dict], modes: tuple[str, ...], base: AcceptanceStage) -> AcceptanceStage:
    repeats = args.repeats if args.repeats is not None else 1
    requested_budget = args.max_model_calls_per_run or base.budget.max_model_calls_per_run
    executions = tuple(StageExecution(
        mode=mode,
        profile_id="adaptive-v1" if mode == "adaptive" else "baseline-v1",
    ) for mode in modes)
    return AcceptanceStage(
        id="ad-hoc-smoke",
        gate_type="smoke",
        claim_scope="selected_cases_only",
        selection=StageSelection(case_ids=tuple(case["id"] for case in cases), all_cases=False),
        executions=executions,
        repeats=repeats,
        budget=StageBudget(
            max_executions=len(cases) * repeats * len(executions),
            max_model_calls_per_run=min(base.budget.max_model_calls_per_run, requested_budget),
        ),
        requires=(),
    )


def _validate_prerequisites(paths: list[Path], stage: AcceptanceStage, dataset: SemanticDataset,
                            spec: AcceptanceSpec) -> list[dict]:
    reports = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append(report)
    by_stage = {(report.get("gate") or {}).get("stage_id"): report for report in reports}
    for required in stage.requires:
        report = by_stage.get(required)
        gate = (report or {}).get("gate") or {}
        manifest = (report or {}).get("manifest") or {}
        if (
            gate.get("passed") is not True
            or manifest.get("dataset_sha256") != fingerprint(dataset.model_dump(mode="json"))
            or manifest.get("acceptance_spec_sha256") != fingerprint(spec.model_dump(mode="json"))
        ):
            raise ValueError(f"阶段 {stage.id} 缺少同一契约下通过的前置报告：{required}")
    return reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--live", action="store_true", help="显式允许真实模型调用；默认只预检")
    execution.add_argument("--dry-run", action="store_true", help="只校验并打印冻结执行计划")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--spec", type=Path, default=ACCEPTANCE_SPEC)
    parser.add_argument("--stage", default=DEFAULT_STAGE)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--group", action="append", dest="groups")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--mode", choices=("auto", "adaptive", "both"))
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-executions", type=int)
    parser.add_argument("--max-model-calls-per-run", type=int)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prerequisite-report", type=Path, action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sleep < 0:
        parser.error("--sleep 不能小于 0")
    selectors = sum(bool(value) for value in (args.all_cases, args.case_ids, args.groups, args.max_cases))
    if selectors > 1:
        parser.error("--all-cases、--case、--group 与 --max-cases 只能选择一种")
    try:
        dataset = load_dataset(args.dataset)
        spec = load_acceptance_spec(args.spec)
        validate_dataset_binding(dataset, spec, args.dataset)
        frozen_stage = spec.stage(args.stage)
        ad_hoc = selectors > 0 or args.mode is not None or args.repeats is not None
        if ad_hoc:
            selected = select_cases(
                dataset,
                case_ids=args.case_ids,
                groups=args.groups,
                max_cases=args.max_cases,
                all_cases=args.all_cases,
            )
            modes = MODES if args.mode == "both" else (args.mode or frozen_stage.executions[0].mode,)
            stage = _ad_hoc_stage(args, selected, modes, frozen_stage)
        else:
            stage = frozen_stage
            selected = [case.model_dump(mode="json") for case in _stage_cases(dataset, stage)]
            modes = tuple(item.mode for item in stage.executions)
        executions = planned_executions(selected, repeats=stage.repeats, modes=modes)
        max_executions = min(
            stage.budget.max_executions,
            args.max_executions if args.max_executions is not None else stage.budget.max_executions,
        )
        if executions > max_executions:
            raise ValueError(f"冻结计划需要 {executions} 次执行，当前上限为 {max_executions}")
        cli_model_cap = args.max_model_calls_per_run
        budgets = {
            mode: effective_model_call_budget(
                mode,
                stage_cap=stage.budget.max_model_calls_per_run,
                cli_cap=cli_model_cap,
            )
            for mode in modes
        }
        prerequisites = _validate_prerequisites(args.prerequisite_report, stage, dataset, spec)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    plan = {
        "mode": "live" if args.live else "dry_run",
        "stage_id": stage.id,
        "gate_type": stage.gate_type,
        "claim_scope": stage.claim_scope,
        "ad_hoc": ad_hoc,
        "dataset_sha256": fingerprint(dataset.model_dump(mode="json")),
        "acceptance_spec_sha256": fingerprint(spec.model_dump(mode="json")),
        "selected_case_ids": [case["id"] for case in selected],
        "repeats": stage.repeats,
        "execution_modes": list(modes),
        "planned_executions": executions,
        "model_call_budgets": budgets,
        "worst_case_model_calls": sum(
            len(selected) * stage.repeats * budgets[mode]["effective"] for mode in modes
        ),
        "prerequisite_reports": [fingerprint(report) for report in prerequisites],
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.live:
        print("预检完成：未调用模型。只有 --live、不可变 --output 和满足前置 Gate 才会执行。")
        return
    if args.output is None:
        parser.error("真实评测必须通过 --output 保存不可变报告")
    if args.output.exists():
        parser.error(f"拒绝覆盖已有报告：{args.output}")
    journal_path = args.output.with_suffix(".journal.jsonl")
    if args.resume and not journal_path.exists():
        parser.error(f"续跑要求已有 Journal：{journal_path}")
    if not args.resume and journal_path.exists():
        parser.error(f"Journal 已存在，拒绝隐式续跑：{journal_path}")
    if not _tracked_worktree_clean():
        parser.error("真实评测要求受版本控制的文件保持干净，请先提交实现再运行")

    from src.llm import get_model_status

    readiness = get_model_status()
    if not readiness.get("configured") or not readiness.get("ready"):
        parser.error("真实模型尚未就绪：" + readiness.get("message", "未知原因"))
    try:
        with EvaluationJournal(journal_path, resume=args.resume) as journal:
            report = evaluate(
                dataset,
                cases=selected,
                repeats=stage.repeats,
                modes=modes,
                max_model_calls_per_run=(
                    args.max_model_calls_per_run or stage.budget.max_model_calls_per_run
                ),
                readiness=readiness,
                sleep_seconds=args.sleep,
                stage=stage,
                spec=spec,
                journal=journal,
            )
            if report["status"] != "COMPLETED":
                print(json.dumps({"status": report["status"], "journal": str(journal_path)}, ensure_ascii=False))
                raise SystemExit(2)
            gate = (SmokeGate if stage.gate_type == "smoke" else AcceptanceGate).evaluate(report, dataset, stage)
            report["gate"] = gate.model_dump(mode="json")
            final_sha = fingerprint(report)
            journal.append("RUN_FINALIZED", {
                "report_sha256": final_sha,
                "gate": gate.model_dump(mode="json"),
                "sample_terminal_sha256": [
                    record["record_sha256"] for record in journal.records
                    if record["record_type"] == "SAMPLE_TERMINAL"
                ],
            })
            write_exclusive_json(args.output, report)
    except (JournalError, ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({"metrics": report["metrics"], "gate": report["gate"]}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["gate"]["passed"] else 1)


if __name__ == "__main__":
    main()
