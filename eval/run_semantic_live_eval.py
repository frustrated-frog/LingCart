"""Low-cost live evaluation for semantic understanding and capability routing.

The command is deliberately dry-run by default. Real provider calls require
``--live``, an immutable output path, and an execution budget large enough for
the selected cases. This keeps a typo or exploratory invocation from running
the complete paid suite.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(__file__).with_name("semantic_cases.json")
DEFAULT_MAX_EXECUTIONS = 2
DEFAULT_MAX_MODEL_CALLS_PER_RUN = 4
MODES = ("auto", "adaptive")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_cases(path: Path) -> dict:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if dataset.get("schema_version") != 1 or not isinstance(dataset.get("cases"), list):
        raise ValueError("semantic dataset must use schema_version 1 and contain cases")
    seen = set()
    for case in dataset["cases"]:
        if not case.get("id") or case["id"] in seen or not case.get("message") or not case.get("expected"):
            raise ValueError(f"invalid or duplicate semantic case: {case.get('id')!r}")
        seen.add(case["id"])
    return dataset


def select_cases(dataset: dict, *, case_ids: list[str] | None = None,
                 max_cases: int | None = None, all_cases: bool = False) -> list[dict]:
    cases = list(dataset["cases"])
    if case_ids:
        index = {case["id"]: case for case in cases}
        missing = [case_id for case_id in case_ids if case_id not in index]
        if missing:
            raise ValueError("未知语义评测案例：" + ", ".join(missing))
        cases = [index[case_id] for case_id in dict.fromkeys(case_ids)]
    elif not all_cases:
        cases = cases[:max_cases or 1]
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("--max-cases 必须大于 0")
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("至少需要选择一个语义评测案例")
    return cases


def planned_executions(cases: list[dict], *, repeats: int, modes: tuple[str, ...]) -> int:
    if repeats < 1:
        raise ValueError("--repeats 必须大于 0")
    return len(cases) * repeats * len(modes)


def _subset(expected, actual) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and _subset(value, actual[key])
                                                for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and all(value in actual for value in expected)
    return expected == actual


def score_result(case: dict, result: dict, intake) -> dict:
    expected = case["expected"]
    understanding = result.get("understanding") or {}
    objectives = understanding.get("objectives") or []
    commands = understanding.get("commands") or []
    kinds = [item.get("kind") for item in objectives]
    command_kinds = [item.get("kind") for item in commands]
    failures = []
    for kind in expected.get("objectives", []):
        if kind not in kinds:
            failures.append(f"missing objective {kind}")
    for kind in expected.get("forbidden_objectives", []):
        if kind in kinds:
            failures.append(f"forbidden objective {kind}")
    for kind in expected.get("forbidden_commands", []):
        if kind in command_kinds:
            failures.append(f"forbidden command {kind}")
    if expected.get("commands_any") and not set(expected["commands_any"]) & set(command_kinds):
        failures.append("missing expected command")
    for kind, fields in expected.get("objective_fields", {}).items():
        candidate = next((item for item in objectives if item.get("kind") == kind), {})
        if not _subset(fields, candidate):
            failures.append(f"objective fields mismatch for {kind}")
    plan_steps = (result.get("plan") or {}).get("steps") or []
    for tool, arguments in expected.get("plan_args", {}).items():
        candidate = next((step.get("args", {}) for step in plan_steps if step.get("tool") == tool), {})
        if not _subset(arguments, candidate):
            failures.append(f"plan args mismatch for {tool}")
    if result.get("status") not in expected.get("statuses", []):
        failures.append(f"status {result.get('status')}")
    if expected.get("terminal_reason") and result.get("terminal_reason") != expected["terminal_reason"]:
        failures.append(f"terminal reason {result.get('terminal_reason')}")
    false_fast_path = bool(case.get("must_use_semantic") and intake.execution_mode != "LLM_REQUIRED")
    if false_fast_path:
        failures.append(f"unsafe fast path {intake.execution_mode}")
    required = set(expected.get("objectives", []))
    recall = len(required & set(kinds)) / len(required) if required else 1.0
    return {
        "passed": not failures,
        "failures": failures,
        "goal_recall": recall,
        "false_fast_path": false_fast_path,
        "objectives": kinds,
        "commands": command_kinds,
    }


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=False,
        capture_output=True, text=True,
    )
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def _tracked_worktree_clean() -> bool:
    completed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def _dataset_digest(dataset: dict) -> str:
    payload = json.dumps(dataset, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _usage_total(result: dict) -> int | None:
    total = (result.get("usage") or {}).get("total_tokens")
    return total if isinstance(total, int) else None


def evaluate(
    dataset: dict,
    *,
    cases: list[dict],
    repeats: int,
    modes: tuple[str, ...],
    max_model_calls_per_run: int,
    readiness: dict,
    runners: dict[str, Callable] | None = None,
    sleep_seconds: float = 0,
) -> dict:
    """Execute an already approved live plan; tests inject non-network runners."""

    from src.intake import analyze_request
    from src.runtime import BudgetLimits

    using_real_model = runners is None
    if max_model_calls_per_run < 1:
        raise ValueError("--max-model-calls-per-run 必须大于 0")
    if runners is None:
        from src.adaptive import run_adaptive
        from src.orchestrator import run_orchestrator
        runners = {"auto": run_orchestrator, "adaptive": run_adaptive}

    samples = []
    for case in cases:
        for repeat in range(1, repeats + 1):
            for mode in modes:
                intake = analyze_request(case["message"])
                base_limits = BudgetLimits.adaptive() if mode == "adaptive" else BudgetLimits()
                limits = replace(base_limits, max_model_calls=max_model_calls_per_run)
                started = time.perf_counter()
                result = runners[mode](case["message"], limits=limits)
                score = score_result(case, result, intake)
                budget = result.get("budget") or {}
                sample = {
                    "case_id": case["id"],
                    "repeat": repeat,
                    "mode": mode,
                    **score,
                    "status": result.get("status"),
                    "terminal_reason": result.get("terminal_reason"),
                    "usage": result.get("usage"),
                    "model_calls": (budget.get("consumed") or {}).get("model_calls"),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "understanding": result.get("understanding"),
                    "plan": result.get("plan"),
                    "tools": [entry.get("tool") for entry in result.get("trace") or []],
                    "model_providers": [
                        {
                            key: event.get(key)
                            for key in ("purpose", "provider", "model", "provider_slot")
                            if event.get(key) is not None
                        }
                        for event in result.get("events") or []
                        if event.get("type") == "model_call" and event.get("status") == "SUCCEEDED"
                    ],
                }
                samples.append(sample)
                print(
                    f"{'PASS' if sample['passed'] else 'FAIL'} "
                    f"{case['id']} repeat={repeat} mode={mode}: "
                    f"{', '.join(sample['failures']) or 'ok'}"
                )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

    consistency = []
    if set(modes) == set(MODES):
        for case in cases:
            for repeat in range(1, repeats + 1):
                pair = [sample for sample in samples
                        if sample["case_id"] == case["id"] and sample["repeat"] == repeat]
                by_mode = {sample["mode"]: sample for sample in pair}
                same = (
                    by_mode["auto"]["status"] == by_mode["adaptive"]["status"]
                    and by_mode["auto"]["terminal_reason"] == by_mode["adaptive"]["terminal_reason"]
                )
                consistency.append(same)
                if not same:
                    for sample in pair:
                        sample["passed"] = False
                        sample["failures"].append("Auto/Adaptive semantic mismatch")

    latencies = [sample["latency_ms"] for sample in samples]
    known_tokens = [total for sample in samples if (total := _usage_total(sample)) is not None]
    known_calls = [sample["model_calls"] for sample in samples if isinstance(sample["model_calls"], int)]
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "execution_source": "real_model" if using_real_model else "test_harness",
        "manifest": {
            "git_commit": _git_commit(),
            "dataset_sha256": _dataset_digest(dataset),
            "selected_case_ids": [case["id"] for case in cases],
            "repeats": repeats,
            "modes": list(modes),
            "max_model_calls_per_run": max_model_calls_per_run,
            "model_preflight": readiness,
        },
        "metrics": {
            "executions": len(samples),
            "pass_rate": sum(sample["passed"] for sample in samples) / len(samples),
            "goal_recall": statistics.mean(sample["goal_recall"] for sample in samples),
            "false_fast_path_rate": sum(sample["false_fast_path"] for sample in samples) / len(samples),
            "auto_adaptive_consistency": statistics.mean(consistency) if consistency else None,
            "model_calls": sum(known_calls) if len(known_calls) == len(samples) else None,
            "total_tokens": sum(known_tokens) if len(known_tokens) == len(samples) else None,
            "latency_ms": {"mean": statistics.mean(latencies), "max": max(latencies)},
        },
        "samples": samples,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--live", action="store_true", help="显式允许调用真实模型；默认仅预检")
    execution.add_argument("--dry-run", action="store_true", help="兼容旧命令；只校验并打印执行计划")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--case", action="append", dest="case_ids", help="只运行指定案例，可重复传入")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--all-cases", action="store_true", help="选择完整数据集，仍受执行预算限制")
    parser.add_argument("--mode", choices=("auto", "adaptive", "both"), default="auto")
    parser.add_argument("--compare-adaptive", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-executions", type=int, default=DEFAULT_MAX_EXECUTIONS)
    parser.add_argument("--max-model-calls-per-run", type=int, default=DEFAULT_MAX_MODEL_CALLS_PER_RUN)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.all_cases and (args.case_ids or args.max_cases is not None):
        parser.error("--all-cases 不能与 --case 或 --max-cases 同时使用")
    if args.max_executions < 1:
        parser.error("--max-executions 必须大于 0")
    if args.max_model_calls_per_run < 1:
        parser.error("--max-model-calls-per-run 必须大于 0")
    if args.sleep < 0:
        parser.error("--sleep 不能小于 0")

    try:
        dataset = load_cases(args.dataset)
        cases = select_cases(
            dataset,
            case_ids=args.case_ids,
            max_cases=args.max_cases,
            all_cases=args.all_cases,
        )
        modes = MODES if args.compare_adaptive or args.mode == "both" else (args.mode,)
        executions = planned_executions(cases, repeats=args.repeats, modes=modes)
    except ValueError as exc:
        parser.error(str(exc))

    plan = {
        "mode": "live" if args.live else "dry_run",
        "dataset_sha256": _dataset_digest(dataset),
        "selected_case_ids": [case["id"] for case in cases],
        "repeats": args.repeats,
        "execution_modes": list(modes),
        "planned_executions": executions,
        "max_executions": args.max_executions,
        "max_model_calls_per_run": args.max_model_calls_per_run,
        "worst_case_model_calls": executions * args.max_model_calls_per_run,
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.live:
        print("预检完成：未调用模型。传入 --live 和新的 --output 路径才会执行真实评测。")
        return
    if executions > args.max_executions:
        parser.error(
            f"计划执行 {executions} 次，超过 --max-executions={args.max_executions}；"
            "请缩小案例/模式/重复次数，或显式提高预算"
        )
    if args.output is None:
        parser.error("真实评测必须通过 --output 保存不可变报告")
    if args.output.exists():
        parser.error(f"拒绝覆盖已有报告：{args.output}")
    if not _tracked_worktree_clean():
        parser.error("真实评测要求受版本控制的文件保持干净，请先提交实现再运行")

    from src.llm import get_model_status

    readiness = get_model_status()
    if not readiness.get("configured") or not readiness.get("ready"):
        parser.error("真实模型尚未就绪：" + readiness.get("message", "未知原因"))
    report = evaluate(
        dataset,
        cases=cases,
        repeats=args.repeats,
        modes=modes,
        max_model_calls_per_run=args.max_model_calls_per_run,
        readiness=readiness,
        sleep_seconds=args.sleep,
    )
    report["manifest"]["model_status_after"] = get_model_status()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(sample["passed"] for sample in report["samples"]) else 1)


if __name__ == "__main__":
    main()
