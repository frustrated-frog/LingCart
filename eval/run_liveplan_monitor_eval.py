#!/usr/bin/env python3
"""Replay the frozen LivePlan monitor benchmark without any model calls."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.harness import get_runner
from src.runtime import BudgetLimits, RunRuntime
from src.task_plan import TaskPlan, TaskStep


DATASET = ROOT / "eval/liveplan_monitor_cases.json"
KNOWN_SIGNALS = {
    "ACTION_OSCILLATION", "PHASE_STAGNATION", "EVIDENCE_STAGNATION",
    "VERIFICATION_LOOP", "BUDGET_RISK",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(path: Path = DATASET) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("dataset_id"), str):
        raise ValueError("LivePlan Monitor 数据集版本或标识无效")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("LivePlan Monitor 数据集必须包含案例")
    ids = [case.get("id") for case in cases]
    if any(not isinstance(case_id, str) or not case_id for case_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("LivePlan Monitor 案例编号必须存在且唯一")
    actual_counts = {split: sum(case.get("split") == split for case in cases) for split in ("dev", "test")}
    if payload.get("split_counts") != actual_counts or any(case.get("split") not in actual_counts for case in cases):
        raise ValueError("LivePlan Monitor split_counts 与案例不一致")
    for case in cases:
        frames = case.get("frames")
        expected = case.get("expected_signal_codes")
        if not isinstance(frames, list) or not 1 <= len(frames) <= 16:
            raise ValueError(f"{case['id']} 必须包含 1 到 16 个轨迹帧")
        if not isinstance(expected, list) or len(expected) != len(set(expected)) or not set(expected) <= KNOWN_SIGNALS:
            raise ValueError(f"{case['id']} 的预期漂移信号无效")
        for frame in frames:
            decision = frame.get("decision")
            if not isinstance(decision, dict) or not isinstance(decision.get("action"), str):
                raise ValueError(f"{case['id']} 包含无效决策帧")
            if decision["action"] == "tool" and (not decision.get("tool") or not isinstance(frame.get("result"), dict)):
                raise ValueError(f"{case['id']} 的工具帧缺少工具或结果")
    return payload


def _initial_state(case: dict) -> dict:
    status = case.get("initial_objective_status", "PENDING")
    if status not in {"PENDING", "VERIFIED"}:
        raise ValueError(f"{case['id']} 的初始目标状态无效")
    return {
        "goal_revision": 0,
        "request": case["description"],
        "goal": {"objectives": [{"id": "objective-1", "kind": "benchmark"}]},
        "objective_ledger": [{"objective_id": "objective-1", "status": status, "evidence_refs": []}],
        "binding_revision": 0,
        "model_cursor": 0,
    }


def run_case(case: dict, *, profile_id: str = "adaptive-liveplan-v1") -> dict:
    runner = get_runner(profile_id)
    mechanism = runner.mechanisms.get("steering")
    if mechanism is None or not hasattr(mechanism, "evaluate"):
        raise ValueError(f"Profile {profile_id} 没有可评测的 steering 机制")
    budget = {"max_model_calls": 8, "max_tool_calls": 8, **case.get("budget", {})}
    context = RunRuntime(f"monitor-{case['id'].lower()}", BudgetLimits(**budget))
    state = _initial_state(case)
    plan = TaskPlan(["benchmark"], [])
    predicted: set[str] = set()
    assessments = []

    for index, frame in enumerate(case["frames"], start=1):
        state.update(deepcopy(frame.get("state_patch", {})))
        decision = deepcopy(frame["decision"])
        step_id = f"step-{index}"
        context.record("adaptive_decision", source="benchmark", decision=decision,
                       goal_revision=state["goal_revision"], step_id=step_id)
        if decision["action"] == "tool":
            result = deepcopy(frame["result"])
            result["observation"] = {
                "reference": f"{case['id']}:{index}",
                "timestamp": f"2026-01-01T00:00:{index:02d}Z",
            }
            plan.steps.append(TaskStep(
                step_id, "benchmark", decision["tool"], deepcopy(decision.get("args", {})),
                status="SUCCEEDED", result=result,
            ))
            context.record("step_completed", step_id=step_id, specialist="benchmark", status="SUCCEEDED")
        if frame.get("verification_failure"):
            context.record("verification_failed", **deepcopy(frame["verification_failure"]))

        assessment, _ = mechanism.evaluate(context, state, plan)
        if assessment is not None:
            codes = [signal.code for signal in assessment.signals]
            predicted.update(codes)
            assessments.append({"frame": index, "signal_codes": codes,
                                "evaluation_id": assessment.evaluation_id})
            # The benchmark evaluates Monitor recall independently of Advisor behavior.
            # Mark the alert consumed so later frames can expose additional signal classes.
            state["liveplan"]["pending_drift"] = None

    expected = set(case["expected_signal_codes"])
    return {
        "case_id": case["id"],
        "split": case["split"],
        "description": case["description"],
        "expected_signal_codes": sorted(expected),
        "predicted_signal_codes": sorted(predicted),
        "true_positives": sorted(expected & predicted),
        "false_positives": sorted(predicted - expected),
        "false_negatives": sorted(expected - predicted),
        "exact_match": expected == predicted,
        "assessments": assessments,
        "model_calls": context.counters["model_calls"],
    }


def summarize(results: list[dict]) -> dict:
    true_positives = sum(len(item["true_positives"]) for item in results)
    false_positives = sum(len(item["false_positives"]) for item in results)
    false_negatives = sum(len(item["false_negatives"]) for item in results)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 1.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "cases": len(results),
        "exact_matches": sum(item["exact_match"] for item in results),
        "exact_match_rate": sum(item["exact_match"] for item in results) / len(results) if results else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_cases": [item["case_id"] for item in results if item["false_positives"]],
        "false_negative_cases": [item["case_id"] for item in results if item["false_negatives"]],
        "model_calls": sum(item["model_calls"] for item in results),
    }


def evaluate(path: Path = DATASET, *, split: str = "test",
             profile_id: str = "adaptive-liveplan-v1") -> dict:
    dataset = load_dataset(path)
    cases = [case for case in dataset["cases"] if split == "all" or case["split"] == split]
    results = [run_case(case, profile_id=profile_id) for case in cases]
    metrics = summarize(results)
    passed = bool(results) and metrics["exact_match_rate"] == 1 and metrics["model_calls"] == 0
    provenance = get_runner(profile_id).provenance()
    steering = next(item for item in provenance["mechanisms"] if item["slot"] == "steering")
    return {
        "schema_version": 1,
        "status": "PASSED" if passed else "FAILED",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "id": dataset["dataset_id"],
            "path": str(path.resolve().relative_to(ROOT)),
            "sha256": _sha256(path),
            "split": split,
        },
        "profile": {
            "profile_id": profile_id,
            "execution_sha256": provenance["execution_sha256"],
            "steering": steering,
        },
        "metrics": metrics,
        "cases": results,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="离线重放 LivePlan 确定性 Monitor 基准")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="test")
    parser.add_argument("--profile", default="adaptive-liveplan-v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args(argv)
    report = evaluate(args.dataset, split=args.split, profile_id=args.profile)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 1 if args.gate and report["status"] != "PASSED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
