"""Live evaluation for open-set semantics, polarity, coverage and evidence outcomes.

This suite intentionally uses the configured real model.  It is never imported
by pytest or the offline gate.  Use ``--dry-run`` to validate cases without a
provider call and ``--compare-adaptive`` to check Auto/Adaptive consistency.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(__file__).with_name("semantic_cases.json")
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


def _subset(expected, actual) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and _subset(value, actual[key])
                                                for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and all(value in actual for value in expected)
    return expected == actual


def _score(case: dict, result: dict, intake) -> dict:
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
    return {"passed": not failures, "failures": failures, "goal_recall": recall,
            "false_fast_path": false_fast_path, "objectives": kinds, "commands": command_kinds}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compare-adaptive", action="store_true")
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset = load_cases(args.dataset)
    digest = hashlib.sha256(json.dumps(dataset, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if args.dry_run:
        print(f"Semantic dataset OK: {len(dataset['cases'])} cases, sha256={digest}")
        return

    from src.adaptive import run_adaptive
    from src.intake import analyze_request
    from src.orchestrator import run_orchestrator

    samples = []
    for index, case in enumerate(dataset["cases"]):
        intake = analyze_request(case["message"])
        started = time.perf_counter()
        result = run_orchestrator(case["message"])
        score = _score(case, result, intake)
        adaptive_consistent = None
        if args.compare_adaptive:
            adaptive = run_adaptive(case["message"])
            adaptive_consistent = (adaptive.get("status") == result.get("status")
                                   and adaptive.get("terminal_reason") == result.get("terminal_reason"))
            if not adaptive_consistent:
                score["failures"].append("Auto/Adaptive terminal mismatch")
                score["passed"] = False
        sample = {"case_id": case["id"], **score, "status": result.get("status"),
                  "terminal_reason": result.get("terminal_reason"), "usage": result.get("usage"),
                  "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                  "adaptive_consistent": adaptive_consistent}
        samples.append(sample)
        print(f"{'PASS' if sample['passed'] else 'FAIL'} {case['id']}: {', '.join(sample['failures']) or 'ok'}")
        if index + 1 < len(dataset["cases"]):
            time.sleep(args.sleep)

    latencies = [sample["latency_ms"] for sample in samples]
    summary = {
        "dataset_sha256": digest,
        "cases": len(samples),
        "pass_rate": sum(sample["passed"] for sample in samples) / len(samples),
        "goal_recall": statistics.mean(sample["goal_recall"] for sample in samples),
        "false_fast_path_rate": sum(sample["false_fast_path"] for sample in samples) / len(samples),
        "auto_adaptive_consistency": (statistics.mean(sample["adaptive_consistent"] for sample in samples)
                                      if args.compare_adaptive else None),
        "latency_ms": {"mean": statistics.mean(latencies), "max": max(latencies)},
        "samples": samples,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if all(sample["passed"] for sample in samples) else 1)


if __name__ == "__main__":
    main()
