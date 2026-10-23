"""比较最近动作定位与轨迹状态定位；全程离线且不调用模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval/pg_localization_tasks.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pg_retrieval import retrieve_state_aware, shadow_action_anchor
from src.procedural_graph import ProceduralGraph, fingerprint


def evaluate(path: Path = DEFAULT_DATASET) -> dict:
    dataset = json.loads(path.read_text())
    if dataset.get("schema_version") != 1 or dataset.get("suite") != "pg_localization_contract":
        raise ValueError("定位评测协议无效")
    graph_path = ROOT / dataset["graph"]
    graph = ProceduralGraph.from_path(graph_path)
    rows = []
    for case in dataset.get("cases", []):
        expected = case.get("expected_anchor")
        required = set(case.get("required_edge_ids") or [])
        trajectory = case["trajectory"]
        baseline = shadow_action_anchor(graph, trajectory)
        candidate = retrieve_state_aware(graph, trajectory)
        retrieved = {edge["id"] for edge in candidate["edges"]}
        rows.append({
            "case_id": case["id"],
            "expected_anchor": expected,
            "baseline_anchor": baseline["anchor"],
            "baseline_fallback": baseline["fallback"],
            "candidate_anchor": candidate["anchor"],
            "candidate_status": candidate["status"],
            "candidate_confidence": candidate["confidence"],
            "candidate_edge_ids": sorted(retrieved),
            "anchor_passed": candidate["anchor"] == expected,
            "edge_passed": required <= retrieved,
            "bounded": len(candidate["edges"]) <= 6,
        })
    total = len(rows)
    passed = sum(row["anchor_passed"] and row["edge_passed"] and row["bounded"] for row in rows)
    baseline_correct = sum(row["baseline_anchor"] == row["expected_anchor"] for row in rows)
    result = {
        "schema_version": 1,
        "execution_source": "offline_deterministic",
        "model_calls": 0,
        "dataset_sha256": fingerprint(dataset),
        "graph_sha256": graph.fingerprint,
        "metrics": {
            "cases": total,
            "candidate_contract_pass_rate": passed / total if total else 0,
            "baseline_anchor_accuracy": baseline_correct / total if total else 0,
            "candidate_anchor_accuracy": sum(row["anchor_passed"] for row in rows) / total if total else 0,
            "baseline_full_graph_fallbacks": sum(row["baseline_fallback"] for row in rows),
            "candidate_full_graph_fallbacks": 0,
            "candidate_abstentions": sum(row["candidate_status"] == "ABSTAINED" for row in rows),
        },
        "cases": rows,
    }
    result["gate_passed"] = passed == total and result["metrics"]["candidate_full_graph_fallbacks"] == 0
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args(argv)
    result = evaluate(args.dataset.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.gate and not result["gate_passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
