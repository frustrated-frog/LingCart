"""验证 PG V4 状态条件检索；全程离线且不调用模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval/pg_state_condition_tasks.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pg_retrieval import retrieve_state_conditioned
from src.procedural_graph import ProceduralGraph, fingerprint


def evaluate(path: Path = DEFAULT_DATASET) -> dict:
    dataset = json.loads(path.read_text())
    if dataset.get("schema_version") != 1 or dataset.get("suite") != "pg_state_condition_contract":
        raise ValueError("状态条件评测协议无效")
    graph = ProceduralGraph.from_path(ROOT / dataset["graph"])
    rows = []
    for case in dataset.get("cases", []):
        result = retrieve_state_conditioned(graph, case["trajectory"], hops=2, max_edges=6)
        applicability = result["applicability"]
        selected = applicability["selected_edge_ids"]
        false = applicability["false_edge_ids"]
        unknown = applicability["unknown_edge_ids"]
        safe = (not (set(selected) & (set(false) | set(unknown)))
                and all(path["currently_applicable_prefix"] == 1 for path in result.get("paths") or []))
        passed = (result.get("anchor") == case["expected_anchor"]
                  and selected == case["expected_selected"]
                  and sorted(false) == sorted(case["expected_false"])
                  and sorted(unknown) == sorted(case["expected_unknown"])
                  and safe)
        rows.append({
            "case_id": case["id"], "anchor": result.get("anchor"), "selected_edge_ids": selected,
            "false_edge_ids": false, "unknown_edge_ids": unknown,
            "path_ids": [path["path_id"] for path in result.get("paths") or []],
            "safe": safe, "passed": passed,
        })
    result = {
        "schema_version": 1,
        "execution_source": "offline_deterministic",
        "model_calls": 0,
        "dataset_sha256": fingerprint(dataset),
        "graph_sha256": graph.fingerprint,
        "metrics": {
            "cases": len(rows),
            "contract_pass_rate": sum(row["passed"] for row in rows) / len(rows) if rows else 0,
            "unsafe_selected_edges": sum(not row["safe"] for row in rows),
            "unknown_evidence_abstentions": sum(bool(row["unknown_edge_ids"]) and not row["selected_edge_ids"] for row in rows),
            "model_calls": 0,
        },
        "cases": rows,
    }
    result["gate_passed"] = bool(rows) and all(row["passed"] for row in rows) and result["model_calls"] == 0
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
