"""验证 PG V6 有界程序预演；全程离线且不调用模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval/pg_preplay_tasks.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pg_preplay import plan_preplay
from src.pg_retrieval import retrieve_state_conditioned
from src.procedural_graph import ProceduralGraph, fingerprint


def evaluate(path: Path = DEFAULT_DATASET) -> dict:
    dataset = json.loads(path.read_text())
    if dataset.get("schema_version") != 1 or dataset.get("suite") != "pg_preplay_contract":
        raise ValueError("程序预演评测协议无效")
    graph = ProceduralGraph.from_path(ROOT / dataset["graph"])
    rows = []
    for case in dataset.get("cases", []):
        selection = retrieve_state_conditioned(graph, case["trajectory"], hops=2, max_edges=6)
        result = plan_preplay(graph, selection, case["trajectory"], max_depth=3, beam_width=4, max_paths=4)
        selected = result.get("selected_path") or {}
        paths = result.get("paths") or []
        safe = (
            result.get("causal_claim") is False
            and result.get("probabilistic_claim") is False
            and all(path.get("executable_prefix") == 1 and len(path.get("edge_ids") or []) <= 3 for path in paths)
            and all(
                predictions and predictions[0].get("state_basis") == "CURRENT_VERIFIED"
                and predictions[0].get("condition_status") == "TRUE"
                and all(item.get("state_basis") == "PREDICTED_ACTION_ONLY" for item in predictions[1:])
                and not any(item.get("condition_status") == "FALSE" for item in predictions)
                for predictions in (path.get("predictions") or [] for path in paths)
            )
        )
        passed = (
            result.get("status") == case["expected_status"]
            and selected.get("first_action") == case["expected_first_action"]
            and list(selected.get("edge_ids") or []) == case["expected_edge_ids"]
            and safe
        )
        rows.append({
            "case_id": case["id"],
            "status": result.get("status"),
            "first_action": selected.get("first_action"),
            "edge_ids": list(selected.get("edge_ids") or []),
            "candidate_paths": len(paths),
            "safe": safe,
            "passed": passed,
        })
    metrics = {
        "cases": len(rows),
        "contract_pass_rate": sum(row["passed"] for row in rows) / len(rows) if rows else 0,
        "unsafe_predictions": sum(not row["safe"] for row in rows),
        "abstentions": sum(row["status"] == "ABSTAINED" for row in rows),
        "model_calls": 0,
    }
    result = {
        "schema_version": 1,
        "execution_source": "offline_deterministic",
        "model_calls": 0,
        "dataset_sha256": fingerprint(dataset),
        "graph_sha256": graph.fingerprint,
        "metrics": metrics,
        "cases": rows,
    }
    result["gate_passed"] = bool(rows) and all(row["passed"] for row in rows) and metrics["model_calls"] == 0
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
