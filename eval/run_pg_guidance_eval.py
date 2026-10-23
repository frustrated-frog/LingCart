"""验证 PG 确定性指导契约与提交前重放边界；全程离线且不调用模型。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "eval/pg_guidance_contract_tasks.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pg_guidance import compile_guidance_contract, decision_fingerprint, review_guidance_contract
from src.pg_retrieval import retrieve_state_aware
from src.procedural_graph import ProceduralGraph, fingerprint


def evaluate(path: Path = DEFAULT_DATASET) -> dict:
    dataset = json.loads(path.read_text())
    if dataset.get("schema_version") != 1 or dataset.get("suite") != "pg_guidance_contract":
        raise ValueError("指导契约评测协议无效")
    graph = ProceduralGraph.from_path(ROOT / dataset["graph"])
    rows = []
    for case in dataset.get("cases", []):
        trajectory = deepcopy(case["trajectory"])
        for action in trajectory.get("recent_actions") or []:
            if isinstance(action.get("decision"), dict):
                action["decision_fingerprint"] = decision_fingerprint(action.pop("decision"))
        selection = retrieve_state_aware(graph, trajectory)
        contract = compile_guidance_contract(graph, selection, trajectory)
        review = review_guidance_contract(contract, case["candidate"])
        required = set(case.get("required_recommended_actions") or [])
        recommendations = set(contract["recommended_actions"])
        passed = (review["status"] == case["expected_status"]
                  and review["blocked"] is case["expected_blocked"]
                  and required <= recommendations)
        rows.append({
            "case_id": case["id"],
            "localization_anchor": selection.get("anchor"),
            "contract_id": contract["contract_id"],
            "recommended_actions": contract["recommended_actions"],
            "forbidden_repeat_count": len(contract["forbidden_repeats"]),
            "review_status": review["status"],
            "blocked": review["blocked"],
            "passed": passed,
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
            "blocked_exact_repeats": sum(row["blocked"] for row in rows),
            "guidance_model_calls": 0,
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
