"""从互斥 dev 训练/验证案例离线演化程序图；默认只验证配置，不调用模型。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.run_live_eval import evaluate, load_dataset
from src.harness import get_registry
from src.harness.profile import HarnessProfile
from src.pg_learning import EvolutionSpec, ModelGraphRefiner, evolve, write_artifact
from src.procedural_graph import ProceduralGraph, fingerprint

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "research/experiments/pg-localization-v2/evolution-spec.json"
DEFAULT_GRAPH = ROOT / "research/graphs/pg-seed-v1.json"
PROFILE_TEMPLATE = ROOT / "configs/harness/adaptive-pg-context-v2.json"


def graph_profile(graph, path, *, online_profile='adaptive-pg-context-v2'):
    if online_profile not in {'adaptive-pg-context-v2', 'adaptive-pg-context-v4'}:
        raise ValueError('未知在线 PG 配置')
    payload = json.loads((ROOT / f'configs/harness/{online_profile}.json').read_text())
    payload["profile_id"] = f"pg-{graph.fingerprint[:16]}-{fingerprint([str(path), online_profile])[:8]}"
    payload["display_name"] = f"PG 冻结图版本 {graph.version} 候选"
    for mechanism in payload["mechanisms"]:
        if mechanism["slot"] == "context":
            mechanism["configuration"]["graph_path"] = path.relative_to(ROOT).as_posix()
    return HarnessProfile.model_validate_json(json.dumps(payload))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--base-graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--live", action="store_true", help="显式调用真实模型并执行 dev 演化")
    args = parser.parse_args(argv)
    spec = EvolutionSpec.model_validate_json(args.spec.read_text())
    dataset_path = ROOT / spec.dataset
    dataset = load_dataset(dataset_path)
    spec.validate_dataset(dataset)
    graph = ProceduralGraph.from_path(args.base_graph)
    if not args.live:
        print(json.dumps({"status": "VALIDATED", "model_calls": 0, "graph_sha256": graph.fingerprint,
            "train_cases": spec.train_cases, "validation_cases": spec.validation_cases,
            "iterations": spec.iterations, "repeats": spec.repeats,
            "message": "配置与 dev 分组通过，尚未学习或测量模型收益；执行需显式 --live"}, ensure_ascii=False, indent=2))
        return 0
    if args.output_dir is None:
        parser.error("--live 必须提供新的 --output-dir（research/graphs 内）")
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT / "research/graphs").resolve()) or output.exists():
        parser.error("输出必须是 research/graphs 内尚不存在的目录，不可覆盖历史产物")
    registry = get_registry()
    profiles = {}

    def evaluator(current, cases, label):
        if current.fingerprint not in profiles:
            path = output / f"graph-{current.fingerprint}.json"
            write_artifact(path, current.model_dump(mode="json"))
            profile = graph_profile(current, path, online_profile=spec.online_profile)
            write_artifact(output / f"profile-{current.fingerprint}.json", profile.model_dump(mode="json"))
            registry.register_profile(profile)
            profiles[current.fingerprint] = profile
        return evaluate(dataset, repeats=spec.repeats, split="dev", case_ids=cases,
                        execution_mode="adaptive", harness_profile=profiles[current.fingerprint].profile_id,
                        dataset_path=dataset_path, max_model_calls_per_run=spec.max_model_calls_per_run,
                        max_prompt_tokens_per_run=spec.max_prompt_tokens_per_run,
                        max_completion_tokens_per_run=spec.max_completion_tokens_per_run,
                        max_seconds_per_run=spec.max_seconds_per_run,
                        max_provider_attempts_per_call=spec.max_provider_attempts_per_call)

    retained, report = evolve(graph, spec, evaluator=evaluator, refiner=ModelGraphRefiner(
        evolution_policy=spec.evolution_policy, max_prompt_tokens=spec.max_refiner_prompt_tokens,
        max_completion_tokens=spec.max_refiner_completion_tokens,
        max_provider_attempts=spec.max_provider_attempts_per_call), output_dir=output)
    write_artifact(output / "retained-profile.json", profiles[retained.fingerprint].model_dump(mode="json"))
    # 冻结后的 test 对照契约：区分 PG 指导效果与图演化效果。
    template = json.loads((ROOT / "research/experiments/pg-localization-v2/experiment-spec.json").read_text())
    for label, baseline in (("baseline", "adaptive-v1"), ("static", profiles[graph.fingerprint].profile_id)):
        contract = json.loads(json.dumps(template))
        contract.update(experiment_id=f"pg-evolved-vs-{label}", baseline_profile=baseline,
                        candidate_profile=profiles[retained.fingerprint].profile_id)
        contract["changes"][0]["paths"] += [
            (output / f"graph-{retained.fingerprint}.json").relative_to(ROOT).as_posix(),
            f"configs/harness/{profiles[retained.fingerprint].profile_id}.json"]
        write_artifact(output / f"test-vs-{label}-spec.json", contract)
    print(json.dumps({"status": report["status"], "report": str(output / "evolution-report.json"),
        "retained_graph": str(output / "retained.json"), "retained_sha256": retained.fingerprint,
        "accepted_iterations": sum(r["status"] == "ACCEPTED" for r in report["history"]),
        "message": "离线选择完成，保留图仍为 candidate；尚未进行 test 对照或产品晋升"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
