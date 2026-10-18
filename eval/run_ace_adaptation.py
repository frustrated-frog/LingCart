"""Build one immutable ACE playbook from a frozen adaptive dev report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from src.ace_context import ExperiencePlaybook
    from src.ace_learning import (
        AceLearner,
        Reflection,
        ModelCurator,
        ModelReflector,
        episodes_from_report,
        evolve_playbook,
        source_for_report,
        validate_curator_operations,
    )
except ModuleNotFoundError:  # pragma: no cover - direct invocation outside repository root
    raise SystemExit("请从仓库根目录运行：venv/bin/python -m eval.run_ace_adaptation")


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "research" / "playbooks" / "ace-empty-v1.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--base-playbook", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-playbook", type=Path)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--replay-proposals", type=Path,
                        help="复用既有反思和 Curator 提案，以当前代码门禁重新生成，不调用模型")
    parser.add_argument("--dry-run", action="store_true", help="只验证来源与 schema，不调用模型")
    args = parser.parse_args(argv)
    report = json.loads(args.input_report.read_text())
    episodes = episodes_from_report(report)
    base = ExperiencePlaybook.from_path(args.base_playbook)
    if args.dry_run:
        result = {
            "status": "VALIDATED",
            "input_report": str(args.input_report),
            "input_samples": len(episodes),
            "input_cases": len({episode.case_id for episode in episodes}),
            "base_playbook_sha256": base.fingerprint,
            "base_bullets": len(base.bullets),
            "message": "ACE dev 来源与 Playbook schema 校验通过，未调用模型",
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if not args.output_playbook or not args.output_report:
        parser.error("真实适配必须同时提供 --output-playbook 与 --output-report")
    for path in (args.output_playbook, args.output_report):
        if path.exists():
            parser.error(f"不可覆盖不可变 ACE 产物：{path}")
    source = source_for_report(args.input_report, report)
    if args.replay_proposals:
        prior = json.loads(args.replay_proposals.read_text())
        reflections = tuple(Reflection.model_validate_json(json.dumps(item, ensure_ascii=False))
                            for item in prior.get("reflections", []))
        expected = {f"reflection-{episode.identity}" for episode in episodes}
        if {item.reflection_id for item in reflections} != expected:
            parser.error("重放报告的反思集合与输入 dev 报告不一致")
        operations, rejected = validate_curator_operations(prior.get("proposed_operations", []))
        playbook, applied, evolve_rejected = evolve_playbook(
            base, reflections, operations, source=source)
        learned = type("ReplayedLearning", (), {
            "playbook": playbook,
            "report": {
                "schema_version": 1,
                "method": "ace_offline",
                "paper_id": source.paper_id,
                "replayed_from": str(args.replay_proposals),
                "base_playbook_sha256": base.fingerprint,
                "playbook_sha256": playbook.fingerprint,
                "episodes": [episode.model_dump(mode="json") for episode in episodes],
                "reflections": [item.model_dump(mode="json") for item in reflections],
                "proposed_operations": prior.get("proposed_operations", []),
                "applied_operations": applied,
                "rejected_operations": [*rejected, *evolve_rejected],
                "model_calls": prior.get("model_calls", []),
            },
        })()
    else:
        learned = AceLearner(ModelReflector(), ModelCurator()).learn(episodes, base, source=source)
    args.output_playbook.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_playbook.write_text(json.dumps(
        learned.playbook.canonical_payload(), ensure_ascii=False, indent=2) + "\n")
    args.output_report.write_text(json.dumps(learned.report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": "COMPLETED",
        "playbook_id": learned.playbook.playbook_id,
        "version": learned.playbook.version,
        "playbook_sha256": learned.playbook.fingerprint,
        "bullets": len(learned.playbook.bullets),
        "applied_operations": len(learned.report["applied_operations"]),
        "rejected_operations": len(learned.report["rejected_operations"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

