"""Read-only engineering projections shared by the API and web console.

Every value is derived from the runtime registry, durable run archive, model
gateway, versioned evaluation datasets, or experiment contracts.  This module
contains no frontend framework dependency so the React client receives the
same facts used by tests and evaluation gates.
"""
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from src.harness import get_registry, get_runner
from src.model_gateway import get_model_status
from src.workspace import list_runs


ROOT = Path(__file__).resolve().parents[1]
SLOT_ORDER = ("planning", "context", "memory", "verification", "tool_protocol")
SLOT_LABELS = {
    "planning": "规划",
    "context": "上下文",
    "memory": "记忆",
    "verification": "核验",
    "tool_protocol": "工具协议",
}
RESEARCH_STATUS_LABELS = {"baseline": "基线", "candidate": "候选", "promoted": "已晋升"}
CATEGORY_LABELS = {
    "order": "订单查询", "refund": "退款处理", "payment": "支付处理",
    "payment_approval": "支付审批", "refund_approval": "退款审批", "shopping": "商品推荐",
    "shopping_validation": "选购核验", "faq": "规则问答", "human": "人工转交",
    "compound": "复合任务", "context": "上下文连续性", "clarification": "澄清",
    "rule_rejection": "规则拒绝", "data_absence": "数据缺失", "model_resilience": "模型韧性",
    "unsupported": "不支持请求", "soft_preferences": "软偏好", "negation": "否定约束",
    "negation_scope": "否定范围", "coverage": "目标覆盖", "multiple_orders": "多订单",
    "policy_evidence": "规则证据", "unknown_policy": "未知规则", "selection": "候选选择",
    "no_match": "无匹配结果", "long_history": "长历史", "price_change": "价格变化",
    "stock_change": "库存变化", "steering": "运行时调整", "combined_constraints": "组合约束",
    "multiple_products": "多商品", "two_categories": "跨品类", "conflict": "约束冲突",
    "numeric_constraint": "数值约束", "multi_turn": "多轮任务",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _current_experiment_identity(spec: dict) -> dict:
    """Build the identities that must still match before a report is called current."""

    from eval.run_live_eval import environment_manifest
    from dotenv import load_dotenv
    from src.model_gateway import ModelGatewayConfigurationError, load_gateway_config

    load_dotenv(ROOT / ".env")
    dataset_path = ROOT / spec["dataset"]
    dataset = _read_json(dataset_path)
    runner = get_runner(spec["candidate_profile"])
    candidate = runner.provenance()
    environment = environment_manifest(dataset, runner.execution_mode, dataset_path)
    purpose_models = {}
    for purpose in ("dialogue_understanding", "goal", "planner", "agent", "final_response"):
        try:
            config = load_gateway_config(purpose=purpose)
            purpose_models[purpose] = [{
                "provider": provider.provider,
                "model": provider.model,
                "thinking_mode": provider.thinking_mode,
                "reasoning_effort": provider.reasoning_effort,
                "strict_tools": provider.strict_tools,
            } for provider in config.providers]
        except ModelGatewayConfigurationError:
            purpose_models[purpose] = None
    return {
        "experiment_spec": spec,
        "dataset_path": spec["dataset"],
        "dataset_sha256": _fingerprint(dataset),
        "suite": spec["suite"],
        "split": spec["evaluation_split"],
        "repeats": spec["repeats"],
        "harness_profile": candidate,
        "source_tree_sha256": environment["source_tree_sha256"],
        "evaluation_protocol_sha256": environment["evaluation_protocol_sha256"],
        "business_protocol_sha256": environment["business_protocol_sha256"],
        "dependency_lock_sha256": environment["dependency_lock_sha256"],
        "purpose_models": purpose_models,
    }


def _report_mismatches(report: dict, current: dict) -> list[str]:
    manifest = report.get("manifest") or {}
    mismatches = []
    for key in ("experiment_spec", "dataset_path", "dataset_sha256", "suite", "split", "repeats",
                "source_tree_sha256", "evaluation_protocol_sha256", "business_protocol_sha256",
                "dependency_lock_sha256", "purpose_models"):
        if manifest.get(key) != current.get(key):
            mismatches.append(key)
    recorded_profile = manifest.get("harness_profile") or {}
    current_profile = current.get("harness_profile") or {}
    for key in ("profile_id", "profile_sha256", "execution_sha256"):
        if recorded_profile.get(key) != current_profile.get(key):
            mismatches.append(f"harness_profile.{key}")
    return mismatches


def profile_catalog() -> list[dict]:
    profiles = []
    for profile in get_registry().profiles():
        provenance = get_runner(profile.profile_id).provenance()
        profiles.append({
            "profile_id": profile.profile_id,
            "version": profile.version,
            "display_name": profile.display_name,
            "description": profile.description,
            "engine": profile.engine.model_dump(),
            "mechanisms": {item.slot: item.model_dump() for item in profile.mechanisms},
            "capabilities": list(profile.capabilities),
            "research": profile.research.model_dump(),
            "fingerprint": profile.fingerprint,
            "execution_sha256": provenance["execution_sha256"],
            "source_revision": provenance.get("source_revision"),
            "resolved_mechanisms": {item["slot"]: item for item in provenance["mechanisms"]},
            "product_eligible": profile.research.status in {"baseline", "promoted"},
        })
    return profiles


def dataset_catalog(*, include_cases: bool = True) -> list[dict]:
    definitions = (
        ("offline-core-v1", "固定离线回归集", "eval/task_cases.json", "support", "scripted",
         "60 项独立客服与故障注入任务；每项任务使用独立种子数据，多轮审批流程按一项任务统计。"),
        ("live-support-v1", "真实模型消费任务集", "eval/live_tasks.json", "support", "live",
         "使用真实模型执行的消费服务任务，开发集用于调试，测试集用于最终结论。"),
        ("adaptive-agent-v1", "自主 Agent 专项集", "eval/adaptive_tasks.json", "adaptive", "live",
         "覆盖自主规划、证据读取、目标核验、重规划与运行时调整的专项任务。"),
        ("pg-challenges-v1", "PG 程序图挑战集", "eval/pg_challenge_tasks.json", "adaptive", "live",
         "18 个独立情境族；训练、验证、测试各 6 项，覆盖恢复、依赖、多目标与环境变化。"),
    )
    datasets = []
    for dataset_id, display_name, relative_path, suite, execution_source, description in definitions:
        payload = _read_json(ROOT / relative_path)
        cases = payload.get("cases", [])
        categories = Counter(case.get("category", "unclassified") for case in cases)
        splits = Counter(case.get("split", "fixed") for case in cases)
        item = {
            "dataset_id": dataset_id,
            "display_name": display_name,
            "path": relative_path,
            "suite": payload.get("suite", suite),
            "execution_source": execution_source,
            "description": description,
            "case_count": len(cases),
            "turn_count": sum(len(case.get("turns", [])) for case in cases),
            "categories": dict(sorted(categories.items())),
            "splits": dict(sorted(splits.items())),
        }
        if include_cases:
            item["cases"] = cases
        datasets.append(item)
    return datasets


def experiment_catalog(*, include_report: bool = True) -> list[dict]:
    reports = []
    output = ROOT / "output"
    for path in sorted(output.rglob("*.json")) if output.exists() else []:
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        contract = (payload.get("manifest") or {}).get("experiment_spec") or {}
        if contract.get("experiment_id"):
            reports.append((path, payload, contract["experiment_id"]))
    reports.sort(key=lambda item: (item[1].get("generated_at", ""), str(item[0])))

    experiments = []
    for path in sorted(ROOT.glob("research/experiments/*/experiment-spec.json")):
        payload = _read_json(path)
        matches = [(report_path, report) for report_path, report, experiment_id in reports
                   if experiment_id == payload["experiment_id"]]
        completed = [(report_path, report) for report_path, report in matches
                     if report.get("status") == "COMPLETED"
                     and (report.get("gate") or {}).get("passed") in {True, False}]
        current_identity = _current_experiment_identity(payload)
        current = [(report_path, report) for report_path, report in completed
                   if not _report_mismatches(report, current_identity)]
        latest = current[-1] if current else (completed[-1] if completed else None)
        mismatches = _report_mismatches(latest[1], current_identity) if latest and not current else []
        passed = (latest[1].get("gate") or {}).get("passed") if latest else None
        if current:
            status = "CURRENT_REPRODUCIBLE" if passed else "CURRENT_REJECTED"
        elif latest:
            status = "HISTORICAL_PASSED" if passed else "HISTORICAL_REJECTED"
        else:
            status = "AWAITING_REPORT"
        item = {
            **payload,
            "path": path.relative_to(ROOT).as_posix(),
            "status": status,
            "report_path": latest[0].relative_to(ROOT).as_posix() if latest else None,
            "report_mismatches": mismatches,
        }
        if include_report:
            item["report"] = latest[1] if latest else None
        experiments.append(item)
    return experiments


def run_profile(result: dict) -> dict:
    harness = result.get("harness")
    if isinstance(harness, dict) and harness.get("profile_id"):
        return harness
    for event in reversed(result.get("events", [])):
        if event.get("type") == "harness_profile_selected" and event.get("profile_id"):
            return event
    return {"profile_id": "legacy/unrecorded", "legacy_profile_reference": True}


def run_catalog(limit: int = 100, *, include_result: bool = True) -> list[dict]:
    runs = []
    for run in list_runs(limit=limit):
        result = run["result"]
        events = result.get("events", [])
        item = {
            "run_id": run["run_id"],
            "conversation_id": run["conversation_id"],
            "request": run["request"],
            "created_at": run["created_at"],
            "status": result.get("status", "UNKNOWN"),
            "agent": result.get("agent", "runtime"),
            "terminal_reason": result.get("terminal_reason"),
            "profile": run_profile(result),
            "event_count": len(events),
            "tool_calls": sum(event.get("type") == "tool_call" for event in events),
            "model_calls": sum(event.get("type") == "model_call" for event in events),
            "latency_ms": result.get("latency_ms"),
            "tokens": (result.get("usage") or {}).get("total_tokens"),
        }
        if include_result:
            item["result"] = result
        runs.append(item)
    return runs


def run_detail(run_id: str) -> dict | None:
    return next((run for run in run_catalog(limit=500, include_result=True) if run["run_id"] == run_id), None)


def stable_kernel_boundaries() -> list[dict]:
    from eval.run_live_eval import KERNEL_BOUNDARIES

    labels = {
        "state_and_side_effects": "状态与副作用",
        "persistence_and_recovery": "持久化与恢复",
        "model_call_protocol": "模型调用协议",
        "harness_assembly": "Harness 装配",
    }
    return [{"boundary_id": key, "label": labels.get(key, key), "paths": list(paths)}
            for key, paths in KERNEL_BOUNDARIES.items()]


def model_gateway_status() -> dict:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        return get_model_status()
    except Exception as exc:
        return {"ready": False, "configured": False, "live_verified": False,
                "message": str(exc), "providers": []}


def engineering_overview() -> dict:
    profiles = profile_catalog()
    datasets = dataset_catalog(include_cases=False)
    experiments = experiment_catalog(include_report=False)
    runs = run_catalog(include_result=False)
    status_counts = Counter(run["status"] for run in runs)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "profiles": profiles,
        "datasets": datasets,
        "experiments": experiments,
        "runs": runs,
        "kernel": stable_kernel_boundaries(),
        "model": model_gateway_status(),
        "counts": {
            "profiles": len(profiles),
            "candidate_profiles": sum(item["research"]["status"] == "candidate" for item in profiles),
            "dataset_cases": sum(item["case_count"] for item in datasets),
            "dataset_turns": sum(item["turn_count"] for item in datasets),
            "runs": len(runs),
            "profiled_runs": sum(run["profile"]["profile_id"] != "legacy/unrecorded" for run in runs),
            "events": sum(run["event_count"] for run in runs),
            "waiting_runs": sum(count for status, count in status_counts.items()
                                if status.startswith("WAITING") or status == "INTERRUPTED"),
        },
        "status_counts": dict(status_counts),
    }
