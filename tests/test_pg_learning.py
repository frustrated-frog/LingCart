from copy import deepcopy
import json
from pathlib import Path

from langchain_core.messages import AIMessage
import pytest

from src.pg_learning import EvolutionSpec, evolve, validate_report
from src.procedural_graph import canonical
from tests.test_procedural_graph import graph


def spec(**updates):
    return EvolutionSpec(experiment_id="pg-test", dataset="eval/adaptive_tasks.json",
                         train_cases=("A01",), validation_cases=("A02",), repeats=1,
                         **{"iterations": 1, **updates})


def proposal(guidance="空结果时更换检索条件，尝试扩大检索范围"):
    return dict(reason="失败轨迹在局部空结果后过早结束", source_cases=["A01"],
                add_edges=[dict(id="search-retry", source="search_products", target="search_products",
                    relation="TRIGGERS", condition="上一次检索为空且还可查证", guidance=guidance,
                    pitfalls="不能把一次空查询当成全目录没有候选")])


class Refiner:
    def __init__(self, proposals):
        self.proposals = iter(proposals)
        self.memories = []

    def propose(self, current, report, rejected):
        self.memories.append(deepcopy(rejected))
        return {"proposal": canonical(next(self.proposals)), "error": None, "events": []}


def report(current, cases, *, passed=True, calls=4, events=None):
    return {"status": "COMPLETED", "execution_source": "test_harness", "providers": ["scripted"],
        "manifest": {"suite": "adaptive", "split": "dev", "repeats": 1,
            **{k: "fixed" for k in ("dataset_sha256", "evaluation_protocol_sha256", "business_protocol_sha256",
                "python_runtime", "dependency_lock_sha256", "limits", "purpose_models", "source_tree_sha256")},
            "harness_profile": {"mechanisms": [{"id": "procedural-graph-context",
                "configuration": {"graph_sha256": current.fingerprint}}]}},
        "samples": [{"case_id": c, "repetition": 1, "split": "dev", "passed": passed,
            "model_attempts": calls, "safety_violations": 0, "protocol_violations": 0,
            "turns": [{"message": "推荐预算500的耳机", "failures": [] if passed else ["过早澄清"],
                "result": {"events": events if events is not None else [{"type": "mechanism_applied",
                    "mechanism_id": "procedural-graph-context", "graph_sha256": current.fingerprint}]}}]} for c in cases]}


def test_evolution_retains_rejections_then_accepts_and_never_uses_test(tmp_path):
    refiner = Refiner([proposal("继续原样重复检索"), proposal()])
    seen = []
    def evaluator(current, cases, label):
        seen.append((current.fingerprint, cases, label))
        return report(current, cases, passed=not any(e.guidance == "继续原样重复检索" for e in current.edges))
    retained, result = evolve(graph(), spec(iterations=2), evaluator=evaluator, refiner=refiner,
        output_dir=tmp_path / "learning", execution_source="test_harness")
    assert [r["status"] for r in result["history"]] == ["REJECTED", "ACCEPTED"]
    assert refiner.memories[1][0]["reason"] == "VALIDATION_REGRESSION"
    assert retained.parent_sha256 == graph().fingerprint and retained.version == 1
    assert seen[3][0] == graph().fingerprint  # 第二轮训练仍从保留图开始。
    assert json.loads((tmp_path / "learning/iteration-01-candidate.json").read_text())
    assert result["test_evaluated"] is False and result["promotion_status"] == "candidate"
    with pytest.raises(FileExistsError):
        evolve(graph(), spec(), evaluator=evaluator, refiner=refiner, output_dir=tmp_path / "learning")


def test_repeated_rejected_structure_skips_rollout(tmp_path):
    labels = []
    def evaluator(current, cases, label):
        labels.append(label)
        return report(current, cases, passed=current.version == 0)
    _, result = evolve(graph(), spec(iterations=2), evaluator=evaluator,
        refiner=Refiner([proposal(), proposal()]), output_dir=tmp_path / "learning", execution_source="test_harness")
    assert result["history"][1]["reason"] == "REPEATED_REJECTED_STRUCTURE"
    assert "iteration-02-validation" not in labels


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "test", "fingerprint", "source"])
def test_learning_refuses_incomplete_mixed_or_mismatched_reports(mutation):
    data = report(graph(), ("A01",))
    if mutation == "missing": data["samples"] = []
    if mutation == "duplicate": data["samples"] *= 2
    if mutation == "test": data["samples"][0]["split"] = "test"
    if mutation == "fingerprint": data["manifest"]["harness_profile"]["mechanisms"] = []
    if mutation == "source": data["execution_source"] = "live_provider"
    with pytest.raises(ValueError):
        validate_report(data, graph=graph(), cases=("A01",), repeats=1, execution_source="test_harness")


def test_split_and_dataset_frozen_before_learning():
    with pytest.raises(ValueError, match="隔离"):
        EvolutionSpec(experiment_id="pg-test", dataset="eval/adaptive_tasks.json", train_cases=("A01",), validation_cases=("A01",))
    with pytest.raises(ValueError, match="test"):
        spec().validate_dataset({"suite": "adaptive", "cases": [{"id": "A01", "split": "dev"}, {"id": "A02", "split": "test"}]})


@pytest.mark.parametrize("kind,reason", [("cost", "COST_REGRESSION"), ("unexposed", "PG_NOT_EXPOSED"),
                                         ("model", "INCOMPARABLE_VALIDATION"), ("safety", "SAFETY_OR_PROTOCOL_VIOLATION")])
def test_validation_guards_are_independent_of_success(tmp_path, kind, reason):
    def evaluator(current, cases, label):
        data = report(current, cases)
        if current.version:
            if kind == "cost": data["samples"][0]["model_attempts"] += 3
            if kind == "unexposed": data["samples"][0]["turns"] = []
            if kind == "model": data["providers"] = ["different"]
            if kind == "safety": data["samples"][0]["safety_violations"] = 1
        return data
    retained, result = evolve(graph(), spec(), evaluator=evaluator, refiner=Refiner([proposal()]),
                             output_dir=tmp_path / "learning", execution_source="test_harness")
    assert retained == graph() and result["history"][0]["reason"] == reason


def test_invalid_proposal_is_saved_and_never_rolls_out(tmp_path):
    bad = proposal()
    bad["source_cases"] = ["A02"]
    labels = []
    def evaluator(current, cases, label):
        labels.append(label)
        return report(current, cases)
    _, result = evolve(graph(), spec(), evaluator=evaluator, refiner=Refiner([bad]),
                      output_dir=tmp_path / "learning", execution_source="test_harness")
    assert result["history"][0]["reason"] == "INVALID_PROPOSAL_OR_REPORT"
    assert labels == ["seed-validation", "iteration-01-training"]
    assert (tmp_path / "learning/iteration-01-proposal.json").exists()


def test_evolved_graph_changes_future_agent_actions_with_real_tools(tmp_path, monkeypatch):
    """完整 Refiner→验证→保留→新 Run；脚本模型只验证机制，不声称模型收益。"""
    from src import tools
    from src.backend.db import get_engine, get_session_factory
    from src.backend.seed import seed
    from src.adaptive import run_adaptive
    from src.context_engine import ContextEngine
    from src.pg_context import PGContextEngine
    from src.runtime import RunRuntime, BudgetLimits, runtime_scope
    from tests.test_adaptive import model, goal, search, recommendation, clarification

    engine = get_engine(f"sqlite:///{tmp_path / 'pg.db'}")
    tools.configure_db(engine)
    with get_session_factory(engine)() as session:
        seed(session)

    class GuidanceModel:
        def invoke(self, messages):
            payload = json.loads(messages[-1].content)
            edges = payload["subgraph"]["edges"]
            selected = next((e for e in edges if e["id"] == "search-retry"), edges[0])
            return AIMessage(content=canonical({"guidance": selected["guidance"], "edge_ids": [selected["id"]]}))

    class Solver:
        def invoke(self, messages):
            payload = json.loads(messages[-1].content)
            if payload["goal"] is None:
                answer = {"goal": goal()}
            elif not payload["observations"]:
                answer = search(query="完全不存在的型号")
            elif not payload["observations"][-1]["result"].get("products"):
                answer = search() if "尝试扩大检索范围" in str(messages) else clarification()
            else:
                answer = recommendation(payload)
            return AIMessage(content=canonical(answer))

    # 复用原 Agent 的独立核验替身；工具和机械约束核验仍真实执行。
    model(monkeypatch)
    monkeypatch.setattr("src.adaptive_decision.get_llm", lambda **_: Solver())
    monkeypatch.setattr("src.pg_context.get_llm", lambda **_: GuidanceModel())
    runs = []
    def execute(current):
        context = RunRuntime(f"pg-run-{len(runs)}", BudgetLimits.adaptive())
        context.context_engine = PGContextEngine(ContextEngine("推荐预算500的耳机"), current)
        with runtime_scope(context):
            result = run_adaptive("推荐预算500的耳机")
        runs.append(result)
        return result
    def evaluator(current, cases, label):
        result = execute(current)
        return report(current, cases, passed=result["status"] == "SUCCEEDED",
                      calls=result["budget"]["consumed"]["model_calls"], events=result["events"])
    retained, result = evolve(graph(), spec(max_extra_model_attempts_per_sample=4), evaluator=evaluator,
        refiner=Refiner([proposal()]), output_dir=tmp_path / "learning", execution_source="test_harness")
    assert result["history"][0]["status"] == "ACCEPTED", result
    assert runs[0]["status"] == "WAITING_CLARIFICATION"
    future = execute(retained)
    assert future["status"] == "SUCCEEDED", future
    searches = [e["decision"] for e in future["events"] if e["type"] == "adaptive_decision"
                and e["decision"]["tool"] == "search_products"]
    assert len(searches) == 2 and searches[-1]["args"]["query"] == ""
    assert any(e["type"] == "pg_guidance" and "search-retry" in e["edge_ids"] for e in future["events"])
    engine.dispose()
