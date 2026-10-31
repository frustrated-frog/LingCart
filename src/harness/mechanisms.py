"""Executable Harness mechanisms and their source-bound identities."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
from dataclasses import replace
from typing import Protocol, runtime_checkable


ROOT = Path(__file__).resolve().parents[2]


def source_fingerprint(paths: tuple[str, ...]) -> str:
    payload = {}
    for name in sorted(paths):
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f"机制源码不存在：{name}")
        payload[name] = path.read_text()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


@runtime_checkable
class HarnessMechanism(Protocol):
    slot: str
    mechanism_id: str
    mechanism_version: str
    compatible_engines: tuple[str, ...]
    source_files: tuple[str, ...]

    def install(self, context) -> None:
        """Bind the implementation to one RunRuntime."""

    def configure(self, configuration: dict[str, str | int | bool]) -> "HarnessMechanism":
        """Return an immutable profile-bound mechanism."""


@dataclass(frozen=True)
class EmbeddedMechanism:
    """A mechanism implemented inside an existing engine but still source-bound."""

    slot: str
    mechanism_id: str
    mechanism_version: str
    compatible_engines: tuple[str, ...]
    source_files: tuple[str, ...]

    def install(self, context) -> None:
        context.mechanisms[self.slot] = self

    def configure(self, configuration: dict[str, str | int | bool]):
        if configuration:
            raise ValueError(f"机制 {self.mechanism_id} 不接受配置")
        return self

    def provenance(self) -> dict:
        return {
            "slot": self.slot,
            "id": self.mechanism_id,
            "version": self.mechanism_version,
            "implementation_sha256": source_fingerprint(self.source_files),
            "source_files": list(self.source_files),
        }


@dataclass(frozen=True)
class EvidenceVerificationMechanism(EmbeddedMechanism):
    """保留既有目标审查，同时提供可替换的证据语义核验槽位。"""

    def assess_evidence(self, inputs, *, model, runtime):
        from src.after_sales.semantic import assess
        return assess(inputs, model=model, runtime=runtime)


@dataclass(frozen=True)
class AfterSalesPlanningMechanism(EmbeddedMechanism):
    """研究对照用的售后计划消费策略，不改变 Stable Kernel 执行边界。"""

    case_execution_mode: str = "policy_aware"
    evidence_intake_mode: str = "model_default"
    action_selection_mode: str = "priority_v2"

    def install(self, context) -> None:
        super().install(context)
        context.record(
            "mechanism_applied",
            slot=self.slot,
            mechanism_id=self.mechanism_id,
            mechanism_version=self.mechanism_version,
            configuration={
                "case_execution_mode": self.case_execution_mode,
                "evidence_intake_mode": self.evidence_intake_mode,
                "action_selection_mode": self.action_selection_mode,
            },
        )

    def provenance(self) -> dict:
        return {
            **super().provenance(),
            "configuration": {
                "case_execution_mode": self.case_execution_mode,
                "evidence_intake_mode": self.evidence_intake_mode,
                "action_selection_mode": self.action_selection_mode,
            },
        }


@dataclass(frozen=True)
class ContextMechanism(EmbeddedMechanism):
    """A real context seam used by durable execution before the engine starts."""

    selection_strategy: str = "recent-first"

    def build_context(self, goal, *, preferences=None, observations=None, revision=0):
        from src.context_engine import ContextEngine
        from src.runtime import current_runtime

        engine = ContextEngine(
            goal,
            preferences=preferences,
            observations=observations,
            revision=revision,
            selection_strategy=self.selection_strategy,
            mechanism_id=self.mechanism_id,
        )
        runtime = current_runtime()
        if runtime is not None:
            runtime.record(
                "mechanism_applied",
                slot=self.slot,
                mechanism_id=self.mechanism_id,
                mechanism_version=self.mechanism_version,
                configuration={"selection_strategy": self.selection_strategy},
                goal_revision=revision,
            )
        return engine


@dataclass(frozen=True)
class AceContextMechanism(EmbeddedMechanism):
    """ACE playbook injection bound to an immutable research artifact."""

    playbook_path: str | None = None
    playbook_sha256: str | None = None
    max_bullets: int = 16
    max_chars: int = 12000

    def configure(self, configuration: dict[str, str | int | bool]):
        expected = {"playbook_path", "max_bullets", "max_chars"}
        if set(configuration) != expected:
            raise ValueError("ACE context 必须配置 playbook_path、max_bullets 与 max_chars")
        path_value = configuration["playbook_path"]
        max_bullets, max_chars = configuration["max_bullets"], configuration["max_chars"]
        if not isinstance(path_value, str) or type(max_bullets) is not int or type(max_chars) is not int:
            raise ValueError("ACE context 配置类型无效")
        relative = PurePosixPath(path_value)
        if (relative.is_absolute() or ".." in relative.parts or relative.suffix != ".json"
                or relative.parts[:2] != ("research", "playbooks")):
            raise ValueError("ACE Playbook 必须是 research/playbooks 下的 JSON 文件")
        if not 1 <= max_bullets <= 64 or not 256 <= max_chars <= 32000:
            raise ValueError("ACE context 预算无效")
        from src.ace_context import ExperiencePlaybook

        path = ROOT / relative
        playbook = ExperiencePlaybook.from_path(path)
        return replace(self, playbook_path=str(relative), playbook_sha256=playbook.fingerprint,
                       max_bullets=max_bullets, max_chars=max_chars)

    @property
    def effective_source_files(self) -> tuple[str, ...]:
        return self.source_files + ((self.playbook_path,) if self.playbook_path else ())

    def provenance(self) -> dict:
        if not self.playbook_path or not self.playbook_sha256:
            raise ValueError("ACE context 尚未绑定 Playbook")
        return {
            "slot": self.slot,
            "id": self.mechanism_id,
            "version": self.mechanism_version,
            "implementation_sha256": source_fingerprint(self.effective_source_files),
            "source_files": list(self.effective_source_files),
            "configuration": {
                "playbook_path": self.playbook_path,
                "playbook_sha256": self.playbook_sha256,
                "max_bullets": self.max_bullets,
                "max_chars": self.max_chars,
            },
        }

    def build_context(self, goal, *, preferences=None, observations=None, revision=0):
        from src.ace_context import AceContextEngine, ExperiencePlaybook
        from src.context_engine import ContextEngine
        from src.runtime import current_runtime

        playbook = ExperiencePlaybook.from_path(ROOT / self.playbook_path)
        if playbook.fingerprint != self.playbook_sha256:
            raise ValueError("ACE Playbook 在 Runner 创建后发生变化")
        runtime = current_runtime()
        if runtime is not None:
            runtime.record(
                "mechanism_applied",
                slot=self.slot,
                mechanism_id=self.mechanism_id,
                mechanism_version=self.mechanism_version,
                configuration={
                    "selection_strategy": "saturate-then-rank",
                    "playbook_id": playbook.playbook_id,
                    "playbook_version": playbook.version,
                    "playbook_sha256": playbook.fingerprint,
                    "max_bullets": self.max_bullets,
                    "max_chars": self.max_chars,
                },
                goal_revision=revision,
            )
        base = ContextEngine(
            goal,
            preferences=preferences,
            observations=observations,
            revision=revision,
            selection_strategy="recent-first",
            mechanism_id=self.mechanism_id,
        )
        return AceContextEngine(base, playbook, max_bullets=self.max_bullets,
                                max_chars=self.max_chars)


def built_in_mechanisms() -> tuple[HarnessMechanism, ...]:
    """Return exact implementations referenced by built-in Profiles."""

    from src.liveplan import LivePlanSteeringMechanism
    from src.pg_context import PGContextMechanism

    after_sales_planning_sources = (
        "src/orchestrator.py", "src/dialogue_understanding.py", "src/task_plan.py",
        "src/after_sales/execution.py", "src/catalog_request_audit.py", "src/catalog_query.py",
        "src/capability_registry.py", "src/catalog_concepts.py", "src/catalog_metadata.py", "src/intake.py",
        "src/harness/mechanisms.py",
    )
    evidence_readiness_sources = after_sales_planning_sources + (
        "src/after_sales/readiness.py", "src/after_sales/planning.py", "src/after_sales/policy.py",
        "src/after_sales/service.py", "src/after_sales/contracts.py", "src/after_sales/actions.py",
        "src/runtime.py", "src/tools.py",
    )
    obligation_graph_sources = evidence_readiness_sources + (
        "src/after_sales/obligation_graph.py",
    )
    active_clarification_sources = obligation_graph_sources + (
        "src/after_sales/clarification.py", "src/after_sales/pending_clarification.py",
    )
    return (
        EmbeddedMechanism("planning", "structured-dialogue-understanding", "1", ("orchestrator",),
                          ("src/orchestrator.py", "src/dialogue_understanding.py",
                           "src/catalog_request_audit.py", "src/catalog_query.py",
                           "src/capability_registry.py", "src/catalog_concepts.py",
                           "src/catalog_metadata.py", "src/intake.py")),
        AfterSalesPlanningMechanism(
            "planning", "case-plan-observer", "0", ("orchestrator",),
            after_sales_planning_sources, case_execution_mode="observe_only",
        ),
        AfterSalesPlanningMechanism(
            "planning", "policy-aware-case-execution", "1", ("orchestrator",),
            after_sales_planning_sources, case_execution_mode="policy_aware",
        ),
        AfterSalesPlanningMechanism(
            "planning", "evidence-aware-case-execution", "2", ("orchestrator",),
            after_sales_planning_sources,
            case_execution_mode="policy_aware",
            evidence_intake_mode="case_first",
        ),
        AfterSalesPlanningMechanism(
            "planning", "evidence-readiness-case-execution", "3", ("orchestrator",),
            evidence_readiness_sources,
            case_execution_mode="policy_aware",
            evidence_intake_mode="readiness_ledger",
        ),
        AfterSalesPlanningMechanism(
            "planning", "obligation-graph-case-execution", "4", ("orchestrator",),
            obligation_graph_sources,
            case_execution_mode="policy_aware",
            evidence_intake_mode="readiness_ledger",
            action_selection_mode="obligation_graph",
        ),
        AfterSalesPlanningMechanism(
            "planning", "active-clarification-case-execution", "5", ("orchestrator",),
            active_clarification_sources,
            case_execution_mode="policy_aware",
            evidence_intake_mode="readiness_ledger",
            action_selection_mode="active_clarification",
        ),
        EmbeddedMechanism("tool_protocol", "runtime-tool-ledger", "2", ("orchestrator",),
                          ("src/runtime.py", "src/persistence.py", "src/backend/db.py",
                           "src/execution_boundaries.py",
                           "src/after_sales/pending_clarification.py")),
        EvidenceVerificationMechanism("verification", "deterministic-business-gates", "2", ("orchestrator",),
                          ("src/direct_executor.py", "src/task_plan.py", "src/catalog_query.py",
                           "src/after_sales/semantic.py", "src/after_sales/evidence.py",
                           "src/after_sales/actions.py", "src/after_sales/pending_clarification.py",
                           "src/after_sales/policy.py", "src/tools.py")),
        EmbeddedMechanism("planning", "bounded-model-loop", "1", ("adaptive",),
                          ("src/adaptive.py", "src/adaptive_decision.py", "src/request_semantics.py",
                           "src/request_semantics_runtime.py")),
        ContextMechanism("context", "evidence-reference-context", "1", ("adaptive",),
                         ("src/context_engine.py", "src/harness/mechanisms.py"), "recent-first"),
        ContextMechanism("context", "objective-first-context", "1", ("adaptive",),
                         ("src/context_engine.py", "src/harness/mechanisms.py"), "objective-first"),
        AceContextMechanism("context", "ace-playbook-context", "1", ("adaptive",),
                            ("src/ace_context.py", "src/context_engine.py", "src/harness/mechanisms.py")),
        EmbeddedMechanism("memory", "objective-ledger", "1", ("adaptive",),
                          ("src/adaptive_goals.py",)),
        EvidenceVerificationMechanism("verification", "independent-request-audit", "2", ("adaptive",),
                          ("src/request_audit.py", "src/adaptive_goals.py", "src/after_sales/semantic.py",
                           "src/after_sales/evidence.py", "src/after_sales/actions.py", "src/after_sales/policy.py", "src/tools.py")),
        EmbeddedMechanism("tool_protocol", "native-strict-tools", "1", ("adaptive",),
                          ("src/native_protocol.py", "src/runtime.py", "src/execution_boundaries.py")),
        LivePlanSteeringMechanism(),
        PGContextMechanism(),
        PGContextMechanism(
            mechanism_version="2",
            retrieval_strategy="trajectory-state-v2",
            max_local_edges=6,
            min_localization_score=55,
        ),
        PGContextMechanism(
            mechanism_version="3",
            retrieval_strategy="trajectory-state-v2",
            max_local_edges=6,
            min_localization_score=55,
            guidance_strategy="deterministic-contract-v1",
        ),
        PGContextMechanism(
            mechanism_version="4",
            retrieval_strategy="trajectory-state-v2",
            max_local_edges=6,
            min_localization_score=55,
            guidance_strategy="deterministic-contract-v1",
            applicability_strategy="typed-evidence-v1",
            path_ranking_strategy="goal-progress-cost-v1",
        ),
        PGContextMechanism(
            mechanism_version="6",
            retrieval_strategy="trajectory-state-v2",
            max_local_edges=6,
            min_localization_score=55,
            guidance_strategy="deterministic-contract-v1",
            applicability_strategy="typed-evidence-v1",
            path_ranking_strategy="goal-progress-cost-v1",
            preplay_strategy="bounded-transition-search-v1",
            preplay_max_depth=3,
            preplay_beam_width=4,
            preplay_max_paths=4,
        ),
    )

