"""Executable Harness mechanisms and their source-bound identities."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
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

    def provenance(self) -> dict:
        return {
            "slot": self.slot,
            "id": self.mechanism_id,
            "version": self.mechanism_version,
            "implementation_sha256": source_fingerprint(self.source_files),
            "source_files": list(self.source_files),
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


def built_in_mechanisms() -> tuple[HarnessMechanism, ...]:
    """Return exact implementations referenced by built-in Profiles."""

    return (
        EmbeddedMechanism("planning", "routed-specialists", "1", ("orchestrator",),
                          ("src/orchestrator.py", "src/planner.py")),
        EmbeddedMechanism("tool_protocol", "runtime-tool-ledger", "2", ("orchestrator",),
                          ("src/runtime.py", "src/persistence.py")),
        EmbeddedMechanism("verification", "deterministic-business-gates", "1", ("orchestrator",),
                          ("src/direct_executor.py", "src/task_plan.py")),
        EmbeddedMechanism("planning", "bounded-model-loop", "1", ("adaptive",),
                          ("src/adaptive.py", "src/adaptive_decision.py")),
        ContextMechanism("context", "evidence-reference-context", "1", ("adaptive",),
                         ("src/context_engine.py", "src/harness/mechanisms.py"), "recent-first"),
        ContextMechanism("context", "objective-first-context", "1", ("adaptive",),
                         ("src/context_engine.py", "src/harness/mechanisms.py"), "objective-first"),
        EmbeddedMechanism("memory", "objective-ledger", "1", ("adaptive",),
                          ("src/adaptive_goals.py",)),
        EmbeddedMechanism("verification", "independent-request-audit", "1", ("adaptive",),
                          ("src/request_audit.py", "src/adaptive_goals.py")),
        EmbeddedMechanism("tool_protocol", "native-strict-tools", "1", ("adaptive",),
                          ("src/native_protocol.py", "src/runtime.py")),
    )

