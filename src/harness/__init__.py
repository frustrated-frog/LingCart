"""Public harness boundary for product runners and research experiments."""

from src.harness.contracts import AgentEngine, RunRequest
from src.harness.profile import HarnessProfile
from src.harness.registry import (
    HarnessRegistry,
    get_registry,
    get_runner,
    persisted_harness,
    resolve_claim_runner,
    unwrap_harness_runner,
)

__all__ = [
    "AgentEngine", "HarnessProfile", "HarnessRegistry", "RunRequest", "get_registry", "get_runner",
    "persisted_harness", "resolve_claim_runner", "unwrap_harness_runner",
]

