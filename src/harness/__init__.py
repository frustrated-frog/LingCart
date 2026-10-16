"""Public harness boundary for product runners and research experiments."""

from src.harness.contracts import AgentEngine, RunRequest
from src.harness.profile import HarnessProfile
from src.harness.registry import (
    HarnessRegistry,
    get_product_runner,
    get_registry,
    get_runner,
    harness_execution_identity,
    harness_execution_matches,
    persisted_harness,
    resolve_claim_runner,
    runner_execution_contract,
    unwrap_harness_runner,
)

__all__ = [
    "AgentEngine", "HarnessProfile", "HarnessRegistry", "RunRequest", "get_product_runner", "get_registry",
    "get_runner", "harness_execution_identity", "harness_execution_matches", "persisted_harness",
    "resolve_claim_runner", "runner_execution_contract", "unwrap_harness_runner",
]

