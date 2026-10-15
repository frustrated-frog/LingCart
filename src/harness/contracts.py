"""Stable contracts at the boundary between the runtime and agent engines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.runtime import BudgetLimits


@dataclass(frozen=True)
class RunRequest:
    """One engine invocation without leaking persistence concerns into engines."""

    message: str
    history: list[Any] | None = None
    pending_action: dict | None = None
    limits: BudgetLimits | None = None


@runtime_checkable
class AgentEngine(Protocol):
    """Small execution seam kept stable while paper mechanisms evolve."""

    engine_id: str
    engine_version: str
    execution_mode: str

    def execute(self, request: RunRequest) -> dict:
        """Execute one request and return the existing runtime result contract."""

