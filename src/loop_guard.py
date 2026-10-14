"""Progress-aware repeated-action guard for agent tool loops."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json


class LoopAction(StrEnum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class LoopDecision:
    action: LoopAction
    occurrence: int
    stagnant_results: int


class ProgressAwareLoopGuard:
    """Warn once, then block when an identical action yields no new result."""

    def __init__(self):
        self._occurrences: dict[str, int] = {}
        self._result_signatures: dict[str, str] = {}
        self._stagnant_results: dict[str, int] = {}

    def before_call(self, fingerprint: str) -> LoopDecision:
        occurrence = self._occurrences.get(fingerprint, 0) + 1
        self._occurrences[fingerprint] = occurrence
        stagnant = self._stagnant_results.get(fingerprint, 0)
        if occurrence >= 3 and stagnant >= 1:
            return LoopDecision(LoopAction.BLOCK, occurrence, stagnant)
        if occurrence >= 2:
            return LoopDecision(LoopAction.WARN, occurrence, stagnant)
        return LoopDecision(LoopAction.ALLOW, occurrence, stagnant)

    def after_result(self, fingerprint: str, result) -> bool:
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        signature = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        previous = self._result_signatures.get(fingerprint)
        self._result_signatures[fingerprint] = signature
        if previous is None or previous != signature:
            self._stagnant_results[fingerprint] = 0
            return True
        self._stagnant_results[fingerprint] = self._stagnant_results.get(fingerprint, 0) + 1
        return False
