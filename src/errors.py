import logging
import time
from enum import StrEnum

from src.tools import escalate_to_human

logger = logging.getLogger("support_agent.errors")


class ErrorCode(StrEnum):
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_RATE_LIMIT = "MODEL_RATE_LIMIT"
    MODEL_FORMAT_ERROR = "MODEL_FORMAT_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    INVALID_TOOL_ARGUMENTS = "INVALID_TOOL_ARGUMENTS"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_ERROR = "TOOL_ERROR"
    TOOL_BUDGET_EXHAUSTED = "TOOL_BUDGET_EXHAUSTED"
    BUSINESS_RULE_REJECTED = "BUSINESS_RULE_REJECTED"


class AgentError(RuntimeError):
    def __init__(self, code: ErrorCode | str, message: str, *, retryable: bool):
        super().__init__(message)
        self.code = str(code)
        self.retryable = retryable


def classify_exception(exc: Exception, *, source: str = "model") -> AgentError:
    """Convert provider/framework exceptions into stable product semantics."""

    if isinstance(exc, AgentError):
        return exc
    text = str(exc)
    lowered = text.lower()
    if "no llm configured" in lowered:
        return AgentError(ErrorCode.MODEL_UNAVAILABLE, text, retryable=False)
    if isinstance(exc, TimeoutError) or "timeout" in lowered or "timed out" in lowered:
        code = ErrorCode.TOOL_TIMEOUT if source == "tool" else ErrorCode.MODEL_TIMEOUT
        return AgentError(code, text, retryable=True)
    if "429" in lowered or "rate limit" in lowered or "resource_exhausted" in lowered:
        return AgentError(ErrorCode.MODEL_RATE_LIMIT, text, retryable=True)
    if source == "tool":
        if "validation error" in lowered or "field required" in lowered or "unexpected keyword" in lowered:
            return AgentError(ErrorCode.INVALID_TOOL_ARGUMENTS, text, retryable=False)
        return AgentError(ErrorCode.TOOL_ERROR, text, retryable=False)
    return AgentError(ErrorCode.MODEL_ERROR, text, retryable=False)


def call_with_retry(fn, *, max_attempts: int = 2, backoff_seconds: float = 3.0, escalate_reason: str, escalate_context: str):
    """Run fn(); on failure, retry once after a short backoff; if it still fails, escalate to a human
    instead of crashing. Returns (result, escalation) — exactly one of which is non-None."""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(), None
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any agent/LLM failure should escalate, not crash
            last_exc = exc
            logger.warning("Attempt %d/%d failed: %s", attempt, max_attempts, exc)
            if attempt < max_attempts:
                time.sleep(backoff_seconds)

    escalation = escalate_to_human(reason=escalate_reason, context=f"{escalate_context} | error: {last_exc}")
    return None, escalation
