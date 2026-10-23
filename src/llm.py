from contextlib import contextmanager
from contextvars import ContextVar

from dotenv import load_dotenv

from src.model_gateway import RotatingChatModel, create_chat_model, get_model_status

load_dotenv()

_model_attempt_limit = ContextVar("model_attempt_limit", default=None)


@contextmanager
def model_attempt_limit(value: int | None):
    if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
        raise ValueError("模型尝试上限必须是正整数")
    token = _model_attempt_limit.set(value)
    try:
        yield
    finally:
        _model_attempt_limit.reset(token)


def get_llm(model: str | None = None, temperature: float = 0.0, *, purpose: str = "default") -> RotatingChatModel:
    result = create_chat_model(purpose=purpose, model=model, temperature=temperature)
    limit = _model_attempt_limit.get()
    if limit is not None:
        result.max_attempts = min(result.max_attempts, limit)
    return result


def extract_usage(ai_message) -> dict:
    """Pull token usage out of a LangChain AIMessage, if present."""
    meta = getattr(ai_message, "usage_metadata", None)
    if not meta:
        return {}
    return {
        "input_tokens": meta.get("input_tokens"),
        "output_tokens": meta.get("output_tokens"),
        "total_tokens": meta.get("total_tokens"),
    }
