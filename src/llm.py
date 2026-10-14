from dotenv import load_dotenv

from src.model_gateway import RotatingChatModel, create_chat_model, get_model_status

load_dotenv()


def get_llm(model: str | None = None, temperature: float = 0.0, *, purpose: str = "default") -> RotatingChatModel:
    return create_chat_model(purpose=purpose, model=model, temperature=temperature)


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
