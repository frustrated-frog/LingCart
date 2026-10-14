from langchain_core.tools import tool

from src import tools as backend_tools
from src.agents.base import run_tool_agent


@tool
def answer_faq(question: str) -> str:
    """Answer a general policy question (shipping, returns, cancellations, refunds, contact info)."""
    return backend_tools.answer_faq(question)


FAQ_TOOLS = [answer_faq]

SYSTEM_PROMPT = (
    "You are the FAQ Agent for an e-commerce support system, answering general policy questions "
    "about shipping, returns, cancellations, refunds, and how to contact support. "
    "Always call answer_faq to get the official policy text before answering; never guess. "
    "Reply concisely and naturally to the customer."
)

def run_faq_agent(user_message: str, history: list | None = None, run_id: str | None = None) -> dict:
    # Deterministic policy requests already use the no-model intake fast path.
    # Model-led results must carry this run's real events, never another run's trace.
    return run_tool_agent("faq", SYSTEM_PROMPT, FAQ_TOOLS, user_message, history, run_id=run_id)
