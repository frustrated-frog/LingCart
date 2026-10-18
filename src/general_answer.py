"""Bounded model-backed answers for questions that need no business tool."""
from __future__ import annotations

from src.agents.base import as_text
from src.errors import AgentError
from src.llm import get_llm
from src.runtime import current_runtime


_PROMPT = """You are LingCart's general response capability. Answer the HUMAN question directly in Chinese unless
the user clearly requested another language. This capability has no live web, private account or business database
access. Use it only for stable knowledge, explanations, drafting and planning. Do not invent current facts, citations,
prices, policies, inventory, order data or actions. If correctness depends on missing current or private evidence,
say exactly what evidence is unavailable and what the user can provide. Be useful and concise.

HUMAN question:
"""


def generate_general_answer(question: str, evidence_requirement: str = "MODEL_KNOWLEDGE") -> dict:
    if evidence_requirement == "REALTIME_EXTERNAL":
        return {
            "answer": "这个问题依赖实时外部信息，但当前 LingCart 没有可验证的联网数据源。我不能用旧知识猜测；你可以提供来源或启用相应检索能力后，我再据此回答。",
            "question": question,
            "evidence_requirement": evidence_requirement,
            "provenance": "NO_REALTIME_SOURCE",
        }
    model = get_llm(purpose="final_response")
    context = current_runtime()
    response = (
        context.model_call(model, _PROMPT + question, purpose="general_answer", agent="knowledge",
                           allow_tools=False)
        if context else model.invoke(_PROMPT + question)
    )
    if getattr(response, "tool_calls", []):
        raise AgentError("MODEL_FORMAT_ERROR", "通用回答能力不能调用业务工具", retryable=False)
    answer = as_text(response.content).strip()
    if not answer:
        raise AgentError("EMPTY_MODEL_RESPONSE", "通用回答模型没有返回内容", retryable=True)
    return {
        "answer": answer,
        "question": question,
        "evidence_requirement": evidence_requirement,
        "provenance": "MODEL_GENERAL_KNOWLEDGE",
    }

