"""Intent routing node for choosing live tools, chat history, or Mem0."""

from __future__ import annotations

from typing import Any, Dict

import structlog

from app.agent.prompts.intent_prompt import INTENT_CLASSIFIER_INSTRUCTION
from app.agent.state import VyaparAgentState
from app.agent.utils import latest_human_prompt, message_text
from app.lib.summarizer import summarize

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.intent")

_VALID_INTENTS = {"live_data", "memory", "conversation_recap", "mixed", "general"}

_LIVE_TERMS = (
    "inventory", "stock", "stockout", "out of stock", "sales", "selling",
    "forecast", "forecasting", "restock", "anomaly", "store insight",
    "store summary", "store overview", "store health",
)
_MEMORY_TERMS = (
    "remember", "last time", "previous", "earlier", "we decided",
    "our decision", "usually", "prefer", "preference", "my language",
    "my style", "tell me about myself", "about myself", "what do you know about me",
)
_RECAP_TERMS = (
    "what did we discuss", "what were we discussing", "recent chat",
    "recent conversation", "summarize our chat", "summarise our chat",
)


def _deterministic_intent(prompt: str) -> str | None:
    lowered = (prompt or "").lower()
    has_live = any(term in lowered for term in _LIVE_TERMS)
    has_memory = any(term in lowered for term in _MEMORY_TERMS)
    has_recap = any(term in lowered for term in _RECAP_TERMS)

    if has_recap:
        return "conversation_recap"
    if has_live and has_memory:
        return "mixed"
    if has_live:
        return "live_data"
    if has_memory:
        return "memory"
    return None


async def intent_node(state: VyaparAgentState) -> Dict[str, Any]:
    prompt = latest_human_prompt(
        state.get("messages", []),
        state.get("user_prompt", ""),
    ).strip()
    deterministic = _deterministic_intent(prompt)
    reason = "deterministic routing"
    intent = deterministic

    if intent is None and prompt:
        result = await summarize(
            prompt,
            instruction=INTENT_CLASSIFIER_INSTRUCTION,
            max_tokens=32,
        )
        first_token = (result or "").strip().lower().split()[0] if result else ""
        if first_token in _VALID_INTENTS:
            intent = first_token
            reason = "small-model routing"

    intent = intent or "general"
    needs_memory = intent in {"memory", "mixed"}

    LOGGER.info(
        "intent_classified",
        intent=intent,
        reason=reason,
        needs_memory=needs_memory,
        prompt_len=len(prompt),
    )
    return {
        "user_prompt": prompt,
        "intent": intent,
        "intent_reason": reason,
        "memory_query_needed": needs_memory,
        "user_memory_loaded": not needs_memory,
        "store_memory_loaded": not needs_memory,
    }