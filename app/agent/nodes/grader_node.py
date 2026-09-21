"""
app/agent/nodes/grader_node.py
==============================
Guardrail node — checks the user's prompt for scope and harm before the
agent engages. Runs as the first node in the graph (entry point).
"""

from __future__ import annotations

from typing import Dict, Any

import structlog

from app.agent.state import VyaparAgentState
from app.agent.utils import latest_human_prompt, message_text
from app.lib.grader import is_retail_follow_up, _SEVERE_HARM_KEYWORDS
from langchain_core.messages import AIMessage, HumanMessage

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.grader")

_REFUSAL_HARMFUL = (
    "I'm sorry, but I can't help with that request. "
    "Vyapar Copilot is designed to assist with retail store operations "
    "such as inventory, sales, forecasting, and restocking. "
    "Please let me know if there's something else I can help you with."
)

_REFUSAL_OFF_TOPIC = (
    "That's outside what Vyapar Copilot handles. I can help with your "
    "store's inventory, sales, forecasting, restocking, and business "
    "insights — ask me about any of those."
)


def _current_user_prompt(state: VyaparAgentState) -> str:
    """Return the latest user turn, even if a checkpoint has stale scalar state."""
    return latest_human_prompt(state.get("messages", []), state.get("user_prompt", ""))


def _recent_context(state: VyaparAgentState, current_prompt: str) -> str:
    """Build bounded context from earlier turns without replacing the current prompt."""
    parts: list[str] = []
    for message in (state.get("messages", []) or [])[:-1]:
        content = message_text(getattr(message, "content", ""))
        if content:
            parts.append(content)
    return "\n".join(parts)[-4_000:]


async def grader_node(state: VyaparAgentState) -> Dict[str, Any]:
    user_prompt = _current_user_prompt(state)
    store_id = state.get("store_id", "unknown")
    user_id = state.get("user_id", "unknown")

    LOGGER.info("grader_node_start", store_id=store_id, user_id=user_id, prompt_len=len(user_prompt))

    # When resuming after a clarification, the prompt is the user's
    # answer which is always safe — skip classification entirely.
    if state.get("resume_from_clarification", False):
        LOGGER.info(
            "grader_node_resume_skip",
            store_id=store_id,
            reason="resuming from clarification — prompt is user answer",
        )
        return {
            "user_prompt": user_prompt,
            "grader_denied": False,
            "grader_reason": "skipped — resume from clarification",
        }

    if not user_prompt.strip():
        return {
            "user_prompt": user_prompt,
            "grader_denied": False,
            "grader_reason": "",
        }

    # TEMP: Disabled scope classification — grader blocks valid queries
    # when no classifier API key is configured (fail-closed).
    # Will re-enable with proper API keys later.
    if is_retail_follow_up(user_prompt, _recent_context(state, user_prompt)):
        LOGGER.info(
            "grader_contextual_retail_follow_up",
            store_id=store_id,
            user_prompt_len=len(user_prompt),
        )
        return {
            "user_prompt": user_prompt,
            "grader_denied": False,
            "grader_reason": "contextual retail operations follow-up",
        }

    # Basic harmful keyword floor (safety net while scope check is disabled)
    p = (user_prompt or "").lower()
    if any(kw in p for kw in _SEVERE_HARM_KEYWORDS):
        LOGGER.warning(
            "grader_node_harmful_keyword",
            store_id=store_id,
            user_prompt_len=len(user_prompt),
        )
        return {
            "user_prompt": user_prompt,
            "grader_denied": True,
            "grader_reason": "temporary safety floor — severe harm keyword matched",
            "goal_status": "complete",
            "goal": "refuse harmful request",
            "final_answer": _REFUSAL_HARMFUL,
            "should_persist_memory": False,
            "messages": [AIMessage(content=_REFUSAL_HARMFUL)],
            "response_metadata": {"grader_denied": True, "grader_verdict": "harmful"},
        }

    # Skip LLM classification for now — allow everything non-harmful
    LOGGER.info(
        "grader_node_disabled",
        store_id=store_id,
        reason="temporarily disabled — all non-harmful prompts allowed",
    )
    return {
        "user_prompt": user_prompt,
        "grader_denied": False,
        "grader_reason": "",
    }