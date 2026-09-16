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
from app.lib.grader import classify_prompt, is_retail_follow_up
from app.agent.prompts.classifier_prompts import CLASSIFIER_HARM_INSTRUCTION
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

    if not user_prompt.strip():
        return {
            "user_prompt": user_prompt,
            "grader_denied": False,
            "grader_reason": "",
        }

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

    verdict, reason = await classify_prompt(user_prompt, instruction=CLASSIFIER_HARM_INSTRUCTION)

    if verdict in ("harmful", "off_topic"):
        refusal = _REFUSAL_HARMFUL if verdict == "harmful" else _REFUSAL_OFF_TOPIC
        LOGGER.warning(
            "grader_node_denied",
            store_id=store_id, verdict=verdict, reason=reason,
            user_prompt_len=len(user_prompt),
        )
        return {
            "user_prompt": user_prompt,
            "grader_denied": True,
            "grader_reason": reason or verdict,
            "goal_status": "complete",
            "goal": f"refuse {verdict} request",
            "final_answer": refusal,
            "should_persist_memory": False,
            "messages": [AIMessage(content=refusal)],
            "response_metadata": {"grader_denied": True, "grader_verdict": verdict, "grader_reason": reason or verdict},
        }

    LOGGER.info("grader_node_safe", store_id=store_id, reason=reason)
    return {
        "user_prompt": user_prompt,
        "grader_denied": False,
        "grader_reason": "",
    }