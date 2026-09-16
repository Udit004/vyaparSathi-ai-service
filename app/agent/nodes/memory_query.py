"""
app/agent/nodes/memory_query.py
================================
Memory query node — conditionally fetches long-term memory from mem0.

This node sits between the think node and the tool node in the agent loop.
The think node sets ``memory_query_needed=True`` when it decides it needs
more context from long-term memory. This node then:

    1. Queries mem0 for user preferences (namespace = user_id)
    2. Queries mem0 for store knowledge (namespace = store_id)
    3. Stores results in ``user_preferences`` / ``store_knowledge``
    4. Sets ``memory_query_needed=False`` to prevent re-querying
    5. Sets ``user_memory_loaded`` / ``store_memory_loaded`` flags

After this node runs, the agent loops back to think so the LLM can
synthesize the memory context with any tool data it has.
"""

from __future__ import annotations

from typing import Dict, Any

import structlog

from app.agent.state import VyaparAgentState
from app.agent.memory import load_memory_context, should_retrieve_memory
from app.agent.utils import latest_human_prompt

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.memory_query")


async def memory_query_node(state: VyaparAgentState) -> Dict[str, Any]:
    """
    Conditionally fetch mem0 long-term memory for user + store.

    Only runs if ``memory_query_needed`` is True and memory hasn't
    been loaded yet for this session.
    """
    store_id = state.get("store_id", "unknown")
    user_id = state.get("user_id", "unknown")
    needs_query = state.get("memory_query_needed", False)
    already_loaded = (
        state.get("user_memory_loaded", False)
        and state.get("store_memory_loaded", False)
    )

    if not needs_query or already_loaded:
        LOGGER.debug(
            "memory_query_skip",
            store_id=store_id,
            needs_query=needs_query,
            already_loaded=already_loaded,
        )
        return {"memory_query_needed": False}

    LOGGER.info(
        "memory_query_start",
        store_id=store_id,
        user_id=user_id,
    )

    user_prompt = latest_human_prompt(
        state.get("messages", []),
        state.get("user_prompt", ""),
    )

    intent = state.get("intent", "general")
    needs_memory_intent = intent in {"memory", "mixed"}
    if not needs_memory_intent and not should_retrieve_memory(user_prompt):
        LOGGER.info(
            "memory_query_not_needed",
            store_id=store_id,
            intent=intent,
            reason="request is answered by live tools or recent conversation",
        )
        return {
            "user_memory_loaded": True,
            "store_memory_loaded": True,
            "memory_query_needed": False,
            "user_preferences": {"raw": [], "summary": ""},
            "store_knowledge": {"raw": [], "summary": ""},
        }

    try:
        user_prefs, store_knowledge = await load_memory_context(
            user_id=user_id,
            store_id=store_id,
            user_prompt=user_prompt,
            current_goal=state.get("goal") or f"Answer the user's question: {user_prompt}",
        )
    except Exception as exc:
        LOGGER.error(
            "memory_query_failed",
            store_id=store_id,
            error=str(exc),
        )
        return {
            "memory_query_needed": False,
            "error": str(exc),
        }

    LOGGER.info(
        "memory_query_complete",
        store_id=store_id,
        user_results=len(user_prefs.get("raw", [])),
        store_results=len(store_knowledge.get("raw", [])),
    )

    return {
        "user_memory_loaded": True,
        "store_memory_loaded": True,
        "memory_query_needed": False,
        "user_preferences": user_prefs,
        "store_knowledge": store_knowledge,
    }