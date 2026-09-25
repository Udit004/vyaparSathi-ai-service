"""
app/agent/nodes/memory_write.py
================================
Memory write node — persists conversation summary to mem0 after the agent
completes its run.

This node runs AFTER the agent has produced its final answer. It:

    1. Summarizes the conversation (user prompt + assistant response)
    2. Stores the summary in user memory (namespace = user_id)
       — captures preferences like language, tone, detail level
    3. Stores the summary in store memory (namespace = store_id)
       — captures store-specific patterns, decisions, and knowledge

Memory is stored at two levels:
    - USER level   → preferences, tone, language, detail level
    - STORE level  → product patterns, category notes, past decisions

The node is designed to never block the response — if mem0 fails, the
error is logged but the agent's final answer is still returned.
"""

from __future__ import annotations

from typing import Dict, Any

import structlog

from app.agent.state import VyaparAgentState
from app.agent.memory import add_user_memory, add_store_memory

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.memory_write")


async def memory_write_node(state: VyaparAgentState) -> Dict[str, Any]:
    """
    Persist conversation context to mem0 long-term memory.

    Runs after the agent has completed its final answer.
    Failures are non-fatal — they are logged but don't affect the response.

    The node checks ``should_persist_memory`` flag set by the think node.
    If False (trivial conversation like "hi"), the node returns immediately
    without calling mem0.
    """
    # Skip if the think node determined this conversation is not worth persisting
    if not state.get("should_persist_memory", False):
        LOGGER.debug(
            "memory_write_skip",
            reason="should_persist_memory=False (trivial conversation)",
        )
        return {}

    store_id = state.get("store_id", "unknown")
    user_id = state.get("user_id", "unknown")
    user_prompt = state.get("user_prompt", "")
    final_answer = state.get("final_answer", "")

    if not user_prompt and not final_answer:
        LOGGER.debug("memory_write_skip", reason="empty conversation")
        return {}

    # Build the message pair to store
    messages = []
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    if final_answer:
        messages.append({"role": "assistant", "content": final_answer})

    if not messages:
        return {}

    LOGGER.info(
        "memory_write_start",
        store_id=store_id,
        user_id=user_id,
        message_count=len(messages),
    )

    # Store at user level (preferences, tone, language, etc.)
    user_ok = await add_user_memory(user_id, messages, curated_messages=messages)

    # Store at store level (patterns, decisions, store knowledge)
    store_ok = await add_store_memory(store_id, messages, curated_messages=messages)

    LOGGER.info(
        "memory_write_complete",
        store_id=store_id,
        user_id=user_id,
        user_ok=user_ok,
        store_ok=store_ok,
    )

    return {}