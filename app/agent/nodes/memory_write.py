"""
app/agent/nodes/memory_write.py
================================
Memory write node — persists conversation summary to the Pinecone Multi-Level
Long-Term Memory System after the agent completes its run.

This node runs AFTER the agent has produced its final answer (typically invoked
as a background task or post-response step). It:

    1. Summarizes the conversation (user prompt + assistant response).
    2. Stores the summary in USER memory (`add_user_memory`):
       — captures preferences like language, tone, detail level, business goals.
    3. Stores the summary in STORE memory (`add_store_memory`):
       — captures store-specific patterns, restock decisions, vendor schedules.
    4. Stores the summary in MULTI-STORE memory (`add_multi_store_memory`):
       — captures cross-store transfer rules, chain strategies, multi-store insights.

Memory persistence is non-blocking — if Pinecone or embedding fails, the error
is logged safely without breaking the user response.
"""

from __future__ import annotations

from typing import Dict, Any

import structlog

from app.agent.state import VyaparAgentState
from app.agent.memory import (
    add_user_memory,
    add_store_memory,
    add_multi_store_memory,
)

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.memory_write")


async def memory_write_node(state: VyaparAgentState) -> Dict[str, Any]:
    """
    Persist conversation context to Pinecone multi-level long-term memory.

    Runs after the agent has completed its final answer.
    Failures are non-fatal — they are logged but don't affect the response.

    Checks ``should_persist_memory`` flag set by the think node.
    If False (trivial conversation like "hi"), the node returns immediately.
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
    store_ids = state.get("store_ids", [store_id]) if isinstance(state.get("store_ids"), list) else [store_id]
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
        "pinecone_memory_write_start",
        store_id=store_id,
        user_id=user_id,
        message_count=len(messages),
    )

    # 1. Store at USER level (preferences, tone, language, detail, etc.)
    user_ok = await add_user_memory(user_id, messages)

    # 2. Store at STORE level (store domain facts, restock patterns, decisions)
    store_ok = await add_store_memory(store_id, messages)

    # 3. Store at MULTI-STORE level (cross-store chain strategies, transfer rules)
    multi_store_ok = await add_multi_store_memory(user_id, store_ids, messages)

    LOGGER.info(
        "pinecone_memory_write_complete",
        store_id=store_id,
        user_id=user_id,
        user_ok=user_ok,
        store_ok=store_ok,
        multi_store_ok=multi_store_ok,
    )

    return {}