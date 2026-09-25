"""
app/agent/nodes/memory_query.py
================================
Bootstrap Memory Node — loads a small, focused amount of long-term memory
context at the start of each agent run, before the think node executes.

Architecture
------------
This node replaces the old "load everything" memory_query_node with
intent-aware, budget-constrained retrieval:

  intent=live_data   → minimal user prefs only (language/tone)
  intent=memory      → user prefs + deep store/decision search
  intent=mixed       → user prefs + store facts + patterns
  intent=general     → user prefs + top store facts
  intent=recap       → user prefs only (agent will scan messages)

The node is designed to run ONCE per graph invocation (guarded by the
already_loaded flag). The LLM can retrieve additional memories on-demand
via the search_memory tool.

What this node does NOT do
--------------------------
- It does NOT load all Pinecone memories into context.
- It does NOT search episodic_event / decision memories by default.
  Those are retrieved on-demand by the search_memory tool.
"""

from __future__ import annotations

from typing import Dict, Any

import structlog

from app.agent.state import VyaparAgentState
from app.agent.memory import (
    search_user_memory,
    search_store_memory,
    search_multi_store_memory,
    build_memory_query,
    summarize_memory_results,
    _cap_results,
    _filter_relevant_results,
)
from app.agent.utils import latest_human_prompt

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.bootstrap_memory")

# Memory types fetched during bootstrap per intent
_BOOTSTRAP_TYPES_BY_INTENT = {
    "live_data": ["user_preference", "user_fact"],
    "general": ["user_preference", "user_fact", "store_fact"],
    "memory": ["user_preference", "user_fact", "store_fact", "store_pattern", "decision"],
    "mixed": ["user_preference", "user_fact", "store_fact", "store_pattern"],
    "conversation_recap": ["user_preference", "user_fact"],
}

# Maximum memories per scope for bootstrap (to keep context small)
_BOOTSTRAP_CAP_USER = 3
_BOOTSTRAP_CAP_STORE = 4
_BOOTSTRAP_CAP_MULTI = 2


async def _bootstrap_user_memory(user_id: str, query: str) -> list[dict]:
    """Fetch stable user preferences/facts for bootstrap."""
    results = await search_user_memory(user_id, query, top_k=_BOOTSTRAP_CAP_USER)
    if not results:
        # Fallback to a general preference query
        results = await search_user_memory(
            user_id,
            "user preference language tone detail level business goals",
            top_k=_BOOTSTRAP_CAP_USER,
        )
    return _cap_results(results)[:_BOOTSTRAP_CAP_USER]


async def _bootstrap_store_memory(store_id: str, query: str, intent: str) -> list[dict]:
    """Fetch relevant store facts / patterns for bootstrap."""
    results = await search_store_memory(store_id, query, top_k=_BOOTSTRAP_CAP_STORE)
    if not results:
        results = await search_store_memory(
            store_id,
            "store facts supplier rules restock schedule inventory patterns",
            top_k=_BOOTSTRAP_CAP_STORE,
        )
    # Filter out low-relevance results for non-memory intents
    if intent not in ("memory", "mixed"):
        results = _filter_relevant_results(results)
    return _cap_results(results)[:_BOOTSTRAP_CAP_STORE]


async def memory_query_node(state: VyaparAgentState) -> Dict[str, Any]:
    """
    Bootstrap Memory Node.

    Loads a small, intent-aware set of long-term memories at graph start.
    Runs once and is skipped on subsequent loops (already_loaded guard).
    """
    store_id = state.get("store_id", "unknown")
    user_id = state.get("user_id", "unknown")
    intent = state.get("intent", "general")

    already_loaded = (
        state.get("user_memory_loaded", False)
        and state.get("store_memory_loaded", False)
        and state.get("multi_store_memory_loaded", False)
    )

    if already_loaded:
        LOGGER.debug("bootstrap_memory_skip", reason="already_loaded", store_id=store_id)
        return {"memory_query_needed": False}

    LOGGER.info(
        "bootstrap_memory_started",
        store_id=store_id,
        user_id=user_id,
        intent=intent,
    )

    user_prompt = latest_human_prompt(
        state.get("messages", []),
        state.get("user_prompt", ""),
    )

    # Build a focused semantic query from the user prompt
    try:
        memory_query = await build_memory_query(
            user_prompt,
            current_goal=state.get("goal") or user_prompt,
        )
    except Exception:
        memory_query = user_prompt[:200]

    user_prefs_raw: list[dict] = []
    store_knowledge_raw: list[dict] = []
    multi_store_raw: list[dict] = []

    # --- User memory: always load ---
    try:
        user_prefs_raw = await _bootstrap_user_memory(user_id, memory_query)
    except Exception as exc:
        LOGGER.warning("bootstrap_user_memory_failed", error=str(exc))

    # --- Store memory: skip for pure live_data intent ---
    if intent != "live_data":
        try:
            store_knowledge_raw = await _bootstrap_store_memory(store_id, memory_query, intent)
        except Exception as exc:
            LOGGER.warning("bootstrap_store_memory_failed", error=str(exc))

    # --- Multi-store memory: only for mixed/memory/general ---
    if intent in ("memory", "mixed", "general"):
        try:
            multi_store_raw = _cap_results(
                await search_multi_store_memory(user_id, memory_query, top_k=_BOOTSTRAP_CAP_MULTI)
            )[:_BOOTSTRAP_CAP_MULTI]
        except Exception as exc:
            LOGGER.warning("bootstrap_multi_store_memory_failed", error=str(exc))

    # Summarize for LLM context injection
    user_prefs_summary = await summarize_memory_results(user_prefs_raw, memory_query)
    store_knowledge_summary = await summarize_memory_results(store_knowledge_raw, memory_query)
    multi_store_summary = await summarize_memory_results(multi_store_raw, memory_query)

    LOGGER.info(
        "bootstrap_memory_completed",
        store_id=store_id,
        intent=intent,
        user_memories=len(user_prefs_raw),
        store_memories=len(store_knowledge_raw),
        multi_store_memories=len(multi_store_raw),
    )

    return {
        "user_memory_loaded": True,
        "store_memory_loaded": True,
        "multi_store_memory_loaded": True,
        "memory_query_needed": False,
        "user_preferences": {
            "raw": user_prefs_raw,
            "summary": user_prefs_summary,
            "source": "bootstrap",
        },
        "store_knowledge": {
            "raw": store_knowledge_raw,
            "summary": store_knowledge_summary,
            "source": "bootstrap",
        },
        "multi_store_knowledge": {
            "raw": multi_store_raw,
            "summary": multi_store_summary,
            "source": "bootstrap",
        },
    }