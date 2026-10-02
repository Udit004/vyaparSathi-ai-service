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

import asyncio
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
from app.agent.memory.redis_cache import (
    get_cached_user_memory, set_cached_user_memory,
    get_cached_store_memory, set_cached_store_memory,
    get_cached_multi_memory, set_cached_multi_memory,
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
    """Fetch user preferences — Redis first, Pinecone on miss."""
    cached = await get_cached_user_memory(user_id)
    if cached is not None:
        LOGGER.debug("user_memory_cache_hit", user_id=user_id)
        return cached

    results = await search_user_memory(user_id, query, top_k=_BOOTSTRAP_CAP_USER)
    if not results:
        results = await search_user_memory(
            user_id,
            "user preference language tone detail level business goals",
            top_k=_BOOTSTRAP_CAP_USER,
        )
    final = _cap_results(results)[:_BOOTSTRAP_CAP_USER]
    await set_cached_user_memory(user_id, final)
    return final


async def _bootstrap_store_memory(store_id: str, query: str, intent: str) -> list[dict]:
    """Fetch store facts/patterns — Redis first, Pinecone on miss."""
    cached = await get_cached_store_memory(store_id)
    if cached is not None:
        LOGGER.debug("store_memory_cache_hit", store_id=store_id)
        return cached

    results = await search_store_memory(store_id, query, top_k=_BOOTSTRAP_CAP_STORE)
    if not results:
        results = await search_store_memory(
            store_id,
            "store facts supplier rules restock schedule inventory patterns",
            top_k=_BOOTSTRAP_CAP_STORE,
        )
    if intent not in ("memory", "mixed"):
        results = _filter_relevant_results(results)
    final = _cap_results(results)[:_BOOTSTRAP_CAP_STORE]
    await set_cached_store_memory(store_id, final)
    return final


async def memory_query_node(state: VyaparAgentState) -> Dict[str, Any]:
    """
    Bootstrap Memory Node — Redis-cached parallel edition.

    On cache HIT  (Redis): <100ms  (skips Pinecone entirely)
    On cache MISS (Redis): ~8-10s  (parallel Pinecone fetch, then writes Redis)
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

    if intent == "greeting":
        LOGGER.debug("bootstrap_memory_skip", reason="greeting_intent", store_id=store_id)
        return {"memory_query_needed": False}

    # live_data: only user prefs needed (language/tone). Skip store+multi search.
    skip_store_memory = intent == "live_data"
    skip_multi_store = intent not in ("memory", "mixed", "general")

    LOGGER.info(
        "bootstrap_memory_started",
        store_id=store_id,
        user_id=user_id,
        intent=intent,
        skip_store=skip_store_memory,
        skip_multi=skip_multi_store,
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

    # ── PARALLEL: fetch all three sources at once ───────────────────────────
    async def _safe_user_mem():
        try:
            return await _bootstrap_user_memory(user_id, memory_query)
        except Exception as exc:
            LOGGER.warning("bootstrap_user_memory_failed", error=str(exc))
            return []

    async def _safe_store_mem():
        if skip_store_memory:
            return []
        try:
            return await _bootstrap_store_memory(store_id, memory_query, intent)
        except Exception as exc:
            LOGGER.warning("bootstrap_store_memory_failed", error=str(exc))
            return []

    async def _safe_multi_mem():
        if skip_multi_store:
            return []
        try:
            cached = await get_cached_multi_memory(user_id)
            if cached is not None:
                LOGGER.debug("multi_memory_cache_hit", user_id=user_id)
                return cached
                
            results = _cap_results(
                await search_multi_store_memory(user_id, memory_query, top_k=_BOOTSTRAP_CAP_MULTI)
            )[:_BOOTSTRAP_CAP_MULTI]
            
            await set_cached_multi_memory(user_id, results)
            return results
        except Exception as exc:
            LOGGER.warning("bootstrap_multi_store_memory_failed", error=str(exc))
            return []

    user_prefs_raw, store_knowledge_raw, multi_store_raw = await asyncio.gather(
        _safe_user_mem(),
        _safe_store_mem(),
        _safe_multi_mem(),
    )

    # ── PARALLEL: summarize all three at once ───────────────────────────────
    user_prefs_summary, store_knowledge_summary, multi_store_summary = await asyncio.gather(
        summarize_memory_results(user_prefs_raw, memory_query),
        summarize_memory_results(store_knowledge_raw, memory_query),
        summarize_memory_results(multi_store_raw, memory_query),
    )

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