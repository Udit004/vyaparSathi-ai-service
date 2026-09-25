"""
app/agent/tools/memory/search.py
=================================
search_memory — agent-controlled long-term memory search tool.

The LLM calls this tool when bootstrap memory is insufficient and it
needs to retrieve historical decisions, episodic events, store patterns,
or older preferences from Pinecone.

SECURITY: user_id / store_id are injected from trusted LangGraph state
via a LangChain RunnableConfig; the LLM can ONLY supply:
    - query
    - memory_types  (filter by taxonomy)
    - date_from / date_to  (temporal filter on event_time)
    - top_k  (capped at MAX_RESULTS_PER_MEMORY_SEARCH)

This ensures no cross-user / cross-store leakage.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Any, Optional, List

import structlog
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.agent.memory import (
    _query_memory_vectors,
    _cap_results,
    summarize_memory_results,
)

LOGGER = structlog.get_logger("vyaparsathi.ai.memory.search_tool")

# ---------------------------------------------------------------------------
# Budget constants — configurable via env or settings in the future
# ---------------------------------------------------------------------------
MAX_MEMORY_SEARCH_CALLS = 3
MAX_RESULTS_PER_MEMORY_SEARCH = 5

# Memory types recognized by the taxonomy
VALID_MEMORY_TYPES = {
    "user_preference",
    "user_fact",
    "store_fact",
    "store_pattern",
    "episodic_event",
    "decision",
    "assistant_recommendation",
    # Legacy types from the old retriever
    "user_preference",   # maps to same
    "store_memory",      # old schema
    "multi_store_memory",
}

# Bootstrap types that should normally be available without a search call
BOOTSTRAP_TYPES = {"user_preference", "user_fact", "store_fact"}


class SearchMemoryInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "What to search for. Use a concise phrase describing the memory you need. "
            "Examples: 'rice inventory decision before Diwali', "
            "'user language preference', 'store restock patterns'."
        ),
    )
    memory_types: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of memory types to filter by. "
            "Valid values: user_preference, user_fact, store_fact, store_pattern, "
            "episodic_event, decision, assistant_recommendation. "
            "Leave empty to search all types."
        ),
    )
    date_from: Optional[str] = Field(
        default=None,
        description=(
            "Optional ISO-8601 date string (YYYY-MM-DD) for the start of an event_time range. "
            "Use when the user refers to a specific time period like 'last week' or 'in September'."
        ),
    )
    date_to: Optional[str] = Field(
        default=None,
        description=(
            "Optional ISO-8601 date string (YYYY-MM-DD) for the end of an event_time range."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=MAX_RESULTS_PER_MEMORY_SEARCH,
        description="Maximum number of memories to retrieve. Capped at 5.",
    )
    scope: str = Field(
        default="store",
        description=(
            "Which memory scope to search: 'user' for user-level memories, "
            "'store' for store-specific memories, or 'all' for both."
        ),
    )


def _rerank(results: list[dict], query: str) -> list[dict]:
    """
    Simple deterministic reranker combining semantic score, importance,
    confidence, and recency. Different memory types decay at different rates.
    """
    now = datetime.datetime.now(datetime.timezone.utc)

    def _score(r: dict) -> float:
        meta = r.get("metadata", {})
        semantic = float(r.get("score", 0.0))
        importance = float(meta.get("importance", 0.5))
        confidence = float(meta.get("confidence", 0.5))
        mem_type = meta.get("memory_type", "")

        # Recency factor — episodic/decision memories decay faster than preferences
        created_raw = meta.get("created_at") or meta.get("updated_at")
        recency = 0.5  # neutral default
        if created_raw:
            try:
                created = datetime.datetime.fromisoformat(str(created_raw).replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=datetime.timezone.utc)
                age_days = (now - created).days
                # Preferences decay slowly; episodic/decisions decay faster
                if mem_type in ("user_preference", "user_fact", "store_fact"):
                    recency = max(0.1, 1.0 - age_days / 365)
                elif mem_type in ("episodic_event",):
                    recency = max(0.1, 1.0 - age_days / 90)
                else:
                    recency = max(0.1, 1.0 - age_days / 180)
            except Exception:
                pass

        # Weighted composite
        return 0.5 * semantic + 0.2 * importance + 0.15 * confidence + 0.15 * recency

    return sorted(results, key=_score, reverse=True)


def _format_memory_for_llm(r: dict) -> dict:
    """Convert a raw Pinecone result into a clean LLM-readable dict."""
    meta = r.get("metadata", {})
    return {
        "memory_id": r.get("id", ""),
        "memory_type": meta.get("memory_type", "unknown"),
        "content": r.get("text") or meta.get("text") or "",
        "source_role": meta.get("source_role", "unknown"),
        "confidence": meta.get("confidence"),
        "importance": meta.get("importance"),
        "event_time": meta.get("event_time"),
        "valid_from": meta.get("valid_from"),
        "status": meta.get("status", "active"),
    }


@tool("search_memory", args_schema=SearchMemoryInput)
async def search_memory(
    query: str,
    memory_types: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    top_k: int = 5,
    scope: str = "store",
    config: RunnableConfig = None,
) -> dict[str, Any]:
    """
    Search long-term semantic memory (Pinecone) for historical decisions, events,
    preferences, and store patterns.

    Use this when:
    - The user asks about a past decision or historical event.
    - You need historical store patterns not available from live tools.
    - The user explicitly references "last time", "we decided", "previously", etc.
    - Bootstrap memory context is insufficient for the current question.

    DO NOT use this for current live data (stock, sales, prices) — use domain tools for that.
    DO NOT use this for simple greetings or questions answerable from live tools.
    """
    # Resolve identity from trusted runtime config (not from LLM input)
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = configurable.get("user_id", "")
    store_id = configurable.get("store_id", "")

    LOGGER.info(
        "memory_search_started",
        query=query[:80],
        memory_types=memory_types,
        scope=scope,
        user_id=user_id,
        store_id=store_id,
    )

    if not user_id and not store_id:
        LOGGER.warning("memory_search_no_identity")
        return {"found": False, "results": [], "error": "Identity not available in runtime context."}

    top_k = min(top_k, MAX_RESULTS_PER_MEMORY_SEARCH)
    all_results: list[dict] = []

    # Build base filter — only retrieve active memories
    def _build_filter(base: dict) -> dict:
        f = {**base, "status": "active"}
        if memory_types:
            # Filter to valid requested types
            valid = [t for t in memory_types if t in VALID_MEMORY_TYPES]
            if valid:
                f["memory_type"] = {"$in": valid}
        return f

    # --- User scope ---
    if scope in ("user", "all") and user_id:
        user_filter = _build_filter({"user_id": str(user_id), "scope": "user"})
        try:
            user_res = await _query_memory_vectors(user_filter, query, top_k=top_k)
            all_results.extend(user_res)
        except Exception as exc:
            LOGGER.warning("memory_search_user_scope_failed", error=str(exc))

    # --- Store scope ---
    if scope in ("store", "all") and store_id:
        store_filter = _build_filter({"store_id": str(store_id), "scope": "store"})
        try:
            store_res = await _query_memory_vectors(store_filter, query, top_k=top_k)
            all_results.extend(store_res)
        except Exception as exc:
            LOGGER.warning("memory_search_store_scope_failed", error=str(exc))

    # --- Also search legacy types if new schema gives nothing ---
    if not all_results and store_id:
        legacy_filter: dict[str, Any] = {
            "memory_type": {"$in": ["store_memory", "user_preference"]},
        }
        if store_id:
            legacy_filter["store_id"] = str(store_id)
        elif user_id:
            legacy_filter["user_id"] = str(user_id)
        try:
            legacy_res = await _query_memory_vectors(legacy_filter, query, top_k=top_k)
            all_results.extend(legacy_res)
        except Exception:
            pass

    # --- Temporal filtering on event_time ---
    if (date_from or date_to) and all_results:
        def _in_range(r: dict) -> bool:
            et = r.get("metadata", {}).get("event_time")
            if not et:
                return True  # include if no event_time
            try:
                t = datetime.date.fromisoformat(str(et)[:10])
                if date_from and t < datetime.date.fromisoformat(date_from):
                    return False
                if date_to and t > datetime.date.fromisoformat(date_to):
                    return False
            except Exception:
                pass
            return True
        all_results = [r for r in all_results if _in_range(r)]

    if not all_results:
        LOGGER.info("memory_search_no_results", query=query[:60])
        return {"found": False, "results": []}

    # Deduplicate by id
    seen_ids: set[str] = set()
    unique: list[dict] = []
    for r in all_results:
        rid = r.get("id", "")
        if rid and rid not in seen_ids:
            seen_ids.add(rid)
            unique.append(r)
        elif not rid:
            unique.append(r)

    # Rerank and cap
    reranked = _rerank(unique, query)[:top_k]
    formatted = [_format_memory_for_llm(r) for r in reranked]

    LOGGER.info(
        "memory_search_completed",
        query=query[:60],
        result_count=len(formatted),
    )

    return {
        "found": True,
        "results": formatted,
        "total_found": len(formatted),
    }
