"""
app/agent/tools/memory/search.py
=================================
Agent-controlled multi-tier memory tools:
1. `search_memory` — High-speed hybrid memory retrieval:
   - Tier 1: Fast Redis cache (query hash + in-memory store memory bank)
   - Tier 2: Pinecone vector database with semantic embeddings
   - Tier 3: Temporal date filtering & multi-factor reranking
   - Tier 4: Auto-caches Pinecone results back to Redis for subsequent turns
2. `remember_store_fact` — Proactive memory recording:
   - Writes directly to Redis cache for sub-millisecond retrieval
   - Persists to Pinecone vector DB with structured metadata

SECURITY: user_id / store_id are injected from trusted runtime context.
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
from app.agent.memory.models import ExtractedMemory
from app.agent.memory.writer import write_new_memory
from app.agent.memory.redis_cache import (
    get_cached_query_memory,
    set_cached_query_memory,
    get_cached_store_memory,
    get_cached_user_memory,
    set_cached_store_memory,
    append_live_memory_to_cache,
    search_in_memory_cache,
)

LOGGER = structlog.get_logger("vyaparsathi.ai.memory.search_tool")

MAX_RESULTS_PER_MEMORY_SEARCH = 5

VALID_MEMORY_TYPES = {
    "user_preference",
    "user_fact",
    "store_fact",
    "store_pattern",
    "episodic_event",
    "decision",
    "assistant_recommendation",
}


class SearchMemoryInput(BaseModel):
    query: str = Field(
        ...,
        description=(
            "What to search for in store/owner memory. Use a concise phrase describing the memory you need. "
            "Examples: 'supplier Ramesh payment discount terms', "
            "'customer Rahul credit limit', 'preferred rice distributor', 'Diwali stock pattern'."
        ),
    )
    memory_types: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of memory types: user_preference, user_fact, store_fact, store_pattern, "
            "episodic_event, decision, assistant_recommendation. Leave empty to search all."
        ),
    )
    date_from: Optional[str] = Field(
        default=None,
        description="Optional ISO-8601 date string (YYYY-MM-DD) for start of event_time range.",
    )
    date_to: Optional[str] = Field(
        default=None,
        description="Optional ISO-8601 date string (YYYY-MM-DD) for end of event_time range.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=MAX_RESULTS_PER_MEMORY_SEARCH,
        description="Maximum number of memories to retrieve (1-5).",
    )
    scope: str = Field(
        default="store",
        description="Scope to search: 'store' for store facts/patterns, 'user' for owner preferences, or 'all'.",
    )


class RememberStoreFactInput(BaseModel):
    content: str = Field(
        ...,
        description=(
            "The specific fact, business rule, customer preference, or supplier term to remember. "
            "Examples: 'Supplier ABC gives 5% discount on cash orders above 10 boxes', "
            "'Do not give credit to customer XYZ until pending bill is cleared', "
            "'Store opens at 7 AM and closes at 10 PM daily'."
        ),
    )
    memory_type: str = Field(
        default="store_fact",
        description=(
            "Category of memory: 'store_fact' (business details/rules), 'store_pattern' (sales/ordering trends), "
            "'user_preference' (owner's personal choices), 'decision' (past management decisions)."
        ),
    )
    importance: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Importance score from 0.1 to 1.0 (default 0.8).",
    )


def _rerank(results: list[dict], query: str) -> list[dict]:
    """
    Deterministic reranker combining semantic score, importance,
    confidence, and recency decay.
    """
    now = datetime.datetime.now(datetime.timezone.utc)

    def _score(r: dict) -> float:
        meta = r.get("metadata", {})
        semantic = float(r.get("score", 0.0))
        importance = float(meta.get("importance", 0.5))
        confidence = float(meta.get("confidence", 0.5))
        mem_type = meta.get("memory_type", "")

        created_raw = meta.get("created_at") or meta.get("updated_at")
        recency = 0.5
        if created_raw:
            try:
                created = datetime.datetime.fromisoformat(str(created_raw).replace("Z", "+00:00"))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=datetime.timezone.utc)
                age_days = (now - created).days
                if mem_type in ("user_preference", "user_fact", "store_fact"):
                    recency = max(0.1, 1.0 - age_days / 365)
                elif mem_type in ("episodic_event",):
                    recency = max(0.1, 1.0 - age_days / 90)
                else:
                    recency = max(0.1, 1.0 - age_days / 180)
            except Exception:
                pass

        return 0.5 * semantic + 0.2 * importance + 0.15 * confidence + 0.15 * recency

    return sorted(results, key=_score, reverse=True)


def _format_memory_for_llm(r: dict, source: str = "pinecone") -> dict:
    meta = r.get("metadata", {})
    return {
        "memory_id": r.get("id") or r.get("memory_id", ""),
        "memory_type": meta.get("memory_type") or r.get("memory_type", "store_fact"),
        "content": r.get("text") or meta.get("text") or r.get("content", ""),
        "confidence": meta.get("confidence", 0.9),
        "importance": meta.get("importance", 0.8),
        "source": source,
    }


def _apply_temporal_filter(results: list[dict], date_from: Optional[str], date_to: Optional[str]) -> list[dict]:
    if not date_from and not date_to:
        return results

    filtered = []
    for r in results:
        meta = r.get("metadata", {})
        event_time_str = meta.get("event_time") or meta.get("created_at") or r.get("event_time")
        if not event_time_str:
            filtered.append(r)
            continue
        try:
            event_date = event_time_str[:10]
            if date_from and event_date < date_from:
                continue
            if date_to and event_date > date_to:
                continue
            filtered.append(r)
        except Exception:
            filtered.append(r)
    return filtered


@tool("search_memory", args_schema=SearchMemoryInput)
async def search_memory(
    query: str,
    memory_types: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    top_k: int = 5,
    scope: str = "store",
    config: RunnableConfig = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Search long-term memory for stored facts, customer notes, supplier terms, and past business decisions.
    Uses multi-tier memory: checks fast Redis cache first, then searches Pinecone vector database.

    Use this when:
    - The store owner asks about past agreements, customer credit notes, or supplier pricing terms.
    - You need to look up historical business rules or store patterns.
    - The owner says "what did we decide about...", "who is our supplier for...", etc.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = configurable.get("user_id") or kwargs.get("user_id", "")
    store_id = configurable.get("store_id") or kwargs.get("store_id", "")

    LOGGER.info(
        "memory_search_invoked",
        query=query[:60],
        scope=scope,
        user_id=user_id,
        store_id=store_id,
    )

    if not user_id and not store_id:
        return {"found": False, "results": [], "source": "none", "error": "Identity not available in runtime context."}

    top_k = min(top_k, MAX_RESULTS_PER_MEMORY_SEARCH)
    target_id = store_id if scope in ("store", "all") else user_id

    # -------------------------------------------------------------
    # TIER 1: Check Redis Query Cache (if no date filters)
    # -------------------------------------------------------------
    if not date_from and not date_to:
        cached_query_results = await get_cached_query_memory(scope, str(target_id), query)
        if cached_query_results:
            LOGGER.info("memory_search_tier1_hit_redis_query", count=len(cached_query_results))
            return {
                "found": True,
                "results": cached_query_results[:top_k],
                "source": "redis_cache",
                "count": len(cached_query_results[:top_k]),
            }

    # -------------------------------------------------------------
    # TIER 2: Check Redis Active Store/User Memory Bank
    # -------------------------------------------------------------
    redis_memories = []
    if scope in ("store", "all") and store_id:
        store_mems = await get_cached_store_memory(str(store_id))
        if store_mems:
            redis_memories.extend(store_mems)
    if scope in ("user", "all") and user_id:
        user_mems = await get_cached_user_memory(str(user_id))
        if user_mems:
            redis_memories.extend(user_mems)

    if redis_memories and not date_from and not date_to:
        fast_matches = search_in_memory_cache(redis_memories, query, memory_types=memory_types, top_k=top_k)
        if fast_matches and len(fast_matches) >= 1:
            LOGGER.info("memory_search_tier2_hit_redis_bank", count=len(fast_matches))
            await set_cached_query_memory(scope, str(target_id), query, fast_matches)
            return {
                "found": True,
                "results": fast_matches,
                "source": "redis_cache",
                "count": len(fast_matches),
            }

    # -------------------------------------------------------------
    # TIER 3: Pinecone Vector Database Semantic Embedding Search
    # -------------------------------------------------------------
    raw_results: list[dict] = []

    def _build_filter(base: dict) -> dict:
        f = {**base, "status": "active"}
        if memory_types:
            valid = [t for t in memory_types if t in VALID_MEMORY_TYPES]
            if valid:
                f["memory_type"] = {"$in": valid}
        return f

    try:
        if scope in ("store", "all") and store_id:
            store_filter = _build_filter({"store_id": str(store_id), "scope": "store"})
            store_res = await _query_memory_vectors(store_filter, query, top_k=top_k * 2)
            raw_results.extend(store_res)

        if scope in ("user", "all") and user_id:
            user_filter = _build_filter({"user_id": str(user_id), "scope": "user"})
            user_res = await _query_memory_vectors(user_filter, query, top_k=top_k * 2)
            raw_results.extend(user_res)
    except Exception as exc:
        LOGGER.warning("memory_search_pinecone_error", error=str(exc))

    # Deduplicate by memory id
    seen_ids = set()
    deduped = []
    for item in raw_results:
        mem_id = item.get("id") or item.get("memory_id")
        if mem_id and mem_id in seen_ids:
            continue
        if mem_id:
            seen_ids.add(mem_id)
        deduped.append(item)

    # Temporal filter
    time_filtered = _apply_temporal_filter(deduped, date_from, date_to)

    # Rerank
    reranked = _rerank(time_filtered, query)
    formatted = [_format_memory_for_llm(r, source="pinecone") for r in reranked[:top_k]]

    # -------------------------------------------------------------
    # TIER 4: Cache Pinecone Results in Redis for Future Turns
    # -------------------------------------------------------------
    if formatted and not date_from and not date_to:
        await set_cached_query_memory(scope, str(target_id), query, formatted)
        for item in formatted:
            await append_live_memory_to_cache(scope, str(target_id), item)

    LOGGER.info("memory_search_completed", total_found=len(formatted))
    return {
        "found": len(formatted) > 0,
        "results": formatted,
        "source": "pinecone" if formatted else "none",
        "count": len(formatted),
    }


@tool("remember_store_fact", args_schema=RememberStoreFactInput)
async def remember_store_fact(
    content: str,
    memory_type: str = "store_fact",
    importance: float = 0.8,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Explicitly remember and save a store fact, customer preference, supplier agreement,
    or business rule to long-term memory.

    Immediately caches the memory in Redis for instant recall and persists it to Pinecone.

    Use this when:
    - The store owner says "remember that...", "note down that...", "keep in mind that...".
    - The owner specifies a new customer credit rule, distributor payment term, or store policy.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = configurable.get("user_id") or kwargs.get("user_id", "")
    store_id = configurable.get("store_id") or kwargs.get("store_id", "")

    LOGGER.info("remember_store_fact_invoked", content=content[:60], store_id=store_id, user_id=user_id)

    scope = "user" if memory_type in ("user_preference", "user_fact") else "store"
    target_id = user_id if scope == "user" else store_id

    memory_record = {
        "content": content,
        "text": content,
        "memory_type": memory_type,
        "importance": importance,
        "confidence": 0.95,
        "scope": scope,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "active",
    }

    # 1. Immediately cache in Redis for 0ms follow-up recall
    if target_id:
        await append_live_memory_to_cache(scope, str(target_id), memory_record)

    # 2. Persist to Pinecone vector DB in background
    try:
        mem_obj = ExtractedMemory(
            text=content,
            memory_type=memory_type,
            user_id=str(user_id) if user_id else None,
            store_id=str(store_id) if store_id else None,
            confidence=0.95,
            importance=importance,
            scope=scope,
        )
        extra_meta = {
            "user_id": str(user_id) if user_id else "",
            "store_id": str(store_id) if store_id else "",
            "scope": scope,
            "status": "active",
        }
        await write_new_memory(mem_obj, extra_meta)
        LOGGER.info("remember_store_fact_persisted_to_pinecone")
    except Exception as exc:
        LOGGER.warning("remember_store_fact_pinecone_persist_error", error=str(exc))

    return {
        "success": True,
        "message": f"Successfully remembered and cached in store memory: '{content}'",
        "memory": memory_record,
    }
