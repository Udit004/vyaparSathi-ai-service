"""
app/agent/tools/memory/search.py
=================================
Agent-controlled multi-tier memory tools for both Store-Level Knowledge & Owner-Level Personalization:
1. `search_memory` — High-speed hybrid memory retrieval (Store facts, customer rules, owner preferences).
2. `get_owner_goals_and_preferences` — Dedicated tool for retrieving owner targets, business goals & preferences.
3. `set_owner_goal_or_preference` — Dedicated tool for recording/updating owner goals, targets & preferences.
4. `remember_store_fact` — Proactive store business rule & customer fact recording.

Architecture:
- Tier 1: Fast Redis cache (query hash + in-memory store/user memory bank <2ms)
- Tier 2: Pinecone vector database with semantic embeddings
- Tier 3: Temporal date filtering & multi-factor reranking
- Tier 4: Write-through & auto-caching to Redis

SECURITY: user_id / store_id are injected from trusted runtime context.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Any, Optional, List, Dict

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
    set_cached_user_memory,
    append_live_memory_to_cache,
    search_in_memory_cache,
)

LOGGER = structlog.get_logger("vyaparsathi.ai.memory.search_tool")

MAX_RESULTS_PER_MEMORY_SEARCH = 5
MAX_MEMORY_SEARCH_CALLS = 3

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
        default="*",
        description=(
            "What to search for in store/owner memory (e.g. 'monthly revenue target', 'customer Rahul credit', 'supplier Ramesh discount terms', or '*' to list all memories)."
        ),
    )
    user_id: Optional[str] = Field(
        default=None,
        description="The store owner / user ID (injected from session).",
    )
    store_id: Optional[str] = Field(
        default=None,
        description="The store ID (injected from session).",
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
        default="all",
        description="Scope to search: 'all' (both owner preferences & store facts), 'user' (owner preferences/goals), or 'store' (store facts/patterns).",
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
    user_id: Optional[str] = Field(
        default=None,
        description="The store owner / user ID (injected from session).",
    )
    store_id: Optional[str] = Field(
        default=None,
        description="The store ID (injected from session).",
    )


class GetOwnerGoalsInput(BaseModel):
    topic: Optional[str] = Field(
        default="business goals targets preferences",
        description="Specific aspect of owner profile to look up: e.g. 'sales targets', 'margin goals', 'communication style', or leave default for all goals.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="The store owner / user ID (injected from session).",
    )
    store_id: Optional[str] = Field(
        default=None,
        description="The store ID (injected from session).",
    )


class SetOwnerGoalInput(BaseModel):
    goal_or_preference: str = Field(
        ...,
        description=(
            "The specific business goal, sales target, or personal preference of the store owner. "
            "Examples: 'Monthly sales target is ₹5,00,000', 'Aim for 18% average profit margin', "
            "'Always alert me when stock is below 15 units', 'I prefer concise advice in Hindi'."
        ),
    )
    category: str = Field(
        default="business_goal",
        description="Type of memory: 'business_goal' (sales/profit target), 'personal_preference' (communication/alerts), or 'strategy' (expansion/focus area).",
    )
    importance: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Importance score from 0.1 to 1.0 (default 0.9).",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="The store owner / user ID (injected from session).",
    )
    store_id: Optional[str] = Field(
        default=None,
        description="The store ID (injected from session).",
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
    query: str = "*",
    user_id: Optional[str] = None,
    store_id: Optional[str] = None,
    memory_types: Optional[List[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    top_k: int = 5,
    scope: str = "all",
    config: RunnableConfig = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Search long-term memory for stored facts, customer notes, supplier terms, business goals, and owner preferences.
    Uses multi-tier memory: checks fast Redis cache first, then searches Pinecone vector database.

    Use this when:
    - The store owner asks about past agreements, customer credit notes, business goals, or supplier pricing terms.
    - You need to look up historical business rules or store patterns.
    - The owner says "what is my monthly target", "what did we decide about...", "who is our supplier for...", etc.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    LOGGER.info(
        "memory_search_invoked",
        query=str(query)[:60],
        scope=scope,
        user_id=user_id,
        store_id=store_id,
    )

    if not user_id and not store_id:
        return {"found": False, "results": [], "source": "none", "error": "Identity not available in runtime context."}

    top_k = min(top_k or 5, MAX_RESULTS_PER_MEMORY_SEARCH)
    target_id = store_id if (scope in ("store", "all") and store_id) else (user_id or store_id)
    clean_q = (query or "*").strip()

    # -------------------------------------------------------------
    # TIER 1: Check Redis Query Cache (if specific query & no date filters)
    # -------------------------------------------------------------
    if clean_q not in ("*", "all", "") and not date_from and not date_to:
        cached_query_results = await get_cached_query_memory(scope, str(target_id), clean_q)
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
        fast_matches = search_in_memory_cache(redis_memories, clean_q, memory_types=memory_types, top_k=top_k)
        if fast_matches and len(fast_matches) >= 1:
            LOGGER.info("memory_search_tier2_hit_redis_bank", count=len(fast_matches))
            if clean_q not in ("*", "all", ""):
                await set_cached_query_memory(scope, str(target_id), clean_q, fast_matches)
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
    vector_query = clean_q if clean_q not in ("*", "all", "") else "store business goals revenue targets preferences rules"

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
            store_res = await _query_memory_vectors(store_filter, vector_query, top_k=top_k * 2)
            raw_results.extend(store_res)

        if scope in ("user", "all") and user_id:
            user_filter = _build_filter({"user_id": str(user_id), "scope": "user"})
            user_res = await _query_memory_vectors(user_filter, vector_query, top_k=top_k * 2)
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
    reranked = _rerank(time_filtered, vector_query)
    formatted = [_format_memory_for_llm(r, source="pinecone") for r in reranked[:top_k]]

    # -------------------------------------------------------------
    # TIER 4: Cache Pinecone Results in Redis for Future Turns
    # -------------------------------------------------------------
    if formatted and not date_from and not date_to:
        if clean_q not in ("*", "all", ""):
            await set_cached_query_memory(scope, str(target_id), clean_q, formatted)
        for item in formatted:
            await append_live_memory_to_cache(scope if scope != "all" else "store", str(target_id), item)

    LOGGER.info("memory_search_completed", total_found=len(formatted))
    return {
        "found": len(formatted) > 0,
        "results": formatted,
        "source": "pinecone" if formatted else ("redis_cache" if redis_memories else "none"),
        "count": len(formatted),
    }


@tool("get_owner_goals_and_preferences", args_schema=GetOwnerGoalsInput)
async def get_owner_goals_and_preferences(
    topic: Optional[str] = "business goals targets preferences",
    user_id: Optional[str] = None,
    store_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Retrieve the store owner's personal business goals, sales targets, operational focus,
    risk tolerance, and communication preferences.

    Combines both owner-level goals (user scope) and store-level targets/rules (store scope).
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    LOGGER.info("get_owner_goals_invoked", user_id=user_id, store_id=store_id, topic=topic)

    if not user_id and not store_id:
        return {"found": False, "owner_goals": [], "owner_preferences": [], "other_personal_notes": [], "message": "User identity not found in context."}

    all_raw_memories: list[dict] = []

    # 1. Fetch User Scope Memories (Redis -> Pinecone fallback)
    if user_id:
        u_mems = await get_cached_user_memory(str(user_id))
        if not u_mems:
            try:
                user_filter = {"user_id": str(user_id), "scope": "user", "status": "active"}
                user_res = await _query_memory_vectors(user_filter, topic or "goals preferences targets", top_k=6)
                u_mems = [_format_memory_for_llm(r, source="pinecone") for r in user_res]
                if u_mems:
                    await set_cached_user_memory(str(user_id), u_mems)
            except Exception as exc:
                LOGGER.warning("get_owner_goals_user_pinecone_error", error=str(exc))
                u_mems = []
        if u_mems:
            all_raw_memories.extend(u_mems)

    # 2. Fetch Store Scope Memories (Redis -> Pinecone fallback)
    if store_id:
        s_mems = await get_cached_store_memory(str(store_id))
        if not s_mems:
            try:
                store_filter = {"store_id": str(store_id), "scope": "store", "status": "active"}
                store_res = await _query_memory_vectors(store_filter, topic or "store goals targets facts rules", top_k=6)
                s_mems = [_format_memory_for_llm(r, source="pinecone") for r in store_res]
                if s_mems:
                    await set_cached_store_memory(str(store_id), s_mems)
            except Exception as exc:
                LOGGER.warning("get_owner_goals_store_pinecone_error", error=str(exc))
                s_mems = []
        if s_mems:
            all_raw_memories.extend(s_mems)

    # Deduplicate memories by text content
    seen_texts = set()
    deduped_memories = []
    for m in all_raw_memories:
        text = (m.get("content") or m.get("text") or "").strip()
        if not text or text.lower() in seen_texts:
            continue
        seen_texts.add(text.lower())
        deduped_memories.append(m)

    goals = []
    preferences = []
    general = []

    for m in deduped_memories:
        text = m.get("content") or m.get("text") or ""
        text_lower = text.lower()
        m_type = m.get("memory_type", "")

        is_goal = any(kw in text_lower for kw in ("target", "goal", "aim", "lakh", "margin", "increase", "revenue", "growth", "sales volume", "%")) or m_type in ("decision", "store_pattern")
        is_pref = any(kw in text_lower for kw in ("prefer", "alert", "language", "lead time", "restock", "threshold", "discount", "rule")) or m_type == "user_preference"

        if is_goal and not ("lead time" in text_lower or "threshold" in text_lower):
            goals.append(text)
        elif is_pref or "lead time" in text_lower or "threshold" in text_lower:
            preferences.append(text)
        else:
            general.append(text)

    return {
        "found": len(goals) > 0 or len(preferences) > 0 or len(general) > 0,
        "owner_goals": goals,
        "owner_preferences": preferences,
        "other_personal_notes": general,
        "total_records": len(goals) + len(preferences) + len(general),
        "source": "redis_and_pinecone",
    }


@tool("set_owner_goal_or_preference", args_schema=SetOwnerGoalInput)
async def set_owner_goal_or_preference(
    goal_or_preference: str,
    category: str = "business_goal",
    importance: float = 0.9,
    user_id: Optional[str] = None,
    store_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Save or update a personal business goal, revenue target, profit margin aim, or operational preference
    for the store owner.

    Immediately caches in Redis for zero-latency personalization and persists to Pinecone.

    Use this when:
    - The owner states a target (e.g. "My goal is to achieve ₹5 Lakh sales this month").
    - The owner shares a preference (e.g. "I want to maintain at least 15% profit margin").
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    LOGGER.info("set_owner_goal_invoked", goal=goal_or_preference[:60], user_id=user_id, store_id=store_id)

    mem_type = "user_preference" if category == "personal_preference" else "user_fact"

    memory_record = {
        "content": goal_or_preference,
        "text": goal_or_preference,
        "memory_type": mem_type,
        "importance": importance,
        "confidence": 0.98,
        "scope": "user",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "active",
    }

    # 1. Immediately cache in Redis User Memory
    target_id = user_id or store_id
    if target_id:
        await append_live_memory_to_cache("user", str(target_id), memory_record)

    # 2. Persist to Pinecone vector DB
    try:
        mem_obj = ExtractedMemory(
            content=goal_or_preference,
            subject="Owner",
            source_role="user",
            evidence=goal_or_preference,
            memory_type=mem_type,
            user_id=str(user_id) if user_id else None,
            store_id=str(store_id) if store_id else None,
            confidence=0.98,
            importance=importance,
            status="active",
        )
        extra_meta = {
            "user_id": str(user_id) if user_id else "",
            "store_id": str(store_id) if store_id else "",
            "scope": "user",
            "status": "active",
        }
        await write_new_memory(mem_obj, extra_meta)
        LOGGER.info("owner_goal_persisted_to_pinecone")
    except Exception as exc:
        LOGGER.warning("owner_goal_pinecone_persist_error", error=str(exc))

    return {
        "success": True,
        "message": f"Successfully updated owner goal/preference: '{goal_or_preference}'",
        "record": memory_record,
    }


@tool("remember_store_fact", args_schema=RememberStoreFactInput)
async def remember_store_fact(
    content: str,
    memory_type: str = "store_fact",
    importance: float = 0.8,
    user_id: Optional[str] = None,
    store_id: Optional[str] = None,
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

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    LOGGER.info("remember_store_fact_invoked", content=content[:60], store_id=store_id, user_id=user_id)

    scope = "user" if memory_type in ("user_preference", "user_fact") else "store"
    target_id = user_id if scope == "user" else (store_id or user_id)

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
            content=content,
            subject="Store" if scope == "store" else "Owner",
            source_role="user",
            evidence=content,
            memory_type=memory_type,
            user_id=str(user_id) if user_id else None,
            store_id=str(store_id) if store_id else None,
            confidence=0.95,
            importance=importance,
            status="active",
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
