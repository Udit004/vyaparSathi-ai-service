"""
app/agent/memory/redis_cache.py
================================
Multi-Tier Redis Cache Layer for VyaparSathi Memory System.

Architecture:
1. Fast Query-Level Cache: Caches exact or normalized semantic memory search queries.
2. In-Memory Store & User Memory Bank: Keeps active store facts, customer notes, and user preferences
   in Redis for sub-millisecond retrieval before touching Pinecone vector DB.
3. Write-Through & Invalidation: Immediate caching of new memories and auto-invalidation on updates.
"""

from __future__ import annotations

import json
import hashlib
import re
import structlog
from typing import Any, List, Optional

from app.config.redis import get_redis

LOGGER = structlog.get_logger("vyaparsathi.ai.memory.cache")

_QUERY_CACHE_TTL = 600      # 10 minutes
_USER_MEM_TTL = 900         # 15 minutes
_STORE_MEM_TTL = 900        # 15 minutes
_MULTI_MEM_TTL = 1200       # 20 minutes


def _normalize_query_key(query: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", query.lower()).strip()
    return hashlib.md5(cleaned.encode("utf-8")).hexdigest()[:16]


def _user_key(user_id: str) -> str:
    return f"memory:user:{user_id}"


def _store_key(store_id: str) -> str:
    return f"memory:store:{store_id}"


def _multi_key(user_id: str) -> str:
    return f"memory:multistore:{user_id}"


def _query_key(scope: str, target_id: str, query: str) -> str:
    return f"memory:query:{scope}:{target_id}:{_normalize_query_key(query)}"


async def get_cached_query_memory(scope: str, target_id: str, query: str) -> Optional[List[dict]]:
    """Retrieve cached search results for an identical or normalized memory query."""
    try:
        redis = await get_redis()
        if not redis:
            return None
        key = _query_key(scope, target_id, query)
        raw = await redis.get(key)
        if raw:
            data = json.loads(raw)
            LOGGER.info("redis_memory_query_cache_hit", scope=scope, target_id=target_id, query=query[:40])
            return data
    except Exception as exc:
        LOGGER.warning("redis_memory_query_get_error", error=str(exc))
    return None


async def set_cached_query_memory(
    scope: str,
    target_id: str,
    query: str,
    results: List[dict],
    ttl: int = _QUERY_CACHE_TTL
) -> None:
    """Store search results for a specific memory query in Redis."""
    try:
        redis = await get_redis()
        if not redis or not results:
            return
        key = _query_key(scope, target_id, query)
        await redis.set(key, json.dumps(results), ex=ttl)
        LOGGER.debug("redis_memory_query_cache_set", scope=scope, target_id=target_id, count=len(results))
    except Exception as exc:
        LOGGER.warning("redis_memory_query_set_error", error=str(exc))


async def get_cached_user_memory(user_id: str) -> Optional[List[dict]]:
    """Retrieve full cached user memory list."""
    try:
        redis = await get_redis()
        if not redis:
            return None
        raw = await redis.get(_user_key(user_id))
        if raw:
            LOGGER.debug("redis_user_memory_hit", user_id=user_id)
            return json.loads(raw)
    except Exception as exc:
        LOGGER.warning("redis_user_memory_get_error", error=str(exc))
    return None


async def set_cached_user_memory(user_id: str, results: List[dict]) -> None:
    """Cache active user memories in Redis."""
    try:
        redis = await get_redis()
        if not redis:
            return
        await redis.set(_user_key(user_id), json.dumps(results), ex=_USER_MEM_TTL)
        LOGGER.debug("redis_user_memory_set", user_id=user_id, count=len(results))
    except Exception as exc:
        LOGGER.warning("redis_user_memory_set_error", error=str(exc))


async def get_cached_store_memory(store_id: str) -> Optional[List[dict]]:
    """Retrieve full cached store memory list."""
    try:
        redis = await get_redis()
        if not redis:
            return None
        raw = await redis.get(_store_key(store_id))
        if raw:
            LOGGER.debug("redis_store_memory_hit", store_id=store_id)
            return json.loads(raw)
    except Exception as exc:
        LOGGER.warning("redis_store_memory_get_error", error=str(exc))
    return None


async def set_cached_store_memory(store_id: str, results: List[dict]) -> None:
    """Cache active store memories in Redis."""
    try:
        redis = await get_redis()
        if not redis:
            return
        await redis.set(_store_key(store_id), json.dumps(results), ex=_STORE_MEM_TTL)
        LOGGER.debug("redis_store_memory_set", store_id=store_id, count=len(results))
    except Exception as exc:
        LOGGER.warning("redis_store_memory_set_error", error=str(exc))


async def get_cached_multi_memory(user_id: str) -> Optional[List[dict]]:
    """Retrieve multi-store cached memory."""
    try:
        redis = await get_redis()
        if not redis:
            return None
        raw = await redis.get(_multi_key(user_id))
        if raw:
            LOGGER.debug("redis_multi_memory_hit", user_id=user_id)
            return json.loads(raw)
    except Exception as exc:
        LOGGER.warning("redis_multi_memory_get_error", error=str(exc))
    return None


async def set_cached_multi_memory(user_id: str, results: List[dict]) -> None:
    """Cache multi-store memory."""
    try:
        redis = await get_redis()
        if not redis:
            return
        await redis.set(_multi_key(user_id), json.dumps(results), ex=_MULTI_MEM_TTL)
        LOGGER.debug("redis_multi_memory_set", user_id=user_id, count=len(results))
    except Exception as exc:
        LOGGER.warning("redis_multi_memory_set_error", error=str(exc))


async def append_live_memory_to_cache(scope: str, target_id: str, memory_dict: dict) -> None:
    """
    Append a newly formed memory directly into the active Redis cache.
    Ensures that immediate follow-up turns or queries find the memory in <2ms.
    """
    try:
        redis = await get_redis()
        if not redis:
            return
        key = _store_key(target_id) if scope == "store" else _user_key(target_id)
        raw = await redis.get(key)
        memories = json.loads(raw) if raw else []

        # Avoid duplicates
        mem_text = memory_dict.get("content") or memory_dict.get("text", "")
        if not any((m.get("content") or m.get("text")) == mem_text for m in memories):
            memories.insert(0, memory_dict)
            ttl = _STORE_MEM_TTL if scope == "store" else _USER_MEM_TTL
            await redis.set(key, json.dumps(memories[:30]), ex=ttl)
            LOGGER.info("redis_live_memory_appended", scope=scope, target_id=target_id, text=mem_text[:40])
    except Exception as exc:
        LOGGER.warning("redis_append_live_memory_error", error=str(exc))


def search_in_memory_cache(
    memories: List[dict],
    query: str,
    memory_types: Optional[List[str]] = None,
    top_k: int = 5
) -> List[dict]:
    """
    High-speed in-memory keyword & token relevance search across cached Redis memory records.
    Filters by memory_types if provided.
    Supports wildcard ('*', 'all', '') queries to list cached memories.
    """
    if not memories:
        return []

    clean_query = (query or "").strip().lower()

    # Wildcard or list-all queries
    if not clean_query or clean_query in ("*", "all", "everything", "list", "show all", "get all"):
        results = []
        for item in memories:
            if memory_types:
                item_type = (item.get("metadata", {}).get("memory_type") or item.get("memory_type", ""))
                if item_type and item_type not in memory_types:
                    continue
            results.append({**item, "source": "redis_cache", "score": 1.0})
        return results[:top_k]

    query_tokens = [w for w in re.sub(r"[^a-zA-Z0-9\s]", " ", clean_query).split() if len(w) > 2]
    if not query_tokens:
        # Fallback to returning recent memories if query tokens are short (e.g., abbreviations or digits)
        query_tokens = [w for w in clean_query.split() if w]
        if not query_tokens:
            return memories[:top_k]

    scored_results = []
    for item in memories:
        if memory_types:
            item_type = (item.get("metadata", {}).get("memory_type") or item.get("memory_type", ""))
            if item_type and item_type not in memory_types:
                continue

        content = (item.get("content") or item.get("text") or "").lower()
        if not content:
            continue

        score = 0.0
        # Exact substring boost
        if clean_query in content:
            score += 3.0

        # Token matching
        matched_tokens = 0
        for token in query_tokens:
            if token in content:
                matched_tokens += 1
                score += 1.0

        if matched_tokens > 0 or score > 0:
            token_ratio = matched_tokens / max(1, len(query_tokens))
            final_score = score * (1.0 + token_ratio)
            scored_results.append((final_score, {**item, "source": "redis_cache", "score": min(0.98, 0.5 + 0.1 * final_score)}))

    scored_results.sort(key=lambda x: x[0], reverse=True)
    return [res[1] for res in scored_results[:top_k]]


async def invalidate_user_memory(user_id: str) -> None:
    try:
        redis = await get_redis()
        if redis:
            await redis.delete(_user_key(user_id))
            LOGGER.info("memory_cache_invalidated", scope="user", user_id=user_id)
    except Exception as exc:
        LOGGER.warning("memory_cache_invalidate_error", error=str(exc))


async def invalidate_store_memory(store_id: str) -> None:
    try:
        redis = await get_redis()
        if redis:
            await redis.delete(_store_key(store_id))
            LOGGER.info("memory_cache_invalidated", scope="store", store_id=store_id)
    except Exception as exc:
        LOGGER.warning("memory_cache_invalidate_error", error=str(exc))
