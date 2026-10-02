"""
app/agent/memory/redis_cache.py
================================
Redis-backed cache layer for Pinecone memory search results.

The memory_query node calls Pinecone (via Gemini embeddings) which takes ~8-10s
per query. By caching the results in Redis, subsequent requests for the same
user/store within the TTL window skip Pinecone entirely -- reducing memory load
from ~25s to <100ms on cache hit.

Cache Keys
----------
  memory:user:{user_id}          -> user preference results  (TTL: 5 min)
  memory:store:{store_id}        -> store knowledge results  (TTL: 5 min)
  memory:multistore:{user_id}    -> multi-store results      (TTL: 10 min)
"""

from __future__ import annotations

import json
import structlog
from typing import Any

from app.config.redis import get_redis

LOGGER = structlog.get_logger("vyaparsathi.ai.memory.cache")

_USER_MEM_TTL = 300
_STORE_MEM_TTL = 300
_MULTI_MEM_TTL = 600


def _user_key(user_id: str) -> str:
    return f"memory:user:{user_id}"

def _store_key(store_id: str) -> str:
    return f"memory:store:{store_id}"

def _multi_key(user_id: str) -> str:
    return f"memory:multistore:{user_id}"


async def get_cached_user_memory(user_id: str) -> list | None:
    try:
        redis = await get_redis()
        if not redis:
            return None
        raw = await redis.get(_user_key(user_id))
        if raw:
            LOGGER.debug("memory_cache_hit", scope="user", user_id=user_id)
            return json.loads(raw)
    except Exception as exc:
        LOGGER.warning("memory_cache_get_error", scope="user", error=str(exc))
    return None


async def set_cached_user_memory(user_id: str, results: list) -> None:
    try:
        redis = await get_redis()
        if not redis:
            return
        await redis.set(_user_key(user_id), json.dumps(results), ex=_USER_MEM_TTL)
        LOGGER.debug("memory_cache_set", scope="user", user_id=user_id, count=len(results))
    except Exception as exc:
        LOGGER.warning("memory_cache_set_error", scope="user", error=str(exc))


async def get_cached_store_memory(store_id: str) -> list | None:
    try:
        redis = await get_redis()
        if not redis:
            return None
        raw = await redis.get(_store_key(store_id))
        if raw:
            LOGGER.debug("memory_cache_hit", scope="store", store_id=store_id)
            return json.loads(raw)
    except Exception as exc:
        LOGGER.warning("memory_cache_get_error", scope="store", error=str(exc))
    return None


async def set_cached_store_memory(store_id: str, results: list) -> None:
    try:
        redis = await get_redis()
        if not redis:
            return
        await redis.set(_store_key(store_id), json.dumps(results), ex=_STORE_MEM_TTL)
        LOGGER.debug("memory_cache_set", scope="store", store_id=store_id, count=len(results))
    except Exception as exc:
        LOGGER.warning("memory_cache_set_error", scope="store", error=str(exc))


async def get_cached_multi_memory(user_id: str) -> list | None:
    try:
        redis = await get_redis()
        if not redis:
            return None
        raw = await redis.get(_multi_key(user_id))
        if raw:
            LOGGER.debug("memory_cache_hit", scope="multi", user_id=user_id)
            return json.loads(raw)
    except Exception as exc:
        LOGGER.warning("memory_cache_get_error", scope="multi", error=str(exc))
    return None


async def set_cached_multi_memory(user_id: str, results: list) -> None:
    try:
        redis = await get_redis()
        if not redis:
            return
        await redis.set(_multi_key(user_id), json.dumps(results), ex=_MULTI_MEM_TTL)
        LOGGER.debug("memory_cache_set", scope="multi", user_id=user_id, count=len(results))
    except Exception as exc:
        LOGGER.warning("memory_cache_set_error", scope="multi", error=str(exc))


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
