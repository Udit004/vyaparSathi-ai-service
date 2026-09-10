"""
app/agent/checkpointer.py
=========================
MongoDB-backed LangGraph checkpointer for short-term cross-session memory.

Import compatibility:
    langgraph-checkpoint-mongodb < 0.2  → langgraph.checkpoint.mongodb.aio
    langgraph-checkpoint-mongodb >= 0.2 → langgraph_checkpoint_mongodb

The checkpointer is initialized once at application startup and shared
across all agent runs. It is injected into the compiled graph at
creation time via graph.compile(checkpointer=...).
"""

from __future__ import annotations

# --- Version-agnostic import ---
try:
    # langgraph-checkpoint-mongodb >= 0.2.0 (installed on Render)
    from langgraph_checkpoint_mongodb import AsyncMongoDBSaver
except ImportError:
    # langgraph-checkpoint-mongodb <= 0.1.x (legacy local install)
    from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver  # type: ignore[no-redef]

from motor.motor_asyncio import AsyncIOMotorClient
from app.config.settings import get_settings

import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.checkpointer")

# Module-level singleton — initialized once during app startup
_checkpointer: AsyncMongoDBSaver | None = None


async def get_checkpointer() -> AsyncMongoDBSaver:
    """
    Return the singleton MongoDB checkpointer, creating it if needed.

    The checkpointer stores the full LangGraph state snapshot after every
    node execution, keyed by:
        thread_id  = "{user_id}:{store_id}"
        checkpoint_ns = "" (default)

    This means each (user, store) pair has one persistent memory thread
    that survives across HTTP requests.

    Call once during FastAPI startup via lifespan and reuse the instance.
    """
    global _checkpointer

    if _checkpointer is None:
        settings = get_settings()
        client = AsyncIOMotorClient(settings.mongo_url)

        _checkpointer = AsyncMongoDBSaver(
            client=client,
            db_name=settings.mongo_db_name,
            checkpoint_collection_name="agent_checkpoints",
            writes_collection_name="agent_checkpoint_writes",
        )
        LOGGER.info(
            "checkpointer_initialized",
            db=settings.mongo_db_name,
            collection="agent_checkpoints",
        )

    return _checkpointer


async def close_checkpointer() -> None:
    """
    Close the checkpointer's internal MongoDB client.
    Call during FastAPI shutdown via lifespan.
    """
    global _checkpointer
    if _checkpointer is not None:
        _checkpointer.client.close()
        _checkpointer = None
        LOGGER.info("checkpointer_closed")
