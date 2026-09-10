"""
app/agent/checkpointer.py
=========================
MongoDB-backed LangGraph checkpointer for short-term cross-session memory.

Uses ``langgraph-checkpoint-mongodb`` which wraps the motor async client
and provides a proper AsyncMongoDBSaver compatible with LangGraph's
StateGraph. Checkpoints are keyed by thread_id = "{user_id}:{store_id}".

The checkpointer is initialized once at application startup and shared
across all agent runs. It is injected into the compiled graph at
creation time via graph.compile(checkpointer=...).
"""

from __future__ import annotations

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
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
        # Reuse the same connection string as the rest of the app
        client = AsyncIOMotorClient(settings.mongo_url)

        _checkpointer = AsyncMongoDBSaver(
            client=client,
            db_name=settings.mongo_db_name,
            # Checkpoints land in a dedicated collection
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
        # The underlying client is motor — call close on it
        _checkpointer.client.close()
        _checkpointer = None
        LOGGER.info("checkpointer_closed")
