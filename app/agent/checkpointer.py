"""
app/agent/checkpointer.py
=========================

MongoDB-backed LangGraph checkpointer for short-term cross-session memory.

Uses:

    langgraph-checkpoint-mongodb

Current API:

    from langgraph.checkpoint.mongodb import MongoDBSaver
    from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

The checkpointer is initialized once during FastAPI application startup
and shared across all agent runs.

The MongoDB client and checkpointer are application-level singletons.

The checkpointer is injected into the compiled LangGraph via:

    workflow.compile(checkpointer=checkpointer)
"""

from __future__ import annotations

import structlog

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

from app.config.settings import get_settings


LOGGER = structlog.get_logger(
    "vyaparsathi.ai.checkpointer"
)


# ---------------------------------------------------------------------------
# Application-level singletons
# ---------------------------------------------------------------------------

_mongo_client: MongoClient | None = None

_checkpointer: MongoDBSaver | None = None

_async_mongo_client: AsyncIOMotorClient | None = None

_async_checkpointer: AsyncMongoDBSaver | None = None


# ---------------------------------------------------------------------------
# Checkpointer initialization
# ---------------------------------------------------------------------------

def get_checkpointer() -> MongoDBSaver:
    """
    Return the application-wide MongoDB LangGraph checkpointer.

    The MongoDB client and checkpointer are created once and reused
    throughout the FastAPI application lifecycle.

    LangGraph checkpoints are separated using the invocation config:

        {
            "configurable": {
                "thread_id": "{user_id}:{store_id}:{chat_id}"
            }
        }

    Each logical chat gets its own ``chat_id`` (UUID). Starting a new chat
    creates a fresh ``thread_id`` so the agent starts with a clean context
    window. The chat history (user/assistant messages) is additionally
    persisted in the ``agent_chats`` and ``agent_chat_messages`` collections
    by ``app/services/chat_history_service.py``.

    Returns:
        MongoDBSaver:
            Initialized MongoDB-backed LangGraph checkpointer.
    """

    global _mongo_client
    global _checkpointer

    # Return existing singleton
    if _checkpointer is not None:
        return _checkpointer

    settings = get_settings()

    # -----------------------------------------------------------------------
    # Create MongoDB client
    # -----------------------------------------------------------------------

    _mongo_client = MongoClient(
        settings.mongo_url
    )

    # -----------------------------------------------------------------------
    # Create LangGraph MongoDB checkpointer
    # -----------------------------------------------------------------------

    _checkpointer = MongoDBSaver(
        client=_mongo_client,
        db_name=settings.mongo_db_name,
        checkpoint_collection_name="agent_checkpoints",
        writes_collection_name="agent_checkpoint_writes",
        ttl=settings.langgraph_checkpointer_ttl_seconds,
    )

    LOGGER.info(
        "checkpointer_initialized",
        db=settings.mongo_db_name,
        checkpoint_collection="agent_checkpoints",
        writes_collection="agent_checkpoint_writes",
    )

    return _checkpointer


def get_async_checkpointer() -> AsyncMongoDBSaver:
    """
    Return the application-wide async MongoDB LangGraph checkpointer.

    Async LangGraph execution methods such as ``astream_events`` require
    async checkpoint methods like ``aget_tuple``. The synchronous
    ``MongoDBSaver`` does not implement those methods.
    """

    global _async_mongo_client
    global _async_checkpointer

    if _async_checkpointer is not None:
        return _async_checkpointer

    settings = get_settings()

    _async_mongo_client = AsyncIOMotorClient(
        settings.mongo_url
    )

    _async_checkpointer = AsyncMongoDBSaver(
        client=_async_mongo_client,
        db_name=settings.mongo_db_name,
        checkpoint_collection_name="agent_checkpoints",
        writes_collection_name="agent_checkpoint_writes",
        ttl=settings.langgraph_checkpointer_ttl_seconds,
    )

    LOGGER.info(
        "async_checkpointer_initialized",
        db=settings.mongo_db_name,
        checkpoint_collection="agent_checkpoints",
        writes_collection="agent_checkpoint_writes",
    )

    return _async_checkpointer


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def close_checkpointer() -> None:
    """
    Close the MongoDB resources used by the LangGraph checkpointer.

    This function should be called during FastAPI application shutdown.

    The MongoDB client is closed and singleton references are cleared.
    """

    global _mongo_client
    global _checkpointer
    global _async_mongo_client
    global _async_checkpointer

    if _mongo_client is not None:

        _mongo_client.close()

        LOGGER.info(
            "checkpointer_closed"
        )

    if _async_mongo_client is not None:

        _async_mongo_client.close()

        LOGGER.info(
            "async_checkpointer_closed"
        )

    _mongo_client = None
    _checkpointer = None
    _async_mongo_client = None
    _async_checkpointer = None
