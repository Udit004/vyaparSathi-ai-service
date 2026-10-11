"""
app/agent/checkpointer.py
=========================

MongoDB-backed LangGraph checkpointer for short-term cross-session memory.

Uses:

    langgraph-checkpoint-mongodb>=0.2.0

IMPORTANT: We use AsyncMongoDBSaver with pymongo's native AsyncMongoClient
(pymongo >= 4.7) instead of the sync MongoDBSaver.

The sync MongoDBSaver uses blocking pymongo calls which block the asyncio
event loop BETWEEN EVERY NODE. With 5-6 nodes per request and each MongoDB
write taking 5-20ms (or more on Atlas), that's 30-120ms+ of unnecessary
blocking per request. Blocking the event loop prevents other async operations
(like LLM streaming responses) from making progress, causing cascading slowdowns.

AsyncMongoDBSaver uses pymongo's native async client under the hood, so
checkpoint writes are properly non-blocking.

NOTE: Do NOT use motor.motor_asyncio.AsyncIOMotorClient here — the
AsyncMongoDBSaver in langgraph-checkpoint-mongodb calls append_metadata()
which Motor does not support. Use pymongo.AsyncMongoClient instead.

The checkpointer is injected into the compiled LangGraph via:

    workflow.compile(checkpointer=checkpointer)
"""

from __future__ import annotations

import structlog

from pymongo import AsyncMongoClient

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

from app.config.settings import get_settings


LOGGER = structlog.get_logger(
    "vyaparsathi.ai.checkpointer"
)


# ---------------------------------------------------------------------------
# Application-level singletons
# ---------------------------------------------------------------------------

_async_client: AsyncMongoClient | None = None

_checkpointer: AsyncMongoDBSaver | None = None


# ---------------------------------------------------------------------------
# Checkpointer initialization
# ---------------------------------------------------------------------------

def get_checkpointer() -> AsyncMongoDBSaver:
    """
    Return the application-wide async MongoDB LangGraph checkpointer.

    The async client and checkpointer are created once and reused
    throughout the FastAPI application lifecycle.

    Uses AsyncMongoDBSaver with pymongo's native AsyncMongoClient instead
    of the sync MongoDBSaver to avoid blocking the asyncio event loop
    between every node execution.

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
        AsyncMongoDBSaver:
            Initialized async MongoDB-backed LangGraph checkpointer.
    """

    global _async_client
    global _checkpointer

    # Return existing singleton
    if _checkpointer is not None:
        return _checkpointer

    settings = get_settings()

    # -----------------------------------------------------------------------
    # Create pymongo native async client (pymongo >= 4.7).
    # This is what AsyncMongoDBSaver expects internally.
    # -----------------------------------------------------------------------

    _async_client = AsyncMongoClient(settings.mongo_url)

    # -----------------------------------------------------------------------
    # Create async LangGraph MongoDB checkpointer.
    # AsyncMongoDBSaver uses pymongo AsyncMongoClient internally —
    # checkpoint writes are non-blocking and won't stall the event loop.
    # -----------------------------------------------------------------------

    _checkpointer = AsyncMongoDBSaver(
        client=_async_client,
        db_name=settings.mongo_db_name,
        checkpoint_collection_name="agent_checkpoints",
        writes_collection_name="agent_checkpoint_writes",
    )

    LOGGER.info(
        "checkpointer_initialized",
        db=settings.mongo_db_name,
        checkpoint_collection="agent_checkpoints",
        writes_collection="agent_checkpoint_writes",
        driver="pymongo_async",
    )

    return _checkpointer


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def close_checkpointer() -> None:
    """
    Close the MongoDB resources used by the LangGraph checkpointer.

    This function should be called during FastAPI application shutdown.

    The async client is closed and singleton references are cleared.
    """

    global _async_client
    global _checkpointer

    if _async_client is not None:
        try:
            _async_client.close()
        except Exception as exc:  # pragma: no cover - best-effort close
            LOGGER.warning(
                "checkpointer_close_failed",
                error=str(exc),
            )

    _async_client = None
    _checkpointer = None

    LOGGER.info("checkpointer_closed")