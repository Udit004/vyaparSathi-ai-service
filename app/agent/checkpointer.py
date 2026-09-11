"""
app/agent/checkpointer.py
=========================

MongoDB-backed LangGraph checkpointer for short-term cross-session memory.

Uses:

    langgraph-checkpoint-mongodb

Current API:

    from langgraph.checkpoint.mongodb import MongoDBSaver

The checkpointer is initialized once during FastAPI application startup
and shared across all agent runs.

The MongoDB client and checkpointer are application-level singletons.

The checkpointer is injected into the compiled LangGraph via:

    workflow.compile(checkpointer=checkpointer)
"""

from __future__ import annotations

import structlog

from pymongo import MongoClient

from langgraph.checkpoint.mongodb import MongoDBSaver

from app.config.settings import get_settings


LOGGER = structlog.get_logger(
    "vyaparsathi.ai.checkpointer"
)


# ---------------------------------------------------------------------------
# Application-level singletons
# ---------------------------------------------------------------------------

_mongo_client: MongoClient | None = None

_checkpointer: MongoDBSaver | None = None


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
                "thread_id": "{user_id}:{store_id}"
            }
        }

    This allows each (user, store) pair to maintain its own persistent
    LangGraph conversation state across HTTP requests.

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

    if _mongo_client is not None:

        _mongo_client.close()

        LOGGER.info(
            "checkpointer_closed"
        )

    _mongo_client = None
    _checkpointer = None