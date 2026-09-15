"""
app/agent/checkpointer.py
=========================

MongoDB-backed LangGraph checkpointer for short-term cross-session memory.

Uses:

    langgraph-checkpoint-mongodb>=0.2.0

Why v0.2.0?
    langgraph 0.2.x requires checkpoints to carry a ``pending_sends`` key.
    The 0.1.x MongoDB saver did not write that key, causing::

        KeyError: 'pending_sends'

    v0.2.0 of the MongoDB saver writes ``pending_sends`` and is compatible
    with ``langgraph-checkpoint >=2.0.23``.

API used here
-------------
``MongoDBSaver(client, db_name, ...)`` — the direct constructor accepts a
``pymongo.MongoClient`` and returns a saver instance that is NOT a context
manager. This lets us hold a single application-wide singleton across many
graph runs (both sync ``invoke`` and async ``astream_events``).

    from pymongo import MongoClient
    from langgraph.checkpoint.mongodb import MongoDBSaver

    saver = MongoDBSaver(
        client=MongoClient(mongo_url),
        db_name=db_name,
        checkpoint_collection_name="agent_checkpoints",
        writes_collection_name="agent_checkpoint_writes",
    )

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

    _mongo_client = MongoClient(settings.mongo_url)

    # -----------------------------------------------------------------------
    # Create LangGraph MongoDB checkpointer (v0.2.0).
    # Using the direct constructor (NOT from_conn_string, which is a
    # context manager that would close the client on exit).
    # -----------------------------------------------------------------------

    _checkpointer = MongoDBSaver(
        client=_mongo_client,
        db_name=settings.mongo_db_name,
        checkpoint_collection_name="agent_checkpoints",
        writes_collection_name="agent_checkpoint_writes",
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
        try:
            _mongo_client.close()
        except Exception as exc:  # pragma: no cover - best-effort close
            LOGGER.warning(
                "checkpointer_close_failed",
                error=str(exc),
            )

    _mongo_client = None
    _checkpointer = None

    LOGGER.info("checkpointer_closed")