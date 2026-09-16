"""
app/services/chat_history_service.py
======================================
MongoDB CRUD for chat sessions and messages.

Collections:
    agent_chats           — one document per chat session (metadata)
    agent_chat_messages   — one document per individual message

All functions are async and use the shared motor client from
app.config.database.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal

import structlog

from app.config.database import get_database
from app.models.chat_history import ChatSessionModel, ChatMessageModel

LOGGER = structlog.get_logger("vyaparsathi.ai.chat_history")


# ---------------------------------------------------------------------------
# Chat session operations
# ---------------------------------------------------------------------------

async def create_chat_session(
    *,
    user_id: str,
    store_id: str,
    title: str = "New Chat",
    metadata: Optional[Dict[str, Any]] = None,
) -> ChatSessionModel:
    """
    Create a new chat session document.

    Generates a fresh chat_id (UUID) and derives the LangGraph thread_id
    as ``"{user_id}:{store_id}:{chat_id}"``.

    Returns:
        ChatSessionModel: The newly created chat session.
    """
    db = get_database()
    chat_id = str(uuid.uuid4())
    thread_id = f"{user_id}:{store_id}:{chat_id}"

    session = ChatSessionModel(
        chat_id=chat_id,
        user_id=user_id,
        store_id=store_id,
        title=title,
        thread_id=thread_id,
        metadata=metadata or {},
    )

    await db["agent_chats"].insert_one(session.model_dump(by_alias=True))
    LOGGER.info(
        "chat_session_created",
        chat_id=chat_id,
        user_id=user_id,
        store_id=store_id,
        thread_id=thread_id,
    )

    return session


async def get_chat_session(
    chat_id: str,
    user_id: Optional[str] = None,
) -> Optional[ChatSessionModel]:
    """
    Retrieve a chat session by chat_id.

    If user_id is provided, the query also filters by user_id for security.
    """
    db = get_database()
    query: Dict[str, Any] = {"chat_id": chat_id, "is_active": True}
    if user_id:
        query["user_id"] = user_id

    doc = await db["agent_chats"].find_one(query)
    if not doc:
        return None

    # Normalize _id -> id for Pydantic
    doc.pop("_id", None)
    return ChatSessionModel(**doc)


async def get_or_create_chat_session(
    *,
    user_id: str,
    store_id: str,
    chat_id: Optional[str] = None,
) -> ChatSessionModel:
    """
    Return an existing chat session if chat_id is provided and valid,
    otherwise create a new one.

    This is the main entry point used by the copilot streaming endpoint.
    """
    if chat_id:
        existing = await get_chat_session(chat_id, user_id=user_id)
        if existing:
            LOGGER.info(
                "chat_session_reused",
                chat_id=chat_id,
                user_id=user_id,
                store_id=store_id,
            )
            return existing
        # chat_id provided but not found — log and create a new one
        LOGGER.warning(
            "chat_session_not_found_creating_new",
            requested_chat_id=chat_id,
            user_id=user_id,
            store_id=store_id,
        )

    return await create_chat_session(user_id=user_id, store_id=store_id)


async def list_chat_sessions(
    *,
    user_id: str,
    store_id: str,
    limit: int = 50,
) -> List[ChatSessionModel]:
    """
    List all active chat sessions for a (user, store) pair, newest first.
    """
    db = get_database()
    cursor = db["agent_chats"].find(
        {"user_id": user_id, "store_id": store_id, "is_active": True}
    ).sort("updated_at", -1).limit(limit)

    sessions = []
    async for doc in cursor:
        doc.pop("_id", None)
        sessions.append(ChatSessionModel(**doc))

    LOGGER.debug(
        "chat_sessions_listed",
        user_id=user_id,
        store_id=store_id,
        count=len(sessions),
    )
    return sessions


async def update_chat_session(
    chat_id: str,
    *,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    message_count: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[ChatSessionModel]:
    """
    Partially update a chat session document.

    Only the provided fields are updated; ``updated_at`` is always refreshed.
    """
    db = get_database()
    update: Dict[str, Any] = {"updated_at": datetime.utcnow()}

    if title is not None:
        update["title"] = title
    if summary is not None:
        update["summary"] = summary
    if message_count is not None:
        update["message_count"] = message_count
    if metadata is not None:
        update["metadata"] = metadata

    doc = await db["agent_chats"].find_one_and_update(
        {"chat_id": chat_id},
        {"$set": update},
        return_document=True,
    )
    if not doc:
        return None

    doc.pop("_id", None)
    return ChatSessionModel(**doc)


async def soft_delete_chat_session(chat_id: str) -> bool:
    """Soft-delete a chat session (sets is_active=False)."""
    db = get_database()
    result = await db["agent_chats"].update_one(
        {"chat_id": chat_id},
        {"$set": {"is_active": False, "updated_at": datetime.utcnow()}},
    )
    return result.modified_count > 0


# ---------------------------------------------------------------------------
# Message operations
# ---------------------------------------------------------------------------

async def add_chat_message(
    chat_id: str,
    message: ChatMessageModel,
) -> str:
    """
    Append a single message document to agent_chat_messages.

    Returns the inserted document's _id as a string.
    """
    db = get_database()
    doc = message.model_dump(by_alias=True)
    doc["chat_id"] = chat_id
    result = await db["agent_chat_messages"].insert_one(doc)
    LOGGER.debug(
        "chat_message_added",
        chat_id=chat_id,
        role=message.role,
        inserted_id=str(result.inserted_id),
    )
    return str(result.inserted_id)


async def add_user_message(chat_id: str, content: str) -> str:
    """Convenience: add a user message."""
    return await add_chat_message(
        chat_id,
        ChatMessageModel(role="user", content=content),
    )


async def add_assistant_message(
    chat_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Convenience: add an assistant message."""
    return await add_chat_message(
        chat_id,
        ChatMessageModel(
            role="assistant",
            content=content,
            metadata=metadata or {},
        ),
    )


async def get_chat_history(
    chat_id: str,
    limit: int = 100,
) -> List[ChatMessageModel]:
    """
    Retrieve the message history for a chat, oldest first.
    """
    db = get_database()
    cursor = db["agent_chat_messages"].find(
        {"chat_id": chat_id}
    ).sort("timestamp", 1).limit(limit)

    messages = []
    async for doc in cursor:
        doc.pop("_id", None)
        messages.append(ChatMessageModel(**doc))

    LOGGER.debug(
        "chat_history_fetched",
        chat_id=chat_id,
        count=len(messages),
    )
    return messages


async def get_recent_messages_for_llm(
    chat_id: str,
    limit: int = 20,
) -> List[Dict[str, str]]:
    """
    Return recent chat history formatted for LLM consumption
    (list of {role, content} dicts, newest first is NOT desired here —
    we return oldest-first so the LLM sees conversation in order).
    """
    messages = await get_chat_history(chat_id, limit=limit)
    return [
        {"role": m.role, "content": m.content}
        for m in messages
    ]


async def count_messages(chat_id: str) -> int:
    """Return the total message count for a chat."""
    db = get_database()
    return await db["agent_chat_messages"].count_documents({"chat_id": chat_id})


# ---------------------------------------------------------------------------
# Chat title generation
# ---------------------------------------------------------------------------

# Default title used until the LLM produces a real one.
DEFAULT_CHAT_TITLE = "New Chat"


async def generate_chat_title(chat_id: str, user_prompt: str) -> str:
    """
    Generate a short, human-readable title for a chat from its first user
    message.

    Uses the small/fast summarizer (GROQ openai/gpt-oss-20b / NVIDIA
    nvidia/llama-3.1-8b-instruct) instead of the main Gemini model, so Gemini
    quota is reserved for retail-agent reasoning. Falls back to a truncated
    version of the prompt if no provider is available or the call fails.

    The title is a single line, <= 60 chars, suitable for a sidebar list.
    """
    if not user_prompt or not user_prompt.strip():
        return DEFAULT_CHAT_TITLE

    # Fast deterministic fallback — a truncated, cleaned version of the prompt.
    def _fallback(prompt: str) -> str:
        cleaned = " ".join(prompt.split())  # collapse whitespace
        if len(cleaned) <= 60:
            return cleaned
        return cleaned[:57] + "..."

    try:
        from app.lib.summarizer import summarize
        from app.agent.prompts.summarizer_prompts import TITLE_GENERATION_INSTRUCTION

        title = await summarize(
            user_prompt,
            instruction=TITLE_GENERATION_INSTRUCTION,
            max_tokens=64,
        )
        if not title:
            return _fallback(user_prompt)
    except Exception as exc:
        LOGGER.warning("generate_chat_title_fallback", chat_id=chat_id, error=str(exc)[:200])
        return _fallback(user_prompt)

    title = title.strip()
    # Strip surrounding quotes if the LLM added them
    if len(title) >= 2 and title[0] in "\"'" and title[-1] == title[0]:
        title = title[1:-1]
    title = " ".join(title.split())  # collapse whitespace
    if not title:
        return _fallback(user_prompt)
    if len(title) > 60:
        title = title[:57] + "..."
    return title


async def ensure_chat_title(chat_id: str, user_prompt: str) -> str:
    """
    Update the chat session title if it is still the default "New Chat".

    Returns the (possibly new) title. Idempotent — if the title was already
    set to something meaningful, it is returned unchanged.
    """
    session = await get_chat_session(chat_id)
    if not session:
        return DEFAULT_CHAT_TITLE
    if session.title and session.title != DEFAULT_CHAT_TITLE:
        return session.title

    title = await generate_chat_title(chat_id, user_prompt)
    await update_chat_session(chat_id, title=title)
    LOGGER.info("chat_title_generated", chat_id=chat_id, title=title)
    return title