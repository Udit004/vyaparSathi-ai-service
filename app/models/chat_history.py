"""
app/models/chat_history.py
===========================
Pydantic models for chat history stored in MongoDB.

Collections:
    agent_chats — one document per chat session, stores metadata + message summary
    agent_chat_messages — one document per individual message (user/assistant/tool)

Design:
    - A "chat" is a logical conversation thread between a user and the copilot
      scoped to a specific store.
    - Each chat has a unique `chat_id` (UUID).
    - The LangGraph checkpointer uses `thread_id = f"{user_id}:{store_id}:{chat_id}"`
      so each chat gets its own isolated checkpoint.
    - When a new chat is started, a fresh thread_id is created, giving the agent
      a clean context window.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Message model
# ---------------------------------------------------------------------------

class ChatMessageModel(BaseModel):
    """
    A single message within a chat session.

    role: "user" | "assistant" | "tool" | "system"
    """
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # Optional metadata (tool name, loop count, etc.)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chat session model
# ---------------------------------------------------------------------------

class ChatSessionModel(BaseModel):
    """
    Metadata document for one chat session.

    Stored in the `agent_chats` collection.
    """
    chat_id: str
    user_id: str
    store_id: str
    title: str = "New Chat"
    # Summary of the conversation (updated periodically or on end)
    summary: str = ""
    # Number of user/assistant turns
    message_count: int = 0
    # LangGraph checkpoint thread id
    thread_id: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    # Soft delete flag
    is_active: bool = True
    # Extra metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)