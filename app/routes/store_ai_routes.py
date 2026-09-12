"""
app/routes/store_ai_routes.py
==============================
FastAPI routes for all AI features scoped to a specific store.

Endpoints:
    GET  /{store_id}/forecast       — demand forecast
    GET  /{store_id}/restock        — restock plan
    GET  /{store_id}/insights       — store insights
    GET  /{store_id}/summary        — store summary
    GET  /{store_id}/product/{id}   — per-product insight
    POST /{store_id}/copilot        — agent-powered Copilot (non-streaming)
    POST /{store_id}/copilot/stream — agent-powered Copilot (streaming SSE)
"""

import json
import uuid
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Any

from app.services.aggregation_service import (
    get_forecast_for_store,
    get_restock_for_store,
    get_insights_for_store,
)
from app.services.chat_history_service import (
    get_or_create_chat_session,
    add_user_message,
    add_assistant_message,
    list_chat_sessions,
    get_chat_history,
    soft_delete_chat_session,
)
from app.models.chat_history import ChatSessionModel, ChatMessageModel

router = APIRouter(tags=["store_ai"])


# ---------------------------------------------------------------------------
# Shared response model
# ---------------------------------------------------------------------------

class ApiResponse(BaseModel):
    data: Any
    message: str
    statusCode: int = 200


# ---------------------------------------------------------------------------
# Existing analytics endpoints (unchanged)
# ---------------------------------------------------------------------------

@router.get("/{store_id}/forecast", response_model=ApiResponse)
async def get_forecast(store_id: str):
    data = await get_forecast_for_store(store_id)
    return ApiResponse(data=data, message="Forecast generated successfully")


@router.get("/{store_id}/restock", response_model=ApiResponse)
async def get_restock(store_id: str):
    data = await get_restock_for_store(store_id)
    return ApiResponse(data=data, message="Restock plan generated successfully")


@router.get("/{store_id}/insights", response_model=ApiResponse)
async def get_insights(store_id: str):
    data = await get_insights_for_store(store_id)
    return ApiResponse(data=data, message="Insights generated successfully")


@router.get("/{store_id}/summary", response_model=ApiResponse)
async def get_summary(store_id: str):
    data = {"title": "Store summary", "summary": "Store summary data placeholder."}
    return ApiResponse(data=data, message="Store summary generated successfully")


@router.get("/{store_id}/product/{product_id}", response_model=ApiResponse)
async def get_product_insight(store_id: str, product_id: str):
    data = {"productId": product_id, "insight": "No specific insight available in local fallback."}
    return ApiResponse(data=data, message="Product insight generated successfully")


# ---------------------------------------------------------------------------
# Chat history endpoints
# ---------------------------------------------------------------------------

class ChatSessionItem(BaseModel):
    chat_id: str
    title: str
    message_count: int = 0
    created_at: str
    updated_at: str


class ChatSessionListResponse(BaseModel):
    chats: list[ChatSessionItem]
    total: int = 0


class ChatMessageItem(BaseModel):
    role: str
    content: str
    timestamp: str


class ChatHistoryResponse(BaseModel):
    chat_id: str
    messages: list[ChatMessageItem]


@router.get("/{store_id}/chats", response_model=ApiResponse)
async def list_chats(store_id: str, request: Request):
    """
    List all active chat sessions for the current user + store.
    """
    user_id = request.headers.get("x-user-id", "default_user")
    sessions = await list_chat_sessions(
        user_id=user_id,
        store_id=store_id,
        limit=100,
    )

    chats = [
        ChatSessionItem(
            chat_id=s.chat_id,
            title=s.title or "New Chat",
            message_count=s.message_count,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
        )
        for s in sessions
    ]

    return ApiResponse(
        data=ChatSessionListResponse(chats=chats, total=len(chats)).model_dump(),
        message="Chat sessions retrieved successfully",
    )


@router.get("/{store_id}/chats/{chat_id}", response_model=ApiResponse)
async def get_chat(store_id: str, chat_id: str, request: Request):
    """
    Retrieve the full message history for a specific chat.
    """
    user_id = request.headers.get("x-user-id", "default_user")
    messages = await get_chat_history(chat_id, limit=200)

    # Verify the chat belongs to this user/store by checking the first message
    # or by looking up the chat session. We use get_chat_history which doesn't
    # enforce ownership, so do an explicit check here.
    from app.services.chat_history_service import get_chat_session
    session = await get_chat_session(chat_id, user_id=user_id)
    if not session or session.store_id != store_id:
        return ApiResponse(
            data=None,
            message="Chat not found",
            statusCode=404,
        )

    items = [
        ChatMessageItem(
            role=m.role,
            content=m.content,
            timestamp=m.timestamp.isoformat(),
        )
        for m in messages
    ]

    return ApiResponse(
        data=ChatHistoryResponse(chat_id=chat_id, messages=items).model_dump(),
        message="Chat history retrieved successfully",
    )


@router.delete("/{store_id}/chats/{chat_id}", response_model=ApiResponse)
async def delete_chat(store_id: str, chat_id: str, request: Request):
    """
    Soft-delete a chat session.
    """
    user_id = request.headers.get("x-user-id", "default_user")
    from app.services.chat_history_service import get_chat_session
    session = await get_chat_session(chat_id, user_id=user_id)
    if not session or session.store_id != store_id:
        return ApiResponse(
            data=None,
            message="Chat not found",
            statusCode=404,
        )

    ok = await soft_delete_chat_session(chat_id)
    return ApiResponse(
        data={"deleted": ok},
        message="Chat deleted successfully" if ok else "Failed to delete chat",
    )


# ---------------------------------------------------------------------------
# Copilot — agent-powered endpoints
# ---------------------------------------------------------------------------

class CopilotStreamPayload(BaseModel):
    message: str
    chat_id: str | None = None
    # Legacy field kept for backwards compatibility; ignored if chat_id is provided.
    session_id: str | None = None


@router.post("/{store_id}/copilot/stream")
async def get_copilot_stream(store_id: str, payload: CopilotStreamPayload, request: Request):
    """
    LangGraph agent Copilot endpoint (streaming SSE).

    Each logical conversation is identified by a ``chat_id``. Starting a new
    chat (no ``chat_id`` or an unknown one) creates a fresh LangGraph
    ``thread_id`` so the agent gets a clean context window. Chat history is
    persisted in MongoDB (``agent_chats`` + ``agent_chat_messages``).

    The frontend sends ``chat_id`` (preferred) or the legacy ``session_id``.
    """
    from app.agent import build_graph, get_async_checkpointer, make_initial_state

    user_id = request.headers.get("x-user-id", "default_user")

    # Resolve the chat session (existing or new)
    chat_session = await get_or_create_chat_session(
        user_id=user_id,
        store_id=store_id,
        chat_id=payload.chat_id,
    )

    chat_id = chat_session.chat_id
    thread_id = chat_session.thread_id

    # Persist the user message
    await add_user_message(chat_id, payload.message)

    # Accumulator for the assistant's streamed text so we can persist it
    # at the end of the stream.
    assistant_text_parts: list[str] = []

    async def event_generator():
        try:
            checkpointer = get_async_checkpointer()
            graph = build_graph(checkpointer=checkpointer)

            initial_state = make_initial_state(
                user_id=user_id,
                store_id=store_id,
                user_prompt=payload.message,
                thread_id=thread_id,
            )

            config = {"configurable": {"thread_id": thread_id}}

            # Stream execution events
            async for event in graph.astream_events(initial_state, config=config, version="v2"):
                kind = event["event"]
                name = event["name"]

                # Stream LLM chunks directly to the UI
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        text = chunk.content
                        assistant_text_parts.append(text)
                        data = json.dumps({"text": text})
                        yield f"event: token\ndata: {data}\n\n"

                # Stream tool execution updates
                elif kind == "on_tool_start":
                    msg = f"Calling tool: {name}..."
                    data = json.dumps({"text": f"\n_[{msg}]_\n"})
                    yield f"event: token\ndata: {data}\n\n"

                elif kind == "on_tool_end":
                    msg = f"Finished tool: {name}."
                    data = json.dumps({"text": f"\n_[{msg}]_\n"})
                    yield f"event: token\ndata: {data}\n\n"

            # Persist assistant response to chat history
            full_response = "".join(assistant_text_parts)
            if full_response:
                await add_assistant_message(
                    chat_id,
                    full_response,
                    metadata={"thread_id": thread_id},
                )

            # Emit session metadata so the frontend can store the chat_id
            session_data = json.dumps({
                "chatId": chat_id,
                "threadId": thread_id,
            })
            yield f"event: session\ndata: {session_data}\n\n"

            # End of stream
            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            # Send error in stream
            error_data = json.dumps({"message": str(e)})
            yield f"event: error\ndata: {error_data}\n\n"
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
