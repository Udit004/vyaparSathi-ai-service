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
# Copilot — agent-powered endpoints
# ---------------------------------------------------------------------------

class CopilotStreamPayload(BaseModel):
    message: str
    session_id: str | None = None



@router.post("/{store_id}/copilot/stream")
async def get_copilot_stream(store_id: str, payload: CopilotStreamPayload, request: Request):
    """
    LangGraph agent Copilot endpoint (streaming SSE).
    Connects to the frontend's EventSource/fetch stream.
    """
    from app.agent import build_graph, get_async_checkpointer, make_initial_state

    session_id = payload.session_id
    if not session_id:
        session_id = str(uuid.uuid4())

    user_id = request.headers.get("x-user-id", "default_user")
    thread_id = f"{user_id}:{store_id}:{session_id}"

    async def event_generator():
        try:
            checkpointer = get_async_checkpointer()
            graph = build_graph(checkpointer=checkpointer)

            initial_state = make_initial_state(
                user_id=user_id,
                store_id=store_id,
                user_prompt=payload.message,
            )

            config = {"configurable": {"thread_id": thread_id}}

            # Stream execution events
            async for event in graph.astream_events(initial_state, config=config, version="v2"):
                kind = event["event"]
                name = event["name"]

                # We can stream LLM chunks directly to the UI
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        data = json.dumps({"text": chunk.content})
                        yield f"event: token\ndata: {data}\n\n"
                
                # We can also stream tool execution updates if we want to show "Calling X..."
                elif kind == "on_tool_start":
                    msg = f"Calling tool: {name}..."
                    data = json.dumps({"text": f"\n_[{msg}]_\n"})
                    yield f"event: token\ndata: {data}\n\n"

                elif kind == "on_tool_end":
                    msg = f"Finished tool: {name}."
                    data = json.dumps({"text": f"\n_[{msg}]_\n"})
                    yield f"event: token\ndata: {data}\n\n"

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
