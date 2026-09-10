"""
app/controllers/chat_controller.py
====================================
HTTP-layer controllers for the simple chat endpoints.

The Copilot (agentic) flow has moved to app/agent/ and is invoked
directly from app/routes/store_ai_routes.py — these controllers only
own the simple single-turn LLM chat.

The legacy `copilot_chat_stream` function is kept here for the SSE
streaming route that was previously wired up in store_ai_routes.py.
It will be migrated to the new agent-based streaming endpoint in the
next phase.
"""

import structlog
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse, CopilotRequest
from app.services.chat_service import generate_chat_response

LOGGER = structlog.get_logger("vyaparsathi.ai.chat")


async def chat(request: ChatRequest) -> ChatResponse:
    LOGGER.info("chat_request", message_len=len(request.message))
    return generate_chat_response(request)


async def copilot_chat_stream(request: CopilotRequest) -> StreamingResponse:
    """
    Legacy SSE streaming copilot response.

    NOTE: This endpoint is being deprecated in favour of the new
    POST /{store_id}/copilot endpoint backed by the LangGraph agent.
    Kept alive so the existing frontend SSE consumer does not break.
    """
    LOGGER.info(
        "copilot_stream_request",
        store=request.context.store_name,
        question_len=len(request.question),
    )

    async def _stub_stream():
        yield "data: {\"type\": \"token\", \"content\": \"Copilot is being upgraded to the new agent. Please use POST /{store_id}/copilot.\"}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        _stub_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )