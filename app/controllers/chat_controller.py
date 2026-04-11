from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse, CopilotRequest, CopilotResponse
from app.services.chat_service import (
    generate_chat_response,
    generate_copilot_response,
    generate_copilot_stream_events,
)


async def chat(request: ChatRequest) -> ChatResponse:
    return generate_chat_response(request)


async def copilot_chat(request: CopilotRequest) -> CopilotResponse:
    return generate_copilot_response(request)


async def copilot_chat_stream(request: CopilotRequest) -> StreamingResponse:
    return StreamingResponse(
        generate_copilot_stream_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
