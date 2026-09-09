import structlog
from fastapi.responses import StreamingResponse

from app.schemas.chat import ChatRequest, ChatResponse, CopilotRequest, CopilotResponse
from app.services.chat_service import (
    generate_chat_response,
    generate_copilot_response,
    generate_copilot_stream_events,
)


LOGGER = structlog.get_logger("vyaparsathi.ai.chat")


async def chat(request: ChatRequest) -> ChatResponse:
    LOGGER.info("chat_request", message_len=len(request.message))
    return generate_chat_response(request)


async def copilot_chat(request: CopilotRequest) -> CopilotResponse:
    LOGGER.info("copilot_request", store=request.context.store_name, question_len=len(request.question))
    return generate_copilot_response(request)


async def copilot_chat_stream(request: CopilotRequest) -> StreamingResponse:
    LOGGER.info("copilot_stream_request", store=request.context.store_name, question_len=len(request.question))
    return StreamingResponse(
        generate_copilot_stream_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )