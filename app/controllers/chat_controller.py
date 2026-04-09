from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import generate_chat_response


async def chat(request: ChatRequest) -> ChatResponse:
    return generate_chat_response(request)
