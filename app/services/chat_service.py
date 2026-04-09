from app.lib.llm import get_llm
from app.schemas.chat import ChatRequest, ChatResponse


def generate_chat_response(request: ChatRequest) -> ChatResponse:
    llm = get_llm()

    if not llm:
        return ChatResponse(
            response="AI chat is unavailable because the Gemini API key is not configured."
        )

    response = llm.invoke(request.message)
    return ChatResponse(response=response.content)
