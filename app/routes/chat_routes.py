from fastapi import APIRouter

from app.controllers.chat_controller import chat, copilot_chat_stream
from app.schemas.chat import ChatResponse


router = APIRouter(tags=["chat"])


router.add_api_route("/chat", chat, methods=["POST"], response_model=ChatResponse)
# Legacy SSE streaming endpoint — kept for backwards compatibility
router.add_api_route("/chat/copilot/stream", copilot_chat_stream, methods=["POST"])
