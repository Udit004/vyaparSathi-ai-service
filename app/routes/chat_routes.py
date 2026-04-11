from fastapi import APIRouter

from app.controllers.chat_controller import chat, copilot_chat, copilot_chat_stream
from app.schemas.chat import ChatResponse, CopilotResponse


router = APIRouter(tags=["chat"])


router.add_api_route("/chat", chat, methods=["POST"], response_model=ChatResponse)
router.add_api_route("/chat/copilot", copilot_chat, methods=["POST"], response_model=CopilotResponse)
router.add_api_route("/chat/copilot/stream", copilot_chat_stream, methods=["POST"])
