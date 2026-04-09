from fastapi import APIRouter

from app.controllers.chat_controller import chat
from app.schemas.chat import ChatResponse


router = APIRouter(tags=["chat"])


router.add_api_route("/chat", chat, methods=["POST"], response_model=ChatResponse)
