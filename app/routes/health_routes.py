from fastapi import APIRouter

from app.controllers.health_controller import root


router = APIRouter(tags=["health"])


router.add_api_route("/", root, methods=["GET"])
