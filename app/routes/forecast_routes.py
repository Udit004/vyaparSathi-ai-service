from fastapi import APIRouter

from app.controllers.forecast_controller import forecast
from app.schemas.forecast import ForecastResponse


router = APIRouter(tags=["forecast"])


router.add_api_route(
    "/forecast",
    forecast,
    methods=["POST"],
    response_model=ForecastResponse,
)
