from fastapi import APIRouter

from app.controllers.analytics_controller import generate_insights, find_anomalies
from app.schemas.analytics import (
    GenerateInsightsRequest,
    GenerateInsightsResponse,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
)


router = APIRouter(tags=["analytics"], prefix="/analytics")


router.add_api_route(
    "/insights",
    generate_insights,
    methods=["POST"],
    response_model=GenerateInsightsResponse,
)

router.add_api_route(
    "/anomalies",
    find_anomalies,
    methods=["POST"],
    response_model=DetectAnomaliesResponse,
)
