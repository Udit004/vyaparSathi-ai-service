from fastapi import APIRouter

from app.controllers.insight_controller import explain_insights
from app.schemas.insight import InsightExplanationResponse


router = APIRouter(tags=["insights"])


router.add_api_route(
    "/explain-insights",
    explain_insights,
    methods=["POST"],
    response_model=InsightExplanationResponse,
)
