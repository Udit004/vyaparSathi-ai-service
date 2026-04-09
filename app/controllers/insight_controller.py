from app.schemas.insight import (
    InsightExplanationRequest,
    InsightExplanationResponse,
)
from app.services.insight_service import generate_insight_explanation


async def explain_insights(
    request: InsightExplanationRequest,
) -> InsightExplanationResponse:
    return generate_insight_explanation(request)
