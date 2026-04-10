import logging

from app.schemas.insight import (
    InsightExplanationRequest,
    InsightExplanationResponse,
)
from app.services.insight_service import generate_insight_explanation


LOGGER = logging.getLogger("uvicorn.error")


async def explain_insights(
    request: InsightExplanationRequest,
) -> InsightExplanationResponse:
    response = generate_insight_explanation(request)

    LOGGER.info(
        "insight explanation subject=%s basis=%s payload=%s",
        request.subject,
        request.basis,
        response.model_dump(),
    )

    return response
