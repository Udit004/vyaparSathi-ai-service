import structlog

from app.schemas.insight import (
    InsightExplanationRequest,
    InsightExplanationResponse,
    StoreInsightExplanationRequest,
)
from app.services.insight_service import (
    generate_insight_explanation,
    generate_store_insight_explanation,
)


LOGGER = structlog.get_logger("vyaparsathi.ai.insights")


async def explain_insights(
    request: InsightExplanationRequest,
) -> InsightExplanationResponse:
    response = generate_insight_explanation(request)

    LOGGER.info(
        "insight_explanation_completed",
        subject=request.subject,
        basis=request.basis,
        llm_used=response.llmUsed,
    )

    return response


async def explain_store_insights(
    request: StoreInsightExplanationRequest,
) -> InsightExplanationResponse:
    response = generate_store_insight_explanation(request)

    LOGGER.info(
        "store_insight_explanation_completed",
        store=request.store_name,
        basis=request.basis,
        llm_used=response.llmUsed,
    )

    return response