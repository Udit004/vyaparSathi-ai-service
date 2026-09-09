import structlog

from app.schemas.analytics import (
    GenerateInsightsRequest,
    GenerateInsightsResponse,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
)
from app.services.analytics_service import compute_insights, detect_anomalies


LOGGER = structlog.get_logger("vyaparsathi.ai.analytics")


async def generate_insights(
    request: GenerateInsightsRequest,
) -> GenerateInsightsResponse:
    LOGGER.info(
        "insights_request",
        forecast_items=len(request.forecast_items),
        restock_items=len(request.restock_items),
        anomalies=len(request.anomalies),
        products=len(request.products),
    )

    response = compute_insights(request)

    LOGGER.info("insights_completed", insights_count=len(response.insights))

    return response


async def find_anomalies(
    request: DetectAnomaliesRequest,
) -> DetectAnomaliesResponse:
    LOGGER.info("anomalies_request", series_count=len(request.series_collection))

    response = detect_anomalies(request)

    LOGGER.info("anomalies_completed", anomalies_count=len(response.anomalies))

    return response