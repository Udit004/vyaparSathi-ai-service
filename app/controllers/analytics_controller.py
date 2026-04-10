import logging

from app.schemas.analytics import (
    GenerateInsightsRequest,
    GenerateInsightsResponse,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
)
from app.services.analytics_service import compute_insights, detect_anomalies


LOGGER = logging.getLogger("uvicorn.error")


async def generate_insights(
    request: GenerateInsightsRequest,
) -> GenerateInsightsResponse:
    """
    Generate insights from forecast and restock data.
    Includes: fastest selling, slow moving, dead stock, category mix, and restock priority insights.
    """
    
    LOGGER.info(
        "Generating insights for forecast_items=%s restock_items=%s anomalies=%s products=%s",
        len(request.forecast_items),
        len(request.restock_items),
        len(request.anomalies),
        len(request.products),
    )
    
    response = compute_insights(request)
    
    LOGGER.info("Generated %s insights", len(response.insights))
    
    return response


async def find_anomalies(
    request: DetectAnomaliesRequest,
) -> DetectAnomaliesResponse:
    """
    Detect statistical anomalies (spikes/drops) in sales time series.
    Uses z-score method to identify unusual patterns.
    """
    
    LOGGER.info(
        "Detecting anomalies in %s series",
        len(request.series_collection),
    )
    
    response = detect_anomalies(request)
    
    LOGGER.info("Found %s anomalies", len(response.anomalies))
    
    return response
