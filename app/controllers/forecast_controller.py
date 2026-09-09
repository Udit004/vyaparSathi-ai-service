import structlog

from app.schemas.forecast import ForecastRequest, ForecastResponse
from app.services.forecast_service import generate_forecast_response


LOGGER = structlog.get_logger("vyaparsathi.ai.forecast")


async def forecast(request: ForecastRequest) -> ForecastResponse:
    response = generate_forecast_response(request)

    LOGGER.info(
        "forecast_completed",
        results_count=len(response.results),
        sources=[r.source for r in response.results],
    )

    return response