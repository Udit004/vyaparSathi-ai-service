import logging

from app.schemas.forecast import ForecastRequest, ForecastResponse
from app.services.forecast_service import generate_forecast_response


LOGGER = logging.getLogger("uvicorn.error")


async def forecast(request: ForecastRequest) -> ForecastResponse:
    response = generate_forecast_response(request)

    LOGGER.info(
        "forecast results count=%s payload=%s",
        len(response.results),
        response.model_dump(),
    )

    return response
