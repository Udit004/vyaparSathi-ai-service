from app.schemas.forecast import ForecastRequest, ForecastResponse
from app.services.forecast_service import generate_forecast_response


async def forecast(request: ForecastRequest) -> ForecastResponse:
    return generate_forecast_response(request)
