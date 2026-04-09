from app.schemas.forecast import ForecastRequest, ForecastResponse, ForecastResult
from app.utils.forecasting import forecast_series


def generate_forecast_response(request: ForecastRequest) -> ForecastResponse:
    results = []

    for series in request.series:
        values = [entry.quantity for entry in series.values]
        forecast_result = forecast_series(values, request.horizon_days)
        results.append(
            ForecastResult(
                product_id=series.product_id,
                product_name=series.product_name,
                predicted_daily_demand=forecast_result["predicted_daily_demand"],
                predicted_demand_7d=forecast_result["predicted_demand"],
                trend_percent=forecast_result["trend_percent"],
                confidence=forecast_result["confidence"],
            )
        )

    return ForecastResponse(results=results)
