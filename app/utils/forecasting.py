from __future__ import annotations

from statistics import mean


def _average(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _regression_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    mean_x = (len(values) - 1) / 2
    mean_y = _average(values)
    numerator = 0.0
    denominator = 0.0

    for index, value in enumerate(values):
        numerator += (index - mean_x) * (value - mean_y)
        denominator += (index - mean_x) ** 2

    return numerator / denominator if denominator else 0.0


def confidence_bucket(values: list[float]) -> str:
    positive_days = len([value for value in values if value > 0])
    total_quantity = sum(values)

    if positive_days >= 12 and total_quantity >= 40:
        return "high"
    if positive_days >= 6 and total_quantity >= 15:
        return "medium"
    return "low"


def forecast_series(values: list[float], horizon_days: int = 7) -> dict:
    avg_7 = _average(values[-7:])
    avg_14 = _average(values[-14:])
    avg_30 = _average(values)
    slope = _regression_slope(values)

    predicted_daily_demand = max(
        0.0,
        round(avg_7 * 0.5 + avg_14 * 0.3 + avg_30 * 0.2 + slope * 2, 2),
    )
    predicted_demand = max(0.0, round(predicted_daily_demand * horizon_days, 2))
    trend_percent = (
        round(((avg_7 - avg_30) / avg_30) * 100, 2)
        if avg_30
        else 100.0
        if predicted_daily_demand > 0
        else 0.0
    )

    return {
        "predicted_daily_demand": predicted_daily_demand,
        "predicted_demand": predicted_demand,
        "trend_percent": trend_percent,
        "confidence": confidence_bucket(values),
    }


def summarize_metrics(subject: str, metrics: dict, basis: str) -> dict:
    current_stock = metrics.get("current_stock", 0)
    predicted_demand_7d = metrics.get("predicted_demand_7d", 0)
    recommended_qty = metrics.get("recommended_reorder_qty", 0)
    days_to_stockout = metrics.get("days_to_stockout")
    trend_percent = metrics.get("trend_percent", 0)

    if predicted_demand_7d:
        summary = (
            f"{subject} is expected to sell about {predicted_demand_7d:.1f} units in the next 7 days, "
            f"with demand trending {'up' if trend_percent >= 0 else 'down'} by {abs(trend_percent):.1f}%."
        )
    else:
        summary = f"{subject} shows very low near-term demand in the current analysis window."

    if recommended_qty > 0:
        recommendation = (
            f"Reorder {recommended_qty} units. Current stock is {current_stock} and projected demand "
            f"could create a stockout in {days_to_stockout:.1f} days."
            if days_to_stockout is not None
            else f"Reorder {recommended_qty} units to maintain a safer stock buffer."
        )
    else:
        recommendation = (
            f"Current stock of {current_stock} units is sufficient for the expected demand window."
        )

    return {
        "title": f"{subject} demand explanation",
        "summary": summary,
        "recommendation": recommendation,
        "basis": basis,
    }
