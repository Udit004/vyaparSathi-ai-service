from __future__ import annotations

from datetime import date, timedelta
import pytest

from app.schemas.forecast import DailyValue, ForecastRequest, ForecastSeriesInput
from app.services.forecast_service import generate_forecast_response, _lazy_load_artifacts
from app.utils.indian_calendar import is_indian_festival_window, is_payday_window, is_weekend, get_calendar_features


def test_calendar_features():
    # Test Saturday weekend
    sat = date(2026, 10, 10)
    assert is_weekend(sat) == 1

    # Test Wednesday weekday
    wed = date(2026, 10, 14)
    assert is_weekend(wed) == 0

    # Test 1st of month payday
    p_first = date(2026, 10, 1)
    assert is_payday_window(p_first) == 1

    # Test 15th non-payday
    p_mid = date(2026, 10, 15)
    assert is_payday_window(p_mid) == 0

    # Test Diwali window (Nov 10, 2026 is inside Diwali window)
    fest_date = date(2026, 11, 10)
    assert is_indian_festival_window(fest_date) == 1

    # Test features dict
    cal = get_calendar_features(sat)
    assert "day_of_week" in cal
    assert "day_of_month" in cal
    assert "is_weekend" in cal
    assert "is_festival_window" in cal


def test_model_loading_and_properties():
    model, features = _lazy_load_artifacts()
    assert model is not None
    assert features is not None
    assert len(features) == 16
    assert "is_festival_window" in features
    assert "is_weekend" in features
    assert "day_of_month" in features
    assert "store" in features
    assert "item" in features

    cat_indices = model.get_cat_feature_indices()
    assert len(cat_indices) > 0  # Native categorical strings registered


def test_direct_horizon_forecast_generation():
    start = date(2026, 9, 1)
    values = [
        DailyValue(
            date=(start + timedelta(days=i)).isoformat(),
            quantity=20.0 + (i % 5),
            revenue=100.0,
        )
        for i in range(35)
    ]

    req = ForecastRequest(
        horizon_days=7,
        series=[
            ForecastSeriesInput(
                store_id="store_2",
                product_id="product_012",
                product_name="Atta 5kg",
                unit_price=220.0,
                current_stock=40.0,
                values=values,
            )
        ],
    )

    response = generate_forecast_response(req)
    assert len(response.results) == 1
    res = response.results[0]

    assert res.source == "model"
    assert res.product_id == "product_012"
    assert res.product_name == "Atta 5kg"
    assert res.predicted_demand_7d > 0.0
    assert res.predicted_daily_demand > 0.0
    assert res.predicted_daily_demand == pytest.approx(res.predicted_demand_7d / 7, abs=0.1)
    assert res.confidence in ("low", "medium", "high")
