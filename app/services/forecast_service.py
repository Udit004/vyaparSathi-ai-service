from __future__ import annotations

import json
import hashlib
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, pstdev
from threading import Lock
from typing import Any

from app.schemas.forecast import ForecastRequest, ForecastResponse, ForecastResult
from app.utils.forecasting import confidence_bucket, forecast_series

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover - optional runtime dependency in dev
    CatBoostRegressor = None


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "app" / "model" / "catboost_model.cbm"
FEATURE_COLUMNS_PATH = ROOT_DIR / "feature_columns.json"
STORE_ENCODER_PATH = ROOT_DIR / "store_encoder_map.json"
ITEM_ENCODER_PATH = ROOT_DIR / "item_encoder_map.json"

_MODEL: Any | None = None
_FEATURE_COLUMNS: list[str] | None = None
_MODEL_LOAD_ERROR: Exception | None = None
_STORE_ENCODER_MAP: dict[str, float] | None = None
_ITEM_ENCODER_MAP: dict[str, float] | None = None
_MODEL_LOCK = Lock()
LOGGER = logging.getLogger("vyaparsathi.ai.forecast")
LOGGER.setLevel(logging.INFO)


def _debug_log(message: str) -> None:
    print(f"[FORECAST DEBUG] {message}")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_prediction_scalar(raw_prediction: Any) -> float:
    if isinstance(raw_prediction, (list, tuple)):
        if not raw_prediction:
            return 0.0
        return _safe_float(raw_prediction[0])

    if hasattr(raw_prediction, "item"):
        try:
            return _safe_float(raw_prediction.item())
        except Exception:
            pass

    if hasattr(raw_prediction, "__len__") and hasattr(raw_prediction, "__getitem__"):
        try:
            if len(raw_prediction) == 0:
                return 0.0
            return _safe_float(raw_prediction[0])
        except Exception:
            return _safe_float(raw_prediction)

    return _safe_float(raw_prediction)


def _stable_id_to_float(value: Any) -> float:
    raw = str(value or "")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return float(int(digest, 16) % 1_000_000)


def _load_encoder_map(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    encoded: dict[str, float] = {}
    for key, value in payload.items():
        try:
            encoded[str(key)] = float(value)
        except (TypeError, ValueError):
            continue

    return encoded or None


def _encode_with_map(value: Any, encoder_map: dict[str, float] | None) -> float:
    raw = str(value or "")
    if encoder_map and raw in encoder_map:
        return encoder_map[raw]
    return _stable_id_to_float(raw)


def _series_anchor_date(values: list) -> date:
    if not values:
        return datetime.utcnow().date()

    try:
        parsed_dates = [datetime.fromisoformat(item.date).date() for item in values if item.date]
        if parsed_dates:
            return max(parsed_dates)
    except ValueError:
        pass

    return datetime.utcnow().date()


def _lazy_load_artifacts() -> tuple[Any | None, list[str] | None]:
    global _MODEL, _FEATURE_COLUMNS, _MODEL_LOAD_ERROR
    global _STORE_ENCODER_MAP, _ITEM_ENCODER_MAP

    if _MODEL_LOAD_ERROR is not None:
        LOGGER.warning("Forecast model is unavailable because a previous load attempt failed: %s", _MODEL_LOAD_ERROR)
        _debug_log(f"model unavailable after previous failure: {_MODEL_LOAD_ERROR}")
        return None, None

    if _MODEL is not None and _FEATURE_COLUMNS is not None:
        LOGGER.info("Forecast model already loaded and cached in memory")
        _debug_log("model already loaded and cached in memory")
        return _MODEL, _FEATURE_COLUMNS

    with _MODEL_LOCK:
        if _MODEL is not None and _FEATURE_COLUMNS is not None:
            LOGGER.info("Forecast model became available while waiting for the lock")
            _debug_log("model became available while waiting for the lock")
            return _MODEL, _FEATURE_COLUMNS

        try:
            LOGGER.info(
                "Loading forecast artifacts from model=%s feature_columns=%s store_encoder=%s item_encoder=%s",
                MODEL_PATH,
                FEATURE_COLUMNS_PATH,
                STORE_ENCODER_PATH,
                ITEM_ENCODER_PATH,
            )
            _debug_log(
                f"loading artifacts model={MODEL_PATH} feature_columns={FEATURE_COLUMNS_PATH} "
                f"store_encoder={STORE_ENCODER_PATH} item_encoder={ITEM_ENCODER_PATH}"
            )
            if CatBoostRegressor is None:
                raise RuntimeError("catboost package is not installed")
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            if not FEATURE_COLUMNS_PATH.exists():
                raise FileNotFoundError(
                    f"Feature columns file not found at {FEATURE_COLUMNS_PATH}"
                )

            with FEATURE_COLUMNS_PATH.open("r", encoding="utf-8") as file_handle:
                loaded_columns = json.load(file_handle)

            if not isinstance(loaded_columns, list) or not all(
                isinstance(column, str) for column in loaded_columns
            ):
                raise ValueError("feature_columns.json must be a JSON array of strings")

            model = CatBoostRegressor()
            model.load_model(str(MODEL_PATH))

            _FEATURE_COLUMNS = loaded_columns
            _MODEL = model
            _STORE_ENCODER_MAP = _load_encoder_map(STORE_ENCODER_PATH)
            _ITEM_ENCODER_MAP = _load_encoder_map(ITEM_ENCODER_PATH)
            LOGGER.info(
                "Forecast model loaded successfully with %s feature columns. store_encoder=%s item_encoder=%s",
                len(_FEATURE_COLUMNS),
                "yes" if _STORE_ENCODER_MAP else "no",
                "yes" if _ITEM_ENCODER_MAP else "no",
            )
            _debug_log(
                f"model loaded successfully with {len(_FEATURE_COLUMNS)} feature columns; "
                f"store_encoder={'yes' if _STORE_ENCODER_MAP else 'no'} "
                f"item_encoder={'yes' if _ITEM_ENCODER_MAP else 'no'}"
            )
        except Exception as exc:
            _MODEL_LOAD_ERROR = exc
            LOGGER.exception("Forecast model load failed; fallback forecasting will be used")
            _debug_log(f"model load failed; fallback will be used: {exc}")
            return None, None

    return _MODEL, _FEATURE_COLUMNS


def _build_feature_map(
    *,
    store_value: float,
    item_value: float,
    target_date: date,
    history: list[float],
) -> dict[str, Any]:
    last_7 = history[-7:] if history else []
    last_14 = history[-14:] if history else []

    feature_map: dict[str, Any] = {
        "store": store_value,
        "item": item_value,
        "day_of_week": target_date.weekday(),
        "month": target_date.month,
        "lag_1": history[-1] if len(history) >= 1 else 0.0,
        "lag_2": history[-2] if len(history) >= 2 else 0.0,
        "lag_3": history[-3] if len(history) >= 3 else 0.0,
        "lag_7": history[-7] if len(history) >= 7 else 0.0,
        "lag_14": history[-14] if len(history) >= 14 else 0.0,
        "rolling_mean_7": mean(last_7) if last_7 else 0.0,
        "rolling_mean_14": mean(last_14) if last_14 else 0.0,
        "rolling_std_7": pstdev(last_7) if len(last_7) > 1 else 0.0,
    }

    return feature_map


def _predict_with_model(
    *,
    model: Any,
    feature_columns: list[str],
    store_id: str,
    item_id: str,
    history_values: list[float],
    horizon_days: int,
    anchor_date: date,
) -> tuple[float, float, float]:
    history = history_values[:] if history_values else [0.0]
    horizon_predictions: list[float] = []
    encoded_store = _encode_with_map(store_id, _STORE_ENCODER_MAP)
    encoded_item = _encode_with_map(item_id, _ITEM_ENCODER_MAP)

    for step in range(1, horizon_days + 1):
        target = anchor_date + timedelta(days=step)
        feature_map = _build_feature_map(
            store_value=encoded_store,
            item_value=encoded_item,
            target_date=target,
            history=history,
        )

        row = [feature_map.get(column, 0.0) for column in feature_columns]
        raw_pred = model.predict([row])
        prediction = max(0.0, _extract_prediction_scalar(raw_pred))

        horizon_predictions.append(prediction)
        history.append(prediction)

    predicted_demand = round(sum(horizon_predictions), 2)
    predicted_daily = round(predicted_demand / max(horizon_days, 1), 2)
    baseline_avg = mean(history_values[-7:]) if history_values else 0.0
    trend_percent = (
        round(((predicted_daily - baseline_avg) / baseline_avg) * 100, 2)
        if baseline_avg
        else 100.0
        if predicted_daily > 0
        else 0.0
    )

    return predicted_daily, predicted_demand, trend_percent


def generate_forecast_response(request: ForecastRequest) -> ForecastResponse:
    model, feature_columns = _lazy_load_artifacts()
    LOGGER.info(
        "Generating forecast for %s series with horizon_days=%s using %s",
        len(request.series),
        request.horizon_days,
        "model" if model is not None and feature_columns is not None else "fallback",
    )
    _debug_log(
        f"generating forecast series={len(request.series)} horizon_days={request.horizon_days} "
        f"using={'model' if model is not None and feature_columns is not None else 'fallback'}"
    )
    results = []

    for series in request.series:
        values = [entry.quantity for entry in series.values]
        source = "fallback"

        if model is not None and feature_columns is not None:
            anchor_date = _series_anchor_date(series.values)
            LOGGER.info(
                "Running model forecast for store=%s product=%s history_points=%s anchor_date=%s",
                series.store_id,
                series.product_id,
                len(values),
                anchor_date,
            )
            _debug_log(
                f"running model forecast store={series.store_id} product={series.product_id} "
                f"history_points={len(values)} anchor_date={anchor_date}"
            )
            predicted_daily, predicted_demand, trend_percent = _predict_with_model(
                model=model,
                feature_columns=feature_columns,
                store_id=series.store_id,
                item_id=series.product_id,
                history_values=values,
                horizon_days=request.horizon_days,
                anchor_date=anchor_date,
            )
            forecast_result = {
                "predicted_daily_demand": predicted_daily,
                "predicted_demand": predicted_demand,
                "trend_percent": trend_percent,
                "confidence": confidence_bucket(values),
            }
            source = "model"
        else:
            forecast_result = forecast_series(values, request.horizon_days)
            LOGGER.warning(
                "Using fallback forecast for store=%s product=%s because the model is not available",
                series.store_id,
                series.product_id,
            )
            _debug_log(
                f"using fallback for store={series.store_id} product={series.product_id} because model is unavailable"
            )

        LOGGER.info(
            "forecast source=%s store=%s product=%s demand7d=%.2f daily=%.2f",
            source,
            series.store_id,
            series.product_id,
            forecast_result["predicted_demand"],
            forecast_result["predicted_daily_demand"],
        )

        results.append(
            ForecastResult(
                product_id=series.product_id,
                product_name=series.product_name,
                predicted_daily_demand=forecast_result["predicted_daily_demand"],
                predicted_demand_7d=forecast_result["predicted_demand"],
                trend_percent=forecast_result["trend_percent"],
                confidence=forecast_result["confidence"],
                source=source,
            )
        )

    return ForecastResponse(results=results)
