from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.schemas.forecast import DailyValue, ForecastRequest, ForecastSeriesInput
from app.services.forecast_service import generate_forecast_response


@dataclass
class SeriesTruth:
    product_id: str
    store_id: str
    history_values: list[float]
    actual_next_7d: float
    actual_daily_next_7d: float
    actual_trend_percent: float


def _clamp_non_negative(value: float) -> float:
    return max(0.0, round(value, 2))


def _safe_pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100.0


def _synthesize_daily_value(
    day_index: int,
    base: float,
    trend_per_day: float,
    weekly_amplitude: float,
    phase: float,
    noise_sigma: float,
    demand_dropout_prob: float,
) -> float:
    seasonal = weekly_amplitude * math.sin((2 * math.pi * day_index / 7.0) + phase)
    trend = trend_per_day * day_index
    noise = random.gauss(0, noise_sigma)

    value = base + seasonal + trend + noise

    if random.random() < demand_dropout_prob:
        value *= random.uniform(0.0, 0.25)

    if random.random() < 0.05:
        value += random.uniform(5.0, 25.0)

    return _clamp_non_negative(value)


def _make_one_series(
    *,
    series_index: int,
    history_days: int,
    horizon_days: int,
    start_date: date,
) -> tuple[ForecastSeriesInput, SeriesTruth, dict]:
    store_id = f"store_{(series_index % 6) + 1}"
    product_id = f"product_{series_index + 1:03d}"
    product_name = f"Seed Product {series_index + 1:03d}"

    base = random.uniform(4.0, 50.0)
    trend_per_day = random.uniform(-0.15, 0.40)
    weekly_amplitude = random.uniform(0.5, 10.0)
    phase = random.uniform(0.0, 2 * math.pi)
    noise_sigma = random.uniform(0.2, 3.5)
    demand_dropout_prob = random.uniform(0.0, 0.12)

    total_days = history_days + horizon_days
    all_values: list[float] = []
    daily_entries: list[DailyValue] = []

    for offset in range(total_days):
        qty = _synthesize_daily_value(
            day_index=offset,
            base=base,
            trend_per_day=trend_per_day,
            weekly_amplitude=weekly_amplitude,
            phase=phase,
            noise_sigma=noise_sigma,
            demand_dropout_prob=demand_dropout_prob,
        )
        all_values.append(qty)

        if offset < history_days:
            current_date = start_date + timedelta(days=offset)
            unit_price = random.uniform(8.0, 200.0)
            daily_entries.append(
                DailyValue(
                    date=current_date.isoformat(),
                    quantity=qty,
                    revenue=round(qty * unit_price, 2),
                )
            )

    history_values = all_values[:history_days]
    future_values = all_values[history_days : history_days + horizon_days]

    last_7_history_avg = mean(history_values[-7:]) if len(history_values) >= 7 else mean(history_values)
    next_7_avg = mean(future_values) if future_values else 0.0
    actual_trend_percent = _safe_pct(next_7_avg - last_7_history_avg, last_7_history_avg) if last_7_history_avg else 0.0

    payload_series = ForecastSeriesInput(
        store_id=store_id,
        product_id=product_id,
        product_name=product_name,
        current_stock=round(random.uniform(20.0, 250.0), 2),
        unit_price=round(random.uniform(10.0, 180.0), 2),
        basis="synthetic-seed",
        values=daily_entries,
    )

    truth = SeriesTruth(
        product_id=product_id,
        store_id=store_id,
        history_values=history_values,
        actual_next_7d=round(sum(future_values), 2),
        actual_daily_next_7d=round(next_7_avg, 2),
        actual_trend_percent=round(actual_trend_percent, 2),
    )

    raw_seed = {
        "store_id": store_id,
        "product_id": product_id,
        "product_name": product_name,
        "history_quantities": history_values,
        "actual_future_quantities": future_values,
        "signal": {
            "base": round(base, 4),
            "trend_per_day": round(trend_per_day, 4),
            "weekly_amplitude": round(weekly_amplitude, 4),
            "phase": round(phase, 4),
            "noise_sigma": round(noise_sigma, 4),
            "demand_dropout_prob": round(demand_dropout_prob, 4),
        },
    }

    return payload_series, truth, raw_seed


def _smape(predicted: float, actual: float) -> float:
    denominator = abs(predicted) + abs(actual)
    if denominator == 0:
        return 0.0
    return (2.0 * abs(predicted - actual) / denominator) * 100.0


def run_validation(
    *,
    series_count: int,
    history_days: int,
    horizon_days: int,
    seed: int,
) -> dict:
    random.seed(seed)

    start_date = date.today() - timedelta(days=history_days + 30)

    payload_series_list: list[ForecastSeriesInput] = []
    truths: dict[str, SeriesTruth] = {}
    raw_seed_dataset: list[dict] = []

    for index in range(series_count):
        payload_series, truth, raw_seed = _make_one_series(
            series_index=index,
            history_days=history_days,
            horizon_days=horizon_days,
            start_date=start_date,
        )
        payload_series_list.append(payload_series)
        truths[truth.product_id] = truth
        raw_seed_dataset.append(raw_seed)

    request = ForecastRequest(horizon_days=horizon_days, series=payload_series_list)
    response = generate_forecast_response(request)

    rows: list[dict] = []
    total_actual = 0.0
    total_abs_error = 0.0
    all_sources = set()
    non_negative_predictions = True
    trend_direction_matches = 0

    for result in response.results:
        truth = truths[result.product_id]

        predicted_7d = float(result.predicted_demand_7d)
        actual_7d = truth.actual_next_7d
        abs_error = abs(predicted_7d - actual_7d)
        ape = (abs_error / actual_7d * 100.0) if actual_7d > 0 else 0.0

        predicted_trend_sign = 1 if result.trend_percent > 0 else -1 if result.trend_percent < 0 else 0
        actual_trend_sign = 1 if truth.actual_trend_percent > 0 else -1 if truth.actual_trend_percent < 0 else 0
        trend_match = predicted_trend_sign == actual_trend_sign
        if trend_match:
            trend_direction_matches += 1

        if predicted_7d < 0 or result.predicted_daily_demand < 0:
            non_negative_predictions = False

        all_sources.add(result.source)
        total_actual += actual_7d
        total_abs_error += abs_error

        rows.append(
            {
                "store_id": truth.store_id,
                "product_id": result.product_id,
                "source": result.source,
                "predicted_7d": round(predicted_7d, 2),
                "actual_7d": round(actual_7d, 2),
                "absolute_error": round(abs_error, 2),
                "ape_percent": round(ape, 2),
                "smape_percent": round(_smape(predicted_7d, actual_7d), 2),
                "predicted_daily": round(float(result.predicted_daily_demand), 2),
                "actual_daily": round(float(truth.actual_daily_next_7d), 2),
                "predicted_trend_percent": round(float(result.trend_percent), 2),
                "actual_trend_percent": round(float(truth.actual_trend_percent), 2),
                "trend_sign_match": trend_match,
                "confidence": result.confidence,
            }
        )

    row_count = len(rows)
    wape_percent = round((total_abs_error / total_actual) * 100.0, 2) if total_actual > 0 else 0.0
    mape_percent = round(mean([row["ape_percent"] for row in rows]), 2) if rows else 0.0
    smape_percent = round(mean([row["smape_percent"] for row in rows]), 2) if rows else 0.0
    trend_accuracy_percent = round((trend_direction_matches / row_count) * 100.0, 2) if row_count else 0.0

    source_mode = "model" if all_sources == {"model"} else "fallback" if all_sources == {"fallback"} else "mixed"

    if wape_percent <= 20:
        quality_band = "excellent"
    elif wape_percent <= 35:
        quality_band = "good"
    elif wape_percent <= 50:
        quality_band = "usable"
    else:
        quality_band = "poor"

    report = {
        "summary": {
            "series_count": row_count,
            "history_days": history_days,
            "horizon_days": horizon_days,
            "seed": seed,
            "source_mode": source_mode,
            "all_sources": sorted(all_sources),
            "wape_percent": wape_percent,
            "mape_percent": mape_percent,
            "smape_percent": smape_percent,
            "trend_direction_accuracy_percent": trend_accuracy_percent,
            "non_negative_predictions": non_negative_predictions,
            "quality_band": quality_band,
        },
        "top_10_worst_predictions": sorted(rows, key=lambda row: row["absolute_error"], reverse=True)[:10],
        "rows": rows,
        "seed_dataset": raw_seed_dataset,
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic seed data and validate forecast model quality.")
    parser.add_argument("--series-count", type=int, default=80, help="Number of synthetic product series")
    parser.add_argument("--history-days", type=int, default=45, help="Historical days given to model")
    parser.add_argument("--horizon-days", type=int, default=7, help="Forecast horizon days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/forecast_seed_validation_report.json",
        help="Path where validation report JSON will be written",
    )
    args = parser.parse_args()

    report = run_validation(
        series_count=args.series_count,
        history_days=args.history_days,
        horizon_days=args.horizon_days,
        seed=args.seed,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = report["summary"]
    print("Forecast seed validation completed")
    print(f"source_mode={summary['source_mode']}")
    print(f"series_count={summary['series_count']}")
    print(f"wape_percent={summary['wape_percent']}")
    print(f"mape_percent={summary['mape_percent']}")
    print(f"smape_percent={summary['smape_percent']}")
    print(f"trend_direction_accuracy_percent={summary['trend_direction_accuracy_percent']}")
    print(f"quality_band={summary['quality_band']}")
    print(f"non_negative_predictions={summary['non_negative_predictions']}")
    print(f"report_path={output_path}")


if __name__ == "__main__":
    main()
