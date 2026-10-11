from __future__ import annotations

import json
import math
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.schemas.forecast import DailyValue, ForecastRequest, ForecastSeriesInput
from app.services.forecast_service import generate_forecast_response
from app.utils.indian_calendar import is_indian_festival_window, is_payday_window, is_weekend


def synthesize_realistic_sku(
    *,
    store_id: str,
    product_id: str,
    start_date: date,
    history_days: int,
    horizon_days: int,
    base_qty: float,
    trend: float,
    weekly_lift: float,
    festival_lift: float,
    payday_lift: float,
    intermittent: float,
) -> tuple[ForecastSeriesInput, dict]:
    total_days = history_days + horizon_days
    all_dates = [start_date + timedelta(days=i) for i in range(total_days)]
    quantities = []

    for i, cur_date in enumerate(all_dates):
        q = base_qty + (trend * i)
        if is_weekend(cur_date):
            q *= (1.0 + weekly_lift)
        if is_payday_window(cur_date):
            q *= (1.0 + payday_lift)
        if is_indian_festival_window(cur_date):
            q *= festival_lift

        # Add noise
        q += random.gauss(0, max(0.5, q * 0.08))
        if random.random() < intermittent:
            q *= random.uniform(0.0, 0.25)
        quantities.append(max(0.0, round(q, 2)))

    history_quantities = quantities[:history_days]
    actual_future = quantities[history_days:history_days + horizon_days]

    daily_entries = [
        DailyValue(date=all_dates[i].isoformat(), quantity=history_quantities[i], revenue=round(history_quantities[i] * 50.0, 2))
        for i in range(history_days)
    ]

    series_input = ForecastSeriesInput(
        store_id=store_id,
        product_id=product_id,
        product_name=f"Product {product_id}",
        unit_price=50.0,
        current_stock=100.0,
        values=daily_entries,
    )

    actual_7d = round(sum(actual_future), 2)
    actual_daily = round(actual_7d / horizon_days, 2)
    last_7_avg = mean(history_quantities[-7:]) if history_quantities else 0.0
    actual_trend = round(((actual_daily - last_7_avg) / last_7_avg * 100.0), 2) if last_7_avg else 0.0

    future_dates = all_dates[history_days:history_days + horizon_days]
    is_festival_horizon = any(is_indian_festival_window(d) for d in future_dates)

    meta = {
        "store_id": store_id,
        "product_id": product_id,
        "actual_7d": actual_7d,
        "actual_daily": actual_daily,
        "actual_trend": actual_trend,
        "is_festival_horizon": is_festival_horizon,
        "is_intermittent": intermittent > 0.08,
    }

    return series_input, meta


def run_in_depth_test(num_series: int = 100, seed: int = 42) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    print(f"Generating {num_series} diverse retail scenarios (festivals, weekends, paydays, intermittent)...")
    
    # We test with start dates that cross into festival windows and regular windows
    scenarios = []
    series_inputs = []
    
    start_candidates = [
        date(2026, 9, 1),   # Early autumn
        date(2026, 10, 15), # Pre-Diwali / Dussehra festive period
        date(2026, 11, 20), # Post-Diwali
        date(2027, 2, 10),  # Pre-Holi period
    ]

    for idx in range(num_series):
        store_id = f"store_{(idx % 8) + 1}"
        product_id = f"product_{idx + 1:03d}"
        start_dt = random.choice(start_candidates)
        
        # Mix of SKU profiles
        is_sparse = random.random() < 0.25
        base_demand = random.uniform(2.0, 12.0) if is_sparse else random.uniform(15.0, 65.0)
        weekly_lift = random.uniform(0.15, 0.40)
        festival_lift = random.uniform(1.4, 2.3)
        payday_lift = random.uniform(0.10, 0.25)
        intermittent = random.uniform(0.12, 0.30) if is_sparse else random.uniform(0.0, 0.05)

        s_input, s_meta = synthesize_realistic_sku(
            store_id=store_id,
            product_id=product_id,
            start_date=start_dt,
            history_days=45,
            horizon_days=7,
            base_qty=base_demand,
            trend=random.uniform(-0.04, 0.06),
            weekly_lift=weekly_lift,
            festival_lift=festival_lift,
            payday_lift=payday_lift,
            intermittent=intermittent,
        )
        series_inputs.append(s_input)
        scenarios.append(s_meta)

    print(f"Running inference for {num_series} series with direct multi-horizon CatBoost...")
    req = ForecastRequest(horizon_days=7, series=series_inputs)
    
    t0 = time.perf_counter()
    resp = generate_forecast_response(req)
    t1 = time.perf_counter()
    latency_total_ms = (t1 - t0) * 1000.0
    latency_per_sku_ms = latency_total_ms / max(num_series, 1)

    print(f"Total inference time: {latency_total_ms:.2f} ms ({latency_per_sku_ms:.2f} ms per SKU)")

    # Aggregate performance
    total_actual = 0.0
    total_abs_error = 0.0
    ape_list = []
    smape_list = []
    trend_correct = 0

    festival_actual = 0.0
    festival_abs_err = 0.0
    regular_actual = 0.0
    regular_abs_err = 0.0

    intermittent_actual = 0.0
    intermittent_abs_err = 0.0
    regular_sku_actual = 0.0
    regular_sku_abs_err = 0.0

    results_table = []

    for res, meta in zip(resp.results, scenarios):
        act_7d = meta["actual_7d"]
        pred_7d = res.predicted_demand_7d
        err = abs(pred_7d - act_7d)
        ape = (err / act_7d * 100.0) if act_7d > 0 else 0.0
        
        denom = pred_7d + act_7d
        smape = (2.0 * err / denom * 100.0) if denom > 0 else 0.0

        total_actual += act_7d
        total_abs_error += err
        ape_list.append(ape)
        smape_list.append(smape)

        # Trend match
        pred_trend_dir = 1 if res.trend_percent > 0 else -1 if res.trend_percent < 0 else 0
        act_trend_dir = 1 if meta["actual_trend"] > 0 else -1 if meta["actual_trend"] < 0 else 0
        if pred_trend_dir == act_trend_dir:
            trend_correct += 1

        # Festival slice
        if meta["is_festival_horizon"]:
            festival_actual += act_7d
            festival_abs_err += err
        else:
            regular_actual += act_7d
            regular_abs_err += err

        # SKU type slice
        if meta["is_intermittent"]:
            intermittent_actual += act_7d
            intermittent_abs_err += err
        else:
            regular_sku_actual += act_7d
            regular_sku_abs_err += err

        results_table.append({
            "store_id": meta["store_id"],
            "product_id": meta["product_id"],
            "predicted_7d": pred_7d,
            "actual_7d": act_7d,
            "err": round(err, 2),
            "source": res.source,
            "is_festival": meta["is_festival_horizon"],
            "is_intermittent": meta["is_intermittent"],
        })

    overall_wape = (total_abs_error / total_actual * 100.0) if total_actual else 0.0
    overall_smape = mean(smape_list)
    trend_acc = (trend_correct / num_series * 100.0)

    festival_wape = (festival_abs_err / festival_actual * 100.0) if festival_actual else 0.0
    regular_wape = (regular_abs_err / regular_actual * 100.0) if regular_actual else 0.0

    intermittent_wape = (intermittent_abs_err / intermittent_actual * 100.0) if intermittent_actual else 0.0
    regular_sku_wape = (regular_sku_abs_err / regular_sku_actual * 100.0) if regular_sku_actual else 0.0

    report = {
        "metrics": {
            "num_series": num_series,
            "latency_total_ms": round(latency_total_ms, 2),
            "latency_per_sku_ms": round(latency_per_sku_ms, 3),
            "overall_wape_percent": round(overall_wape, 2),
            "overall_smape_percent": round(overall_smape, 2),
            "trend_direction_accuracy_percent": round(trend_acc, 2),
            "festival_slice_wape_percent": round(festival_wape, 2),
            "regular_slice_wape_percent": round(regular_wape, 2),
            "regular_sku_wape_percent": round(regular_sku_wape, 2),
            "intermittent_sku_wape_percent": round(intermittent_wape, 2),
        },
        "sample_results": results_table[:10],
    }

    print("\n================== DEEP FORECAST TEST REPORT ==================")
    print(f"Overall WAPE:                     {report['metrics']['overall_wape_percent']}%")
    print(f"Overall SMAPE:                    {report['metrics']['overall_smape_percent']}%")
    print(f"Trend Direction Accuracy:         {report['metrics']['trend_direction_accuracy_percent']}%")
    print(f"Festival Period WAPE:             {report['metrics']['festival_slice_wape_percent']}%")
    print(f"Non-Festival Period WAPE:         {report['metrics']['regular_slice_wape_percent']}%")
    print(f"Regular Moving SKUs WAPE:         {report['metrics']['regular_sku_wape_percent']}%")
    print(f"Intermittent / Sparse SKUs WAPE:  {report['metrics']['intermittent_sku_wape_percent']}%")
    print(f"Per-SKU Inference Latency:        {report['metrics']['latency_per_sku_ms']} ms")
    print("===============================================================")

    out_file = ROOT_DIR / "artifacts" / "deep_forecast_test_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report saved to: {out_file}")

    return report


if __name__ == "__main__":
    run_in_depth_test(num_series=120)
