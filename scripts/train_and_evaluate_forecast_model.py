from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from catboost import CatBoostRegressor, Pool
from app.utils.indian_calendar import (
    get_calendar_features,
    is_indian_festival_window,
    is_payday_window,
    is_weekend,
)

FEATURE_COLUMNS = [
    "store",
    "item",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "is_payday_window",
    "is_festival_window",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
]

CAT_FEATURES = ["store", "item"]


def build_features_for_anchor(
    *,
    store_id: str,
    item_id: str,
    anchor_date: date,
    history: list[float],
) -> dict:
    last_7 = history[-7:] if history else []
    last_14 = history[-14:] if history else []

    target_date = anchor_date + timedelta(days=1)
    cal = get_calendar_features(target_date)

    return {
        "store": str(store_id),
        "item": str(item_id),
        "day_of_week": cal["day_of_week"],
        "day_of_month": cal["day_of_month"],
        "month": cal["month"],
        "is_weekend": cal["is_weekend"],
        "is_payday_window": cal["is_payday_window"],
        "is_festival_window": cal["is_festival_window"],
        "lag_1": history[-1] if len(history) >= 1 else 0.0,
        "lag_2": history[-2] if len(history) >= 2 else 0.0,
        "lag_3": history[-3] if len(history) >= 3 else 0.0,
        "lag_7": history[-7] if len(history) >= 7 else 0.0,
        "lag_14": history[-14] if len(history) >= 14 else 0.0,
        "rolling_mean_7": mean(last_7) if last_7 else 0.0,
        "rolling_mean_14": mean(last_14) if last_14 else 0.0,
        "rolling_std_7": pstdev(last_7) if len(last_7) > 1 else 0.0,
    }


def synthesize_retail_series(
    *,
    start_date: date,
    total_days: int,
    base_demand: float,
    trend_rate: float,
    weekly_seasonality: float,
    festival_multiplier: float,
    noise_level: float,
    intermittent_prob: float,
) -> list[tuple[date, float]]:
    series = []
    for day_idx in range(total_days):
        current_date = start_date + timedelta(days=day_idx)

        # Base + trend
        val = base_demand + (trend_rate * day_idx)

        # Day of week pattern: Indian retail sees higher sales on Sat/Sun
        dow = current_date.weekday()
        if dow in (5, 6):
            val *= 1.0 + (weekly_seasonality * 0.45)
        elif dow in (0, 1):
            val *= 1.0 - (weekly_seasonality * 0.15)

        # Payday surge (1st-5th, 28th-31st)
        if is_payday_window(current_date):
            val *= 1.20

        # Indian Festival surge (Diwali, Holi, Navratri, Eid, etc.)
        if is_indian_festival_window(current_date):
            val *= festival_multiplier

        # Add Gaussian noise
        noise = random.gauss(0, noise_level)
        val = max(0.0, val + noise)

        # Intermittent zero-sales simulation for slow SKUs
        if random.random() < intermittent_prob:
            val *= random.uniform(0.0, 0.2)

        series.append((current_date, round(val, 2)))

    return series


def generate_training_samples(
    num_stores: int = 6,
    num_products: int = 40,
    history_days: int = 35,
    horizon_days: int = 7,
    seed: int = 42,
) -> tuple[list[list], list[list], list[list], list[list]]:
    random.seed(seed)
    np.random.seed(seed)

    base_start_date = date(2025, 7, 1)
    total_simulation_days = 200  # July 2025 to mid Jan 2026

    train_X, train_y = [], []
    test_X, test_y = [], []

    product_profiles = []
    for p_idx in range(num_products):
        product_profiles.append(
            {
                "id": f"product_{p_idx + 1:03d}",
                "base": random.uniform(5.0, 55.0),
                "trend": random.uniform(-0.03, 0.07),
                "weekly": random.uniform(0.1, 0.35),
                "festival_mult": random.uniform(1.3, 2.2),
                "noise": random.uniform(0.5, 2.5),
                "intermittent": random.uniform(0.0, 0.12),
            }
        )

    cutoff_day = total_simulation_days - 30

    for s_idx in range(num_stores):
        store_id = f"store_{s_idx + 1}"
        store_scale = random.uniform(0.85, 1.3)

        for prof in product_profiles:
            series = synthesize_retail_series(
                start_date=base_start_date,
                total_days=total_simulation_days,
                base_demand=prof["base"] * store_scale,
                trend_rate=prof["trend"],
                weekly_seasonality=prof["weekly"],
                festival_multiplier=prof["festival_mult"],
                noise_level=prof["noise"],
                intermittent_prob=prof["intermittent"],
            )

            # Slide window with step of 2 to balance coverage and train speed
            for anchor_idx in range(history_days, total_simulation_days - horizon_days, 2):
                anchor_date, _ = series[anchor_idx]
                history_window = [qty for _, qty in series[anchor_idx - history_days : anchor_idx + 1]]
                future_7d = [qty for _, qty in series[anchor_idx + 1 : anchor_idx + 1 + horizon_days]]

                feat_dict = build_features_for_anchor(
                    store_id=store_id,
                    item_id=prof["id"],
                    anchor_date=anchor_date,
                    history=history_window,
                )
                row = [feat_dict[col] for col in FEATURE_COLUMNS]

                if anchor_idx < cutoff_day:
                    train_X.append(row)
                    train_y.append(future_7d)
                else:
                    test_X.append(row)
                    test_y.append(future_7d)

    return train_X, train_y, test_X, test_y


def train_and_evaluate() -> None:
    print("Generating synthetic retail training and evaluation datasets...", flush=True)
    train_X, train_y, test_X, test_y = generate_training_samples(
        num_stores=6,
        num_products=40,
        history_days=35,
        horizon_days=7,
        seed=101,
    )
    print(f"Dataset generated. Train samples: {len(train_X)}, Test samples: {len(test_X)}", flush=True)

    train_pool = Pool(
        data=train_X,
        label=train_y,
        cat_features=CAT_FEATURES,
        feature_names=FEATURE_COLUMNS,
    )
    test_pool = Pool(
        data=test_X,
        label=test_y,
        cat_features=CAT_FEATURES,
        feature_names=FEATURE_COLUMNS,
    )

    print("Training Direct Horizon CatBoostRegressor with MultiRMSE...", flush=True)
    model = CatBoostRegressor(
        loss_function="MultiRMSE",
        eval_metric="MultiRMSE",
        iterations=350,
        learning_rate=0.08,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=42,
        early_stopping_rounds=30,
        verbose=50,
        thread_count=-1,
    )

    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    print("\n--- Model Evaluation on Test Set ---", flush=True)
    predictions = model.predict(test_pool)

    test_y_arr = np.array(test_y)
    pred_arr = np.maximum(0.0, np.array(predictions))

    # Metric 1: 7-day cumulative WAPE
    actual_7d_total = np.sum(test_y_arr, axis=1)
    pred_7d_total = np.sum(pred_arr, axis=1)

    abs_errors = np.abs(pred_7d_total - actual_7d_total)
    wape = (np.sum(abs_errors) / np.sum(actual_7d_total)) * 100.0

    # Metric 2: SMAPE
    denom = np.abs(pred_7d_total) + np.abs(actual_7d_total)
    smape = np.mean(2.0 * abs_errors / np.where(denom == 0, 1e-6, denom)) * 100.0

    # Metric 3: Daily step RMSE
    step_rmse = np.sqrt(np.mean((pred_arr - test_y_arr) ** 2, axis=0))

    print(f"Cumulative 7-Day WAPE: {wape:.2f}%", flush=True)
    print(f"Cumulative 7-Day SMAPE: {smape:.2f}%", flush=True)
    print("Direct Horizon Step-by-Step RMSE (Days 1 to 7):", flush=True)
    for d_idx, err in enumerate(step_rmse, 1):
        print(f"  Day +{d_idx}: {err:.2f}", flush=True)

    # Feature Importance
    importances = model.get_feature_importance(train_pool)
    print("\nFeature Importances:", flush=True)
    for col, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: x[1], reverse=True):
        print(f"  {col:20s}: {imp:.2f}%", flush=True)

    # Save artifacts
    model_dir = ROOT_DIR / "app" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "catboost_model.cbm"
    model.save_model(str(model_path))
    print(f"\nModel saved successfully to: {model_path}", flush=True)

    feature_cols_path = ROOT_DIR / "feature_columns.json"
    feature_cols_path.write_text(json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8")
    print(f"Feature columns saved to: {feature_cols_path}", flush=True)


if __name__ == "__main__":
    train_and_evaluate()
