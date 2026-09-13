"""
app/services/agent/forecast.py
=================================
Restock-priority and demand-forecast queries.

Used by:
    - app/agent/tools/forecast.py (get_restock_priorities, get_forecast_summary)

Restock priorities are computed here from raw stock + real average daily
sales (no ML model). Demand forecasts delegate to
``app.services.aggregation_service.get_forecast_for_store``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from bson import ObjectId

import structlog

from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.forecast")

# Days between ordering and receiving stock.
LEAD_TIME_DAYS = 3
# Extra safety-stock buffer (days of average sales).
SAFETY_STOCK_DAYS = 7
# Lookback window for computing average daily sales.
SALES_LOOKBACK_DAYS = 30
# Maximum products returned by the sales aggregation step.
MAX_SALES_RESULTS = 5000
# Maximum products fetched for restock computation.
MAX_PRODUCTS = 2000


async def fetch_restock_priorities(store_id: str) -> list[dict]:
    """
    Computes restock priority per product using actual stock + real avg daily sales.

    Priority:
        RED    — out of stock OR days_to_stockout <= lead_time
        YELLOW — days_to_stockout <= lead_time * 2
        GREEN  — healthy stock

    Each item: { product_id, name, current_quantity, avg_daily_sales,
                 days_to_stockout, suggested_restock_quantity, priority }
    """
    db = get_database()
    cutoff = datetime.utcnow() - timedelta(days=SALES_LOOKBACK_DAYS)

    # 1. Avg daily sales per product over last 30 days
    sales_pipeline = [
        {
            "$match": {
                "store": ObjectId(store_id),
                "completedAt": {"$gte": cutoff},
            }
        },
        {"$unwind": "$items"},
        {
            "$group": {
                "_id": "$items.productId",
                "total_qty_sold": {"$sum": "$items.quantity"},
            }
        },
    ]
    sales_cursor = db["sales"].aggregate(sales_pipeline)
    sales_list = await sales_cursor.to_list(length=MAX_SALES_RESULTS)
    sales_map: dict[str, float] = {
        str(s["_id"]): s["total_qty_sold"] / SALES_LOOKBACK_DAYS for s in sales_list
    }

    # 2. All active products
    prod_cursor = db["products"].find(
        {"store": ObjectId(store_id), "isActive": True},
        {"_id": 1, "name": 1, "quantity": 1, "price": 1},
    )
    products = await prod_cursor.to_list(length=MAX_PRODUCTS)
    LOGGER.debug("fetch_restock_priorities", store_id=store_id, product_count=len(products))

    output = []
    for p in products:
        pid = str(p["_id"])
        current_qty = p.get("quantity", 0)
        avg_daily = sales_map.get(pid, 0.0)

        if avg_daily > 0:
            days_to_stockout = round(current_qty / avg_daily, 1)
        else:
            days_to_stockout = None  # No sales data — can't predict

        # Suggested qty to cover lead time + safety stock buffer
        target_stock = avg_daily * (LEAD_TIME_DAYS + SAFETY_STOCK_DAYS)
        suggested_qty = max(0, round(target_stock - current_qty))

        if current_qty <= 0:
            priority = "RED"
        elif days_to_stockout is not None and days_to_stockout <= LEAD_TIME_DAYS:
            priority = "RED"
        elif days_to_stockout is not None and days_to_stockout <= LEAD_TIME_DAYS * 2:
            priority = "YELLOW"
        elif suggested_qty > 0:
            priority = "YELLOW"
        else:
            priority = "GREEN"

        output.append(
            {
                "product_id": pid,
                "name": p.get("name", "Unknown"),
                "current_quantity": current_qty,
                "avg_daily_sales": round(avg_daily, 2),
                "days_to_stockout": days_to_stockout,
                "suggested_restock_quantity": suggested_qty,
                "priority": priority,
            }
        )

    # Sort: RED first, then YELLOW, then GREEN, then by days_to_stockout asc
    priority_rank = {"RED": 0, "YELLOW": 1, "GREEN": 2}
    output.sort(
        key=lambda x: (
            priority_rank.get(x["priority"], 3),
            x["days_to_stockout"] if x["days_to_stockout"] is not None else 9999,
        )
    )

    return output


async def fetch_forecast_summary(store_id: str, horizon_days: int = 7) -> list[dict]:
    """
    Returns a demand forecast per product using avg daily sales * horizon_days.
    Reuses existing aggregation_service logic but returns a clean dict list.
    Each item: { product_id, name, current_stock, predicted_demand, days_to_stockout }
    """
    from app.services.aggregation_service import get_forecast_for_store

    try:
        forecasts = await get_forecast_for_store(store_id)
    except Exception as exc:
        LOGGER.error("fetch_forecast_summary_error", store_id=store_id, error=str(exc))
        return []

    return [
        {
            "product_id": f["productId"],
            "name": f["productName"],
            "current_stock": f["currentStock"],
            "predicted_demand": round(f["predictedDailyDemand"] * horizon_days, 1),
            "predicted_daily": f["predictedDailyDemand"],
            "trend_percent": f["trendPercent"],
            "days_to_stockout": f["daysToStockout"],
            "horizon_days": horizon_days,
        }
        for f in forecasts
    ]