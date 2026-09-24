"""
app/agent/service/sales/daily_trend.py
=======================================
Day-by-day revenue and transaction count for the last N days.
Used to identify growth/decline trends and weekly patterns.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.sales.daily_trend")

DEFAULT_LOOKBACK_DAYS = 30


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    LOGGER.warning("daily_trend_store_not_found", store_id=store_id)
    return None


async def fetch_daily_sales_trend(
    store_id: str, days: int = DEFAULT_LOOKBACK_DAYS
) -> dict:
    """
    Returns day-by-day revenue and transaction count for the last `days` days.

    Output:
        {
          "days": [...],           # list of date strings (YYYY-MM-DD)
          "revenue": [...],        # float per day
          "transactions": [...],   # int per day
          "peak_day": {"date": ..., "revenue": ...},
          "avg_daily_revenue": float,
          "trend": "up" | "down" | "flat"
        }
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return {"days": [], "revenue": [], "transactions": [], "trend": "flat"}

    cutoff = datetime.utcnow() - timedelta(days=days)

    pipeline = [
        {"$match": {"store": store_oid, "completedAt": {"$gte": cutoff}}},
        {
            "$project": {
                "day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$completedAt"}},
                "order_total": {"$sum": "$items.lineTotal"},
            }
        },
        {
            "$group": {
                "_id": "$day",
                "revenue": {"$sum": "$order_total"},
                "transactions": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    cursor = db["sales"].aggregate(pipeline)
    results = await cursor.to_list(length=days + 5)
    LOGGER.debug("fetch_daily_sales_trend", store_id=store_id, points=len(results))

    if not results:
        return {"days": [], "revenue": [], "transactions": [], "trend": "flat"}

    day_list = [r["_id"] for r in results]
    revenue_list = [round(r["revenue"], 2) for r in results]
    tx_list = [r["transactions"] for r in results]

    avg_rev = round(sum(revenue_list) / len(revenue_list), 2) if revenue_list else 0.0
    peak = max(results, key=lambda r: r["revenue"])

    # Simple trend: compare first half vs second half average
    mid = len(revenue_list) // 2
    if mid > 0:
        first_half_avg = sum(revenue_list[:mid]) / mid
        second_half_avg = sum(revenue_list[mid:]) / len(revenue_list[mid:])
        if second_half_avg > first_half_avg * 1.05:
            trend = "up"
        elif second_half_avg < first_half_avg * 0.95:
            trend = "down"
        else:
            trend = "flat"
    else:
        trend = "flat"

    return {
        "days": day_list,
        "revenue": revenue_list,
        "transactions": tx_list,
        "peak_day": {"date": peak["_id"], "revenue": round(peak["revenue"], 2)},
        "avg_daily_revenue": avg_rev,
        "trend": trend,
        "lookback_days": days,
    }
