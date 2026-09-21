from __future__ import annotations

from datetime import datetime, timedelta
from bson import ObjectId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.sales.summary")

DEFAULT_LOOKBACK_DAYS = 30

async def fetch_sales_summary(store_id: str, days_lookback: int = DEFAULT_LOOKBACK_DAYS) -> dict:
    """
    Returns sales KPIs for the last N days:
        total_revenue, total_sales_count, average_order_value, days_covered
    """
    db = get_database()
    cutoff = datetime.utcnow() - timedelta(days=days_lookback)

    pipeline = [
        {
            "$match": {
                "store": ObjectId(store_id),
                "completedAt": {"$gte": cutoff},
            }
        },
        {
            "$project": {
                "order_total": {"$sum": "$items.lineTotal"},
            }
        },
        {
            "$group": {
                "_id": None,
                "total_revenue": {"$sum": "$order_total"},
                "total_sales_count": {"$sum": 1},
            }
        },
    ]

    cursor = db["sales"].aggregate(pipeline)
    result = await cursor.to_list(length=1)
    LOGGER.debug("fetch_sales_summary", store_id=store_id, days=days_lookback, result=result)

    if not result:
        return {
            "total_revenue": 0.0,
            "total_sales_count": 0,
            "average_order_value": 0.0,
            "days_covered": days_lookback,
        }

    r = result[0]
    count = r.get("total_sales_count", 0)
    revenue = r.get("total_revenue", 0.0)

    return {
        "total_revenue": round(revenue, 2),
        "total_sales_count": count,
        "average_order_value": round(revenue / count, 2) if count > 0 else 0.0,
        "days_covered": days_lookback,
    }
