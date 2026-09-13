"""
app/services/agent/sales.py
==============================
Sales queries against the MongoDB ``sales`` collection.

Used by:
    - app/agent/tools/sales.py (get_sales_summary, get_top_selling_products)

All functions are async and use the shared motor client from
app.config.database.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from bson import ObjectId

import structlog

from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.sales")

# Default lookback window for sales aggregations (30 days).
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


async def fetch_top_selling_products(
    store_id: str, limit: int = 5, days_lookback: int = DEFAULT_LOOKBACK_DAYS
) -> list[dict]:
    """
    Returns the top N products by total quantity sold in the last N days.
    Each item: { product_id, name, total_quantity_sold, revenue_generated }
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
        {"$unwind": "$items"},
        {
            "$group": {
                "_id": "$items.productId",
                "total_quantity_sold": {"$sum": "$items.quantity"},
                "revenue_generated": {"$sum": "$items.lineTotal"},
            }
        },
        {"$sort": {"total_quantity_sold": -1}},
        {"$limit": limit},
        # Join with products to get name
        {
            "$lookup": {
                "from": "products",
                "let": {"pid": {"$toObjectId": "$_id"}},
                "pipeline": [
                    {"$match": {"$expr": {"$eq": ["$_id", "$$pid"]}}},
                    {"$project": {"name": 1}},
                ],
                "as": "product_info",
            }
        },
    ]

    cursor = db["sales"].aggregate(pipeline)
    results = await cursor.to_list(length=limit)
    LOGGER.debug("fetch_top_selling_products", store_id=store_id, count=len(results))

    output = []
    for r in results:
        name = "Unknown"
        if r.get("product_info"):
            name = r["product_info"][0].get("name", "Unknown")
        output.append(
            {
                "product_id": str(r["_id"]),
                "name": name,
                "total_quantity_sold": r.get("total_quantity_sold", 0),
                "revenue_generated": round(r.get("revenue_generated", 0.0), 2),
            }
        )

    return output