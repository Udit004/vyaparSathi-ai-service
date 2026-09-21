from __future__ import annotations

from datetime import datetime, timedelta
from bson import ObjectId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.sales.top_products")

DEFAULT_LOOKBACK_DAYS = 30

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
