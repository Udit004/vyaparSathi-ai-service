"""
app/agent/service/inventory/category_stock_health.py
======================================================
Per-category breakdown of inventory health: total products, low stock,
and out-of-stock counts with percentage metrics.
"""
from __future__ import annotations

from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.inventory.category_stock_health")

DEFAULT_LOW_STOCK_THRESHOLD = 10


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    LOGGER.warning("category_health_store_not_found", store_id=store_id)
    return None


async def fetch_category_stock_health(
    store_id: str, threshold: int = DEFAULT_LOW_STOCK_THRESHOLD
) -> list[dict]:
    """
    Returns per-category inventory health summary sorted by health score asc
    (unhealthiest categories first).

    Each item:
        category, total, low_stock, out_of_stock,
        health_pct (% of products that are healthy),
        total_value (quantity * price sum)
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return []

    pipeline = [
        {"$match": {"store": store_oid, "isActive": True}},
        {
            "$group": {
                "_id": "$category",
                "total": {"$sum": 1},
                "low_stock": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$gt": ["$quantity", 0]},
                                    {"$lte": ["$quantity", threshold]},
                                ]
                            },
                            1,
                            0,
                        ]
                    }
                },
                "out_of_stock": {
                    "$sum": {"$cond": [{"$lte": ["$quantity", 0]}, 1, 0]}
                },
                "total_value": {
                    "$sum": {"$multiply": ["$quantity", "$price"]}
                },
            }
        },
        {"$sort": {"total": -1}},
    ]

    cursor = db["products"].aggregate(pipeline)
    results = await cursor.to_list(length=100)
    LOGGER.debug("fetch_category_stock_health", store_id=store_id, categories=len(results))

    output = []
    for r in results:
        total = r.get("total", 0)
        low = r.get("low_stock", 0)
        oos = r.get("out_of_stock", 0)
        healthy = total - low - oos
        health_pct = round((healthy / total) * 100, 1) if total > 0 else 100.0
        output.append(
            {
                "category": r.get("_id", "Uncategorized"),
                "total": total,
                "low_stock": low,
                "out_of_stock": oos,
                "healthy": healthy,
                "health_pct": health_pct,
                "total_value": round(r.get("total_value", 0.0), 2),
            }
        )

    # Sort unhealthiest first
    output.sort(key=lambda x: x["health_pct"])
    return output
