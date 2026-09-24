from __future__ import annotations

from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.inventory.summary")

# Products with quantity in (0, threshold] are considered "low stock".
DEFAULT_LOW_STOCK_THRESHOLD = 10


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    """
    Resolve a store identifier to an ObjectId.

    Accepts either a valid 24-char hex ObjectId string or a store name.
    Returns None if not found.
    """
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass

    doc = await db["stores"].find_one(
        {"name": store_id},
        {"_id": 1},
    )
    if doc:
        return doc["_id"]

    LOGGER.warning("inventory_summary_store_not_found", store_id=store_id)
    return None


async def fetch_inventory_summary(store_id: str) -> dict:
    """
    Returns a high-level inventory summary:
        total_products, low_stock_count, out_of_stock_count,
        total_inventory_value, low_stock_threshold_used
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return {
            "total_products": 0,
            "low_stock_count": 0,
            "out_of_stock_count": 0,
            "total_inventory_value": 0.0,
            "low_stock_threshold_used": DEFAULT_LOW_STOCK_THRESHOLD,
        }
    threshold = DEFAULT_LOW_STOCK_THRESHOLD

    pipeline = [
        {"$match": {"store": store_oid, "isActive": True}},
        {
            "$group": {
                "_id": None,
                "total_products": {"$sum": 1},
                "low_stock_count": {
                    "$sum": {
                        "$cond": [
                            {"$and": [
                                {"$gt": ["$quantity", 0]},
                                {"$lte": ["$quantity", threshold]},
                            ]},
                            1,
                            0,
                        ]
                    }
                },
                "out_of_stock_count": {
                    "$sum": {"$cond": [{"$lte": ["$quantity", 0]}, 1, 0]}
                },
                "total_inventory_value": {
                    "$sum": {"$multiply": ["$quantity", "$price"]}
                },
            }
        },
    ]

    cursor = db["products"].aggregate(pipeline)
    result = await cursor.to_list(length=1)
    LOGGER.debug("fetch_inventory_summary", store_id=store_id, result=result)

    if not result:
        return {
            "total_products": 0,
            "low_stock_count": 0,
            "out_of_stock_count": 0,
            "total_inventory_value": 0.0,
            "low_stock_threshold_used": threshold,
        }

    r = result[0]
    return {
        "total_products": r.get("total_products", 0),
        "low_stock_count": r.get("low_stock_count", 0),
        "out_of_stock_count": r.get("out_of_stock_count", 0),
        "total_inventory_value": round(r.get("total_inventory_value", 0.0), 2),
        "low_stock_threshold_used": threshold,
    }
