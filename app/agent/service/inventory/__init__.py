"""
app/agent/service/inventory/__init__.py
==========================================
Inventory queries against the MongoDB ``products`` collection.

Used by:
    - app/agent/tools/inventory/__init__.py (get_inventory_summary, get_low_stock_products)

All functions are async and use the shared motor client from
app.config.database.
"""

from __future__ import annotations

from bson import ObjectId

import structlog

from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.inventory")

# Products with quantity in (0, threshold] are considered "low stock".
DEFAULT_LOW_STOCK_THRESHOLD = 10


async def fetch_inventory_summary(store_id: str) -> dict:
    """
    Returns a high-level inventory summary:
        total_products, low_stock_count, out_of_stock_count,
        total_inventory_value, low_stock_threshold_used
    """
    db = get_database()
    threshold = DEFAULT_LOW_STOCK_THRESHOLD

    pipeline = [
        {"$match": {"store": ObjectId(store_id), "isActive": True}},
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


async def fetch_low_stock_products(store_id: str, threshold: int = DEFAULT_LOW_STOCK_THRESHOLD) -> list[dict]:
    """
    Returns products whose quantity is > 0 but <= threshold, sorted by quantity asc.
    Each item: { product_id, name, category, current_quantity, price }
    """
    db = get_database()

    cursor = db["products"].find(
        {
            "store": ObjectId(store_id),
            "isActive": True,
            "quantity": {"$gt": 0, "$lte": threshold},
        },
        {"_id": 1, "name": 1, "category": 1, "quantity": 1, "price": 1},
    ).sort("quantity", 1)

    products = await cursor.to_list(length=200)
    LOGGER.debug("fetch_low_stock_products", store_id=store_id, count=len(products))

    return [
        {
            "product_id": str(p["_id"]),
            "name": p.get("name", "Unknown"),
            "category": p.get("category", "General"),
            "current_quantity": p.get("quantity", 0),
            "price": p.get("price", 0.0),
        }
        for p in products
    ]
