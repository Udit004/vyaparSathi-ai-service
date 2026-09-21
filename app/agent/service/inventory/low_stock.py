from __future__ import annotations

from bson import ObjectId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.inventory.low_stock")

DEFAULT_LOW_STOCK_THRESHOLD = 10

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
