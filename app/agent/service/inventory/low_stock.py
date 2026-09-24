from __future__ import annotations

from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.inventory.low_stock")

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

    LOGGER.warning("low_stock_store_not_found", store_id=store_id)
    return None


async def fetch_low_stock_products(store_id: str, threshold: int = DEFAULT_LOW_STOCK_THRESHOLD, limit: int = 50) -> list[dict]:
    """
    Returns products whose quantity is > 0 but <= threshold, sorted by quantity asc.
    Each item: { product_id, name, category, current_quantity, price }
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return []

    cursor = db["products"].find(
        {
            "store": store_oid,
            "isActive": True,
            "quantity": {"$gt": 0, "$lte": threshold},
        },
        {"_id": 1, "name": 1, "category": 1, "quantity": 1, "price": 1},
    ).sort("quantity", 1)

    products = await cursor.to_list(length=min(limit, 500))
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
