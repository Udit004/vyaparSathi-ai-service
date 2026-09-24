"""
app/agent/service/inventory/slow_moving.py
==========================================
Products that have stock (quantity > 0) but ZERO sales in the last
N days. Unlike dead_stock (0 qty), these are dormant products that
still occupy shelf space and working capital.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.inventory.slow_moving")

DEFAULT_LOOKBACK_DAYS = 30


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    LOGGER.warning("slow_moving_store_not_found", store_id=store_id)
    return None


async def fetch_slow_moving_products(
    store_id: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS
) -> list[dict]:
    """
    Returns products with quantity > 0 that had ZERO sales in the last
    `lookback_days` days, sorted by quantity desc (most stock tied up first).

    Each item: { product_id, name, category, quantity, value, days_since_sold }
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return []

    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    # 1. Product IDs that DID sell in the lookback window
    sales_pipeline = [
        {"$match": {"store": store_oid, "completedAt": {"$gte": cutoff}}},
        {"$unwind": "$items"},
        {"$group": {"_id": "$items.productId"}},
    ]
    sales_cursor = db["sales"].aggregate(sales_pipeline)
    sold_ids = {str(r["_id"]) async for r in sales_cursor}

    # 2. All active products with qty > 0
    prod_cursor = db["products"].find(
        {"store": store_oid, "isActive": True, "quantity": {"$gt": 0}},
        {"_id": 1, "name": 1, "category": 1, "quantity": 1, "price": 1},
    )
    products = await prod_cursor.to_list(length=2000)
    LOGGER.debug(
        "fetch_slow_moving_products",
        store_id=store_id,
        total_active=len(products),
        sold_count=len(sold_ids),
    )

    slow = []
    for p in products:
        if str(p["_id"]) not in sold_ids:
            qty = p.get("quantity", 0)
            price = p.get("price", 0.0)
            slow.append(
                {
                    "product_id": str(p["_id"]),
                    "name": p.get("name", "Unknown"),
                    "category": p.get("category", "General"),
                    "quantity": qty,
                    "tied_up_value": round(qty * price, 2),
                    "lookback_days": lookback_days,
                }
            )

    # Sort by value desc — most capital tied up first
    slow.sort(key=lambda x: x["tied_up_value"], reverse=True)
    return slow
