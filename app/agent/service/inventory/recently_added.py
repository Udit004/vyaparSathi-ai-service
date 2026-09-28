"""
app/agent/service/inventory/recently_added.py
=============================================
Fetch products added (created) within the last N days.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Dict, Any
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.inventory.recently_added")

DEFAULT_LOOKBACK_DAYS = 7


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    LOGGER.warning("recently_added_store_not_found", store_id=store_id)
    return None


async def fetch_recently_added_products(
    store_id: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS
) -> List[Dict[str, Any]]:
    """
    Returns products created within the last `lookback_days` days,
    sorted by creation date (newest first).

    Each item: {
        product_id, name, category, price, quantity, sku, created_at
    }
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return []

    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    cursor = db["products"].find(
        {
            "store": store_oid,
            "isActive": True,
            "createdAt": {"$gte": cutoff},
        },
        {
            "_id": 1,
            "name": 1,
            "category": 1,
            "price": 1,
            "quantity": 1,
            "sku": 1,
            "createdAt": 1,
        },
    ).sort("createdAt", -1)

    products = await cursor.to_list(length=500)
    LOGGER.debug(
        "fetch_recently_added_products",
        store_id=store_id,
        lookback_days=lookback_days,
        count=len(products),
    )

    results: List[Dict[str, Any]] = []
    for p in products:
        created = p.get("createdAt")
        results.append(
            {
                "product_id": str(p["_id"]),
                "name": p.get("name", "Unknown"),
                "category": p.get("category", "General"),
                "price": float(p.get("price", 0.0)),
                "quantity": p.get("quantity", 0),
                "sku": p.get("sku", "N/A"),
                "created_at": created.isoformat() if created else None,
                "days_ago": round((datetime.utcnow() - created).days, 1) if created else None,
            }
        )

    return results
