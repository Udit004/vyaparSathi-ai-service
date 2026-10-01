"""
app/agent/service/buyers/dues.py
=================================
Fetch buyers with outstanding dues for a store.
"""
from __future__ import annotations

from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.buyers.dues")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    return None


async def fetch_buyer_dues(store_id: str, min_due: float = 0.01, limit: int = 20) -> dict:
    """
    Returns buyers with outstanding dues sorted by highest due amount.
    Also returns store-level totals.
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return {"buyers": [], "total_outstanding": 0.0, "buyers_with_dues": 0}

    cursor = db["buyers"].find(
        {"store": store_oid, "totalDue": {"$gt": min_due}, "status": "active"},
        {"name": 1, "phone": 1, "totalSales": 1, "totalPaid": 1, "totalDue": 1},
    ).sort("totalDue", -1).limit(limit)

    results = await cursor.to_list(length=limit)
    LOGGER.debug("fetch_buyer_dues", store_id=store_id, count=len(results))

    buyers = []
    total_outstanding = 0.0
    for b in results:
        due = round(b.get("totalDue", 0), 2)
        total_outstanding += due
        buyers.append({
            "buyer_id": str(b["_id"]),
            "name": b.get("name", ""),
            "phone": b.get("phone", ""),
            "total_sales": round(b.get("totalSales", 0), 2),
            "total_paid": round(b.get("totalPaid", 0), 2),
            "total_due": due,
        })

    return {
        "buyers": buyers,
        "total_outstanding": round(total_outstanding, 2),
        "buyers_with_dues": len(buyers),
    }
