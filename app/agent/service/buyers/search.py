"""
app/agent/service/buyers/search.py
====================================
Fetch buyers (customers) for a store from MongoDB.
"""
from __future__ import annotations

from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.buyers.search")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    LOGGER.warning("buyers_store_not_found", store_id=store_id)
    return None


async def fetch_buyers(store_id: str, query: str = "", status: str = "active", limit: int = 20) -> list[dict]:
    """
    Search buyers (customers) for a store by name or phone.
    Returns key contact details and financial summary.
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return []

    match: dict = {"store": store_oid}
    if status in ("active", "inactive"):
        match["status"] = status

    if query:
        match["$or"] = [
            {"name": {"$regex": query, "$options": "i"}},
            {"phone": {"$regex": query, "$options": "i"}},
        ]

    cursor = db["buyers"].find(match, {
        "name": 1, "phone": 1, "email": 1, "address": 1,
        "GSTIN": 1, "totalSales": 1, "totalPaid": 1, "totalDue": 1, "status": 1,
    }).sort("totalSales", -1).limit(limit)

    results = await cursor.to_list(length=limit)
    LOGGER.debug("fetch_buyers", store_id=store_id, count=len(results))

    output = []
    for b in results:
        output.append({
            "buyer_id": str(b["_id"]),
            "name": b.get("name", ""),
            "phone": b.get("phone", ""),
            "email": b.get("email"),
            "address": b.get("address"),
            "gstin": b.get("GSTIN"),
            "total_sales": round(b.get("totalSales", 0), 2),
            "total_paid": round(b.get("totalPaid", 0), 2),
            "total_due": round(b.get("totalDue", 0), 2),
            "status": b.get("status", "active"),
        })
    return output
