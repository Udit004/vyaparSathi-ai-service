"""
app/agent/service/sellers/search.py
===================================
Fetch and search sellers (vendors / suppliers) for a store from MongoDB.
"""
from __future__ import annotations

from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.sellers.search")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    LOGGER.warning("sellers_store_not_found", store_id=store_id)
    return None


async def fetch_sellers(store_id: str, query: str = "", status: str = "active", limit: int = 20) -> list[dict]:
    """
    Search sellers (distributors/suppliers) for a store by contact name, business name, or phone.
    Returns contact details, GSTIN, and financial balance (total purchase, paid, due).
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
            {"businessName": {"$regex": query, "$options": "i"}},
            {"phone": {"$regex": query, "$options": "i"}},
            {"email": {"$regex": query, "$options": "i"}},
        ]

    cursor = db["sellers"].find(match, {
        "name": 1, "businessName": 1, "phone": 1, "email": 1, "address": 1,
        "GSTIN": 1, "totalPurchase": 1, "totalPaid": 1, "totalDue": 1, "status": 1,
    }).sort("totalPurchase", -1).limit(limit)

    results = await cursor.to_list(length=limit)
    LOGGER.debug("fetch_sellers", store_id=store_id, count=len(results))

    output = []
    for s in results:
        output.append({
            "seller_id": str(s["_id"]),
            "name": s.get("name", ""),
            "business_name": s.get("businessName") or s.get("name", ""),
            "phone": s.get("phone", ""),
            "email": s.get("email"),
            "address": s.get("address"),
            "gstin": s.get("GSTIN"),
            "total_purchase": round(s.get("totalPurchase", 0), 2),
            "total_paid": round(s.get("totalPaid", 0), 2),
            "total_due": round(s.get("totalDue", 0), 2),
            "status": s.get("status", "active"),
        })
    return output
