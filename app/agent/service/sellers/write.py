"""
app/agent/service/sellers/write.py
====================================
Create, update, and delete sellers directly via MongoDB.
"""
from __future__ import annotations

from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.sellers.write")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    return None


async def create_seller(store_id: str, data: dict) -> str:
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        raise ValueError(f"Store {store_id} not found.")

    doc = {
        "store": store_oid,
        "name": data["name"],
        "businessName": data.get("business_name"),
        "phone": data["phone"],
        "email": data.get("email"),
        "address": data.get("address"),
        "GSTIN": data.get("gstin"),
        "totalPurchase": 0.0,
        "totalPaid": 0.0,
        "totalDue": 0.0,
        "status": "active",
        "createdAt": datetime.now(timezone.utc),
        "updatedAt": datetime.now(timezone.utc),
    }

    result = await db["sellers"].insert_one(doc)
    LOGGER.info("seller_created", seller_id=str(result.inserted_id), store_id=store_id)
    return str(result.inserted_id)


async def update_seller(store_id: str, seller_id: str, data: dict) -> bool:
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        raise ValueError(f"Store {store_id} not found.")

    try:
        seller_oid = ObjectId(seller_id)
    except InvalidId:
        raise ValueError("Invalid seller ID.")

    update_fields = {}
    if "name" in data: update_fields["name"] = data["name"]
    if "business_name" in data: update_fields["businessName"] = data["business_name"]
    if "phone" in data: update_fields["phone"] = data["phone"]
    if "email" in data: update_fields["email"] = data["email"]
    if "address" in data: update_fields["address"] = data["address"]
    if "gstin" in data: update_fields["GSTIN"] = data["gstin"]
    if "status" in data: update_fields["status"] = data["status"]

    if not update_fields:
        return True

    update_fields["updatedAt"] = datetime.now(timezone.utc)
    
    result = await db["sellers"].update_one(
        {"_id": seller_oid, "store": store_oid},
        {"$set": update_fields}
    )
    LOGGER.info("seller_updated", seller_id=seller_id, modified=result.modified_count)
    return result.modified_count > 0


async def delete_seller(store_id: str, seller_id: str) -> bool:
    """Soft-delete seller by setting status to inactive"""
    return await update_seller(store_id, seller_id, {"status": "inactive"})
