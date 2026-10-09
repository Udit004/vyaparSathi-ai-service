from __future__ import annotations
from typing import Any, Dict, Optional
from bson import ObjectId
from bson.errors import InvalidId
import structlog

from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.stores.summary")


async def fetch_store_summary(store_id: str) -> Dict[str, Any]:
    """
    Fetch comprehensive store and owner summary from MongoDB.
    """
    db = get_database()
    store_doc = None

    # 1. Resolve Store Document
    if ObjectId.is_valid(store_id):
        try:
            store_doc = await db["stores"].find_one({"_id": ObjectId(store_id)})
        except Exception:
            pass

    if not store_doc:
        store_doc = await db["stores"].find_one({
            "$or": [
                {"name": store_id},
                {"storeName": store_id},
                {"_id": store_id},
            ]
        })

    if not store_doc:
        LOGGER.warning("store_summary_not_found", store_id=store_id)
        return {
            "store_id": store_id,
            "name": "Vyapar Store",
            "owner_name": "Store Owner",
            "location": "India",
            "status": "ACTIVE",
            "phone": None,
            "email": None,
            "business_type": "retail",
        }

    store_name = store_doc.get("name") or store_doc.get("storeName") or "Vyapar Store"
    store_oid = store_doc.get("_id")
    is_active = store_doc.get("isActive", True)
    status = "ACTIVE" if is_active else "INACTIVE"
    business_type = store_doc.get("businessType", "retail")
    phone = store_doc.get("phone")
    email = store_doc.get("email")

    address_data = store_doc.get("address") or {}
    if isinstance(address_data, dict):
        location = address_data.get("fullAddress") or address_data.get("city") or address_data.get("state") or "India"
    else:
        location = str(address_data) or "India"

    # 2. Resolve Store Owner Name & Email
    owner_name = "Store Owner"
    owner_email = email

    owner_ref = store_doc.get("owner")
    owner_fb_uid = store_doc.get("ownerFirebaseUid")

    user_query_clauses = []
    if owner_ref:
        if isinstance(owner_ref, ObjectId):
            user_query_clauses.append({"_id": owner_ref})
        elif ObjectId.is_valid(str(owner_ref)):
            user_query_clauses.append({"_id": ObjectId(str(owner_ref))})
    if owner_fb_uid:
        user_query_clauses.append({"firebaseUid": owner_fb_uid})
        user_query_clauses.append({"uid": owner_fb_uid})

    if user_query_clauses:
        try:
            user_doc = await db["users"].find_one({"$or": user_query_clauses})
            if user_doc:
                owner_name = user_doc.get("name") or user_doc.get("displayName") or user_doc.get("fullName") or owner_name
                owner_email = user_doc.get("email") or owner_email
        except Exception as exc:
            LOGGER.warning("store_summary_owner_fetch_failed", error=str(exc))

    # 3. Fast Product Counts
    total_products = 0
    low_stock_count = 0
    if store_oid:
        try:
            total_products = await db["products"].count_documents({"store": store_oid, "isActive": {"$ne": False}})
            low_threshold = (store_doc.get("settings") or {}).get("lowStockThreshold", 10)
            low_stock_count = await db["products"].count_documents({
                "store": store_oid,
                "isActive": {"$ne": False},
                "$expr": {"$lte": ["$stock", {"$ifNull": ["$minStock", low_threshold]}]}
            })
        except Exception:
            pass

    LOGGER.info("store_summary_fetched", store_name=store_name, owner_name=owner_name)

    return {
        "store_id": str(store_oid) if store_oid else store_id,
        "name": store_name,
        "owner_name": owner_name,
        "owner_email": owner_email,
        "location": location,
        "status": status,
        "phone": phone,
        "business_type": business_type,
        "total_products": total_products,
        "low_stock_count": low_stock_count,
    }
