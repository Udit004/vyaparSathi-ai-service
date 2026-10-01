"""
app/agent/service/expenses/write.py
=====================================
Create, update, and delete expenses via MongoDB.
"""
from __future__ import annotations

from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.expenses.write")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    return None


async def create_expense(store_id: str, data: dict) -> str:
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        raise ValueError(f"Store {store_id} not found.")

    doc = {
        "store": store_oid,
        "title": data["title"],
        "category": data["category"],
        "amount": float(data["amount"]),
        "paymentMethod": data.get("payment_method", "Cash"),
        "description": data.get("description", ""),
        "date": datetime.now(timezone.utc),
        "createdAt": datetime.now(timezone.utc),
        "updatedAt": datetime.now(timezone.utc),
    }

    if data.get("date"):
        doc["date"] = datetime.fromisoformat(data["date"].replace("Z", "+00:00"))

    result = await db["expenses"].insert_one(doc)
    LOGGER.info("expense_created", expense_id=str(result.inserted_id), store_id=store_id)
    return str(result.inserted_id)


async def update_expense(store_id: str, expense_id: str, data: dict) -> bool:
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        raise ValueError(f"Store {store_id} not found.")

    try:
        exp_oid = ObjectId(expense_id)
    except InvalidId:
        raise ValueError("Invalid expense ID.")

    update_fields = {}
    if "title" in data: update_fields["title"] = data["title"]
    if "category" in data: update_fields["category"] = data["category"]
    if "amount" in data: update_fields["amount"] = float(data["amount"])
    if "payment_method" in data: update_fields["paymentMethod"] = data["payment_method"]
    if "description" in data: update_fields["description"] = data["description"]
    if "date" in data:
        update_fields["date"] = datetime.fromisoformat(data["date"].replace("Z", "+00:00"))

    if not update_fields:
        return True

    update_fields["updatedAt"] = datetime.now(timezone.utc)
    
    result = await db["expenses"].update_one(
        {"_id": exp_oid, "store": store_oid},
        {"$set": update_fields}
    )
    LOGGER.info("expense_updated", expense_id=expense_id, modified=result.modified_count)
    return result.modified_count > 0


async def delete_expense(store_id: str, expense_id: str) -> bool:
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        raise ValueError(f"Store {store_id} not found.")
        
    try:
        exp_oid = ObjectId(expense_id)
    except InvalidId:
        raise ValueError("Invalid expense ID.")
        
    result = await db["expenses"].delete_one({"_id": exp_oid, "store": store_oid})
    LOGGER.info("expense_deleted", expense_id=expense_id, deleted=result.deleted_count)
    return result.deleted_count > 0
