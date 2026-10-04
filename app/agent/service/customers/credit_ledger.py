"""
Customer Credit Ledger (Udhar / Khata) Service.
Tracks pending customer balances, payment due timelines, credit risk levels, and reminder triggers.
Collection: customer_credit_ledger
"""
from __future__ import annotations

from datetime import datetime, timezone
import structlog
from bson import ObjectId
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.customers.credit_ledger")


async def get_customer_credit_ledger(
    user_id: str,
    store_id: str | None = None,
    min_due_days: int = 0,
    limit: int = 15
) -> list[dict]:
    """
    Fetch customer credit (Khata) ledger, pending amounts, due days, and payment risk levels.
    """
    if not user_id:
        return []

    db = get_database()
    store_oid = None

    if store_id:
        try:
            store_oid = ObjectId(store_id)
        except Exception:
            pass

    # Query buyers (customers) with pending balances
    query: dict = {
        "$or": [
            {"store": store_oid} if store_oid else {},
            {"user_id": str(user_id)}
        ]
    }
    # Clean up empty $or conditions
    query["$or"] = [cond for cond in query["$or"] if cond]
    if not query["$or"]:
        query = {}

    # Query customer credit ledger documents
    cursor = db["customer_credit_ledger"].find({"user_id": str(user_id)}).sort("due_days", -1).limit(limit)
    records = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        if doc.get("due_days", 0) >= min_due_days:
            records.append(doc)

    if not records:
        # Fallback: scan buyers collection for pending credit
        buyer_cursor = db["buyers"].find({"pending_amount": {"$gt": 0}}).limit(limit)
        async for buyer in buyer_cursor:
            record = {
                "_id": str(buyer["_id"]),
                "user_id": str(user_id),
                "customer_name": buyer.get("name") or buyer.get("fullName") or "Unknown Customer",
                "phone": buyer.get("phone") or buyer.get("phoneNumber"),
                "outstanding_balance": float(buyer.get("pending_amount", buyer.get("balance", 0.0))),
                "due_days": 15,
                "risk_level": "MEDIUM" if buyer.get("pending_amount", 0) > 1000 else "LOW",
                "last_payment_date": buyer.get("updatedAt", datetime.now(timezone.utc))
            }
            records.append(record)

    return records


async def record_customer_credit_transaction(
    user_id: str,
    store_id: str | None,
    customer_name: str,
    amount: float,
    transaction_type: str = "CREDIT",  # "CREDIT" (udhar) or "PAYMENT" (jama)
) -> dict:
    """
    Update or create customer credit record.
    """
    if not user_id or not customer_name or amount <= 0:
        return {"error": "Invalid credit transaction parameters."}

    db = get_database()
    now = datetime.now(timezone.utc)

    existing = await db["customer_credit_ledger"].find_one({
        "user_id": str(user_id),
        "customer_name": {"$regex": f"^{customer_name.strip()}$", "$options": "i"}
    })

    current_balance = float(existing.get("outstanding_balance", 0.0)) if existing else 0.0

    if transaction_type.upper() == "CREDIT":
        new_balance = current_balance + float(amount)
    else:
        new_balance = max(0.0, current_balance - float(amount))

    risk = "HIGH" if new_balance > 3000 else ("MEDIUM" if new_balance > 1000 else "LOW")

    doc = {
        "user_id": str(user_id),
        "store_id": str(store_id) if store_id else None,
        "customer_name": customer_name.strip(),
        "outstanding_balance": round(new_balance, 2),
        "due_days": existing.get("due_days", 1) if existing else 1,
        "risk_level": risk,
        "last_transaction_type": transaction_type.upper(),
        "last_transaction_amount": float(amount),
        "updated_at": now
    }

    await db["customer_credit_ledger"].update_one(
        {"user_id": str(user_id), "customer_name": customer_name.strip()},
        {"$set": doc},
        upsert=True
    )

    LOGGER.info("customer_credit_updated", user_id=user_id, customer=customer_name, new_balance=new_balance)
    return doc
