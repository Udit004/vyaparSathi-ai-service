"""
app/agent/service/expenses/summary.py
=======================================
Aggregate expense data from the expenses collection.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.expenses.summary")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    return None


async def fetch_expense_summary(store_id: str, days_lookback: int = 30) -> dict:
    """
    Returns total expenses, expense count and a per-category breakdown.
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return {}

    since = datetime.now(timezone.utc) - timedelta(days=days_lookback)

    pipeline = [
        {"$match": {"store": store_oid, "date": {"$gte": since}}},
        {
            "$group": {
                "_id": "$category",
                "total_amount": {"$sum": "$amount"},
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"total_amount": -1}},
    ]

    cursor = db["expenses"].aggregate(pipeline)
    results = await cursor.to_list(length=100)
    LOGGER.debug("fetch_expense_summary", store_id=store_id, days=days_lookback)

    categories = []
    grand_total = 0.0
    for r in results:
        amt = round(r.get("total_amount", 0.0), 2)
        grand_total += amt
        categories.append({
            "category": r.get("_id", "Other"),
            "total_amount": amt,
            "count": r.get("count", 0),
        })

    return {
        "total_expenses": round(grand_total, 2),
        "expense_count": sum(c["count"] for c in categories),
        "categories": categories,
        "days_covered": days_lookback,
    }
