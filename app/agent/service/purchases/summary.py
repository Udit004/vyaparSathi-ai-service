"""
app/agent/service/purchases/summary.py
========================================
Aggregate purchase summaries from the purchases collection.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.purchases.summary")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    return None


async def fetch_purchase_summary(store_id: str, days_lookback: int = 30) -> dict:
    """
    Returns aggregated purchase data for the given period:
    total spend, order count, average order value, total due, and payment status breakdown.
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return {}

    since = datetime.now(timezone.utc) - timedelta(days=days_lookback)

    pipeline = [
        {"$match": {"store": store_oid, "purchaseDate": {"$gte": since}}},
        {
            "$group": {
                "_id": None,
                "total_spend": {"$sum": "$grandTotal"},
                "total_paid": {"$sum": "$paidAmount"},
                "total_due": {"$sum": "$dueAmount"},
                "order_count": {"$sum": 1},
                "paid_count": {"$sum": {"$cond": [{"$eq": ["$paymentStatus", "paid"]}, 1, 0]}},
                "partial_count": {"$sum": {"$cond": [{"$eq": ["$paymentStatus", "partial"]}, 1, 0]}},
                "unpaid_count": {"$sum": {"$cond": [{"$eq": ["$paymentStatus", "unpaid"]}, 1, 0]}},
            }
        },
    ]

    cursor = db["purchases"].aggregate(pipeline)
    results = await cursor.to_list(length=1)
    LOGGER.debug("fetch_purchase_summary", store_id=store_id, days=days_lookback)

    if not results:
        return {
            "total_spend": 0.0, "total_paid": 0.0, "total_due": 0.0,
            "order_count": 0, "average_order_value": 0.0,
            "paid_count": 0, "partial_count": 0, "unpaid_count": 0,
            "days_covered": days_lookback,
        }

    r = results[0]
    count = r.get("order_count", 0)
    spend = r.get("total_spend", 0.0)
    return {
        "total_spend": round(spend, 2),
        "total_paid": round(r.get("total_paid", 0.0), 2),
        "total_due": round(r.get("total_due", 0.0), 2),
        "order_count": count,
        "average_order_value": round(spend / count, 2) if count else 0.0,
        "paid_count": r.get("paid_count", 0),
        "partial_count": r.get("partial_count", 0),
        "unpaid_count": r.get("unpaid_count", 0),
        "days_covered": days_lookback,
    }
