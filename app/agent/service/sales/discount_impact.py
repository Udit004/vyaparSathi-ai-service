"""
app/agent/service/sales/discount_impact.py
==========================================
Aggregate discount analysis: how much revenue is being given away
as discounts, the effective discount rate, and top discounted sales.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.sales.discount_impact")

DEFAULT_LOOKBACK_DAYS = 30


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    LOGGER.warning("discount_impact_store_not_found", store_id=store_id)
    return None


async def fetch_discount_impact(
    store_id: str, days: int = DEFAULT_LOOKBACK_DAYS
) -> dict:
    """
    Returns discount analytics for the last `days` days:
        total_gross_revenue    — revenue before discounts (subtotal sum)
        total_discount_given   — total discount amount applied
        total_net_revenue      — actual revenue collected (totalAmount sum)
        discount_rate_pct      — discount_given / gross * 100
        discounted_sales_count — number of sales that had a discount
        total_sales_count      — total sales in the period
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return {
            "total_gross_revenue": 0.0,
            "total_discount_given": 0.0,
            "total_net_revenue": 0.0,
            "discount_rate_pct": 0.0,
            "discounted_sales_count": 0,
            "total_sales_count": 0,
        }

    cutoff = datetime.utcnow() - timedelta(days=days)

    pipeline = [
        {"$match": {"store": store_oid, "completedAt": {"$gte": cutoff}}},
        {
            "$group": {
                "_id": None,
                "total_gross": {"$sum": "$subtotal"},
                "total_discount": {"$sum": "$discount.amount"},
                "total_net": {"$sum": "$totalAmount"},
                "total_sales": {"$sum": 1},
                "discounted_sales": {
                    "$sum": {
                        "$cond": [{"$gt": ["$discount.amount", 0]}, 1, 0]
                    }
                },
            }
        },
    ]

    cursor = db["sales"].aggregate(pipeline)
    results = await cursor.to_list(length=1)
    LOGGER.debug("fetch_discount_impact", store_id=store_id, days=days)

    if not results:
        return {
            "total_gross_revenue": 0.0,
            "total_discount_given": 0.0,
            "total_net_revenue": 0.0,
            "discount_rate_pct": 0.0,
            "discounted_sales_count": 0,
            "total_sales_count": 0,
            "days_covered": days,
        }

    r = results[0]
    gross = r.get("total_gross", 0.0) or 0.0
    discount = r.get("total_discount", 0.0) or 0.0
    net = r.get("total_net", 0.0) or 0.0
    rate = round((discount / gross) * 100, 2) if gross > 0 else 0.0

    return {
        "total_gross_revenue": round(gross, 2),
        "total_discount_given": round(discount, 2),
        "total_net_revenue": round(net, 2),
        "discount_rate_pct": rate,
        "discounted_sales_count": r.get("discounted_sales", 0),
        "total_sales_count": r.get("total_sales", 0),
        "days_covered": days,
    }
