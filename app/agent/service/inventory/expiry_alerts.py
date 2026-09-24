"""
app/agent/service/inventory/expiry_alerts.py
=============================================
Fetch products whose expiry date is within a specified number of days,
grouped by urgency level.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.inventory.expiry_alerts")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    LOGGER.warning("expiry_alerts_store_not_found", store_id=store_id)
    return None


async def fetch_expiry_alerts(store_id: str, alert_days: int = 30) -> dict:
    """
    Returns products expiring within `alert_days` days, split into
    urgency buckets:
        - expired     : expDate < today
        - critical    : expDate in [today, today + 7d)
        - warning     : expDate in [today + 7d, today + alert_days)

    Each item: { product_id, name, category, quantity, exp_date, days_left }
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return {"expired": [], "critical": [], "warning": [], "total_at_risk": 0}

    now = datetime.utcnow()
    cutoff = now + timedelta(days=alert_days)

    cursor = db["products"].find(
        {
            "store": store_oid,
            "isActive": True,
            "quantity": {"$gt": 0},
            "expDate": {"$ne": None, "$lte": cutoff},
        },
        {"_id": 1, "name": 1, "category": 1, "quantity": 1, "expDate": 1},
    ).sort("expDate", 1)

    products = await cursor.to_list(length=500)
    LOGGER.debug("fetch_expiry_alerts", store_id=store_id, found=len(products))

    expired, critical, warning = [], [], []

    for p in products:
        exp = p.get("expDate")
        if not exp:
            continue
        days_left = (exp - now).days
        item = {
            "product_id": str(p["_id"]),
            "name": p.get("name", "Unknown"),
            "category": p.get("category", "General"),
            "quantity": p.get("quantity", 0),
            "exp_date": exp.strftime("%Y-%m-%d"),
            "days_left": days_left,
        }
        if days_left < 0:
            expired.append(item)
        elif days_left <= 7:
            critical.append(item)
        else:
            warning.append(item)

    return {
        "expired": expired,
        "critical": critical,
        "warning": warning,
        "total_at_risk": len(expired) + len(critical) + len(warning),
        "alert_days": alert_days,
    }
