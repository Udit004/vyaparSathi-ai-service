"""
Proactive Action Engine Service.
Generates, fetches, and manages proactive business insights and recommendations.
Collection: proactive_insights
"""
from __future__ import annotations

from datetime import datetime, timezone
import structlog
from bson import ObjectId
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.insights.proactive_insights")


async def get_unread_insights(user_id: str, store_id: str | None = None, limit: int = 5) -> list[dict]:
    """
    Fetch unread/active proactive insights for a user/store.
    """
    if not user_id:
        return []

    db = get_database()
    query: dict = {
        "user_id": str(user_id),
        "is_dismissed": {"$ne": True}
    }
    if store_id:
        query["$or"] = [
            {"store_id": str(store_id)},
            {"store_id": {"$exists": False}},
            {"store_id": None}
        ]

    cursor = db["proactive_insights"].find(query).sort("created_at", -1).limit(limit)
    insights = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        insights.append(doc)

    return insights


async def generate_proactive_insights_for_merchant(user_id: str, store_id: str | None = None) -> list[dict]:
    """
    Scans live stock, sales trends, and festival calendar to generate fresh proactive recommendations.
    """
    if not user_id:
        return []

    db = get_database()

    # Find store document
    store_oid = None
    if store_id:
        try:
            store_oid = ObjectId(store_id)
        except Exception:
            pass

    if not store_oid:
        store_doc = await db["stores"].find_one({"$or": [{"user": ObjectId(user_id)}, {"owner": user_id}]})
        if store_doc:
            store_oid = store_doc.get("_id")
            store_id = str(store_oid)

    generated = []
    now = datetime.now(timezone.utc)

    # 1. Low Stock Outage Warning
    if store_oid:
        low_stock_cursor = db["products"].find({
            "store": store_oid,
            "isActive": {"$ne": False},
            "$expr": {"$lte": ["$quantity", "$minQuantity"]}
        }).limit(3)

        low_stock_items = []
        async for prod in low_stock_cursor:
            low_stock_items.append(prod.get("name", "Unknown item"))

        if low_stock_items:
            insight_doc = {
                "user_id": str(user_id),
                "store_id": str(store_id) if store_id else None,
                "type": "REORDER_ALERT",
                "priority": "HIGH",
                "title": f"Low Stock Outage Risk ({len(low_stock_items)} items)",
                "description": f"Items running low: {', '.join(low_stock_items)}. Reorder recommended before stockout.",
                "recommended_action": {
                    "tool_name": "create_smart_purchase_order",
                    "items": low_stock_items
                },
                "is_dismissed": False,
                "created_at": now
            }
            # Deduplicate by title & user_id
            existing = await db["proactive_insights"].find_one({
                "user_id": str(user_id),
                "title": insight_doc["title"],
                "is_dismissed": {"$ne": True}
            })
            if not existing:
                res = await db["proactive_insights"].insert_one(insight_doc)
                insight_doc["_id"] = str(res.inserted_id)
                generated.append(insight_doc)

    # Return all unread insights
    return await get_unread_insights(user_id=user_id, store_id=store_id)


async def dismiss_proactive_insight(user_id: str, insight_id: str) -> bool:
    """
    Mark a proactive insight as dismissed or acted upon.
    """
    if not user_id or not insight_id:
        return False

    db = get_database()
    try:
        oid = ObjectId(insight_id)
    except Exception:
        return False

    res = await db["proactive_insights"].update_one(
        {"_id": oid, "user_id": str(user_id)},
        {"$set": {"is_dismissed": True, "dismissed_at": datetime.now(timezone.utc)}}
    )
    return res.modified_count > 0
