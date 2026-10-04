"""
Store Behavioral Baselines Service.
Computes and retrieves 30-day historical benchmarks, peak hours, dead-stock counts, and daily revenue averages.
Collection: store_baselines
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import structlog
from bson import ObjectId
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.analytics.store_baselines")


async def get_store_baselines(user_id: str, store_id: str | None = None) -> dict:
    """
    Retrieve or compute 30-day sales and inventory behavioral baselines for a store/user.
    """
    if not user_id and not store_id:
        return {"error": "user_id or store_id is required."}

    db = get_database()
    store_oid = None

    if store_id:
        try:
            store_oid = ObjectId(store_id)
        except Exception:
            pass

    if not store_oid and user_id:
        store_doc = await db["stores"].find_one({"$or": [{"user": ObjectId(user_id)}, {"owner": user_id}]})
        if store_doc:
            store_oid = store_doc.get("_id")
            store_id = str(store_oid)

    if not store_oid:
        return {"error": "Store context not found."}

    # Check cached baselines from DB
    existing = await db["store_baselines"].find_one({"store_id": str(store_oid)})
    if existing and (datetime.now(timezone.utc) - existing.get("calculated_at", datetime.min.replace(tzinfo=timezone.utc))).days < 1:
        existing["_id"] = str(existing["_id"])
        return existing

    # Compute live baselines for the last 30 days
    now = datetime.now(timezone.utc)
    thirty_days_ago = now - timedelta(days=30)

    # 1. Sales Aggregation (Revenue, Order Count, Avg Order Value)
    pipeline = [
        {"$match": {"store": store_oid, "createdAt": {"$gte": thirty_days_ago}}},
        {"$group": {
            "_id": None,
            "total_revenue": {"$sum": "$totalAmount"},
            "order_count": {"$sum": 1},
            "avg_ticket_size": {"$avg": "$totalAmount"}
        }}
    ]
    sales_res = await db["sales"].aggregate(pipeline).to_list(length=1)
    sales_stats = sales_res[0] if sales_res else {}

    total_revenue = float(sales_stats.get("total_revenue", 0.0))
    order_count = int(sales_stats.get("order_count", 0))
    avg_ticket_size = float(sales_stats.get("avg_ticket_size", 0.0))
    avg_daily_revenue = total_revenue / 30.0

    # 2. Peak Hours
    peak_pipeline = [
        {"$match": {"store": store_oid, "createdAt": {"$gte": thirty_days_ago}}},
        {"$project": {"hour": {"$hour": "$createdAt"}}},
        {"$group": {"_id": "$hour", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 3}
    ]
    peak_res = await db["sales"].aggregate(peak_pipeline).to_list(length=3)
    peak_hours = [item["_id"] for item in peak_res if "_id" in item]

    # 3. Dead Stock Count (>45 days without sale or update)
    forty_five_days_ago = now - timedelta(days=45)
    dead_stock_count = await db["products"].count_documents({
        "store": store_oid,
        "isActive": {"$ne": False},
        "updatedAt": {"$lt": forty_five_days_ago}
    })

    baseline_doc = {
        "store_id": str(store_oid),
        "user_id": str(user_id) if user_id else None,
        "avg_daily_revenue": round(avg_daily_revenue, 2),
        "total_30d_revenue": round(total_revenue, 2),
        "total_30d_orders": order_count,
        "avg_ticket_size": round(avg_ticket_size, 2),
        "peak_sales_hours": peak_hours,
        "slow_moving_products_count": dead_stock_count,
        "calculated_at": now
    }

    await db["store_baselines"].update_one(
        {"store_id": str(store_oid)},
        {"$set": baseline_doc},
        upsert=True
    )

    doc = await db["store_baselines"].find_one({"store_id": str(store_oid)})
    if doc:
        doc["_id"] = str(doc["_id"])
        return doc
    return baseline_doc
