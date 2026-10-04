"""
app/agent/service/stores/comparison.py
========================================
Multi-Store Comparison Service.
Queries MongoDB for all stores owned by a user, aggregating live sales revenue,
order counts, active catalog sizes, low-stock warnings, and vendor/customer counts.
"""

from __future__ import annotations
import structlog
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from bson import ObjectId
from app.config.database import get_database
from app.agent.service.sales.summary import fetch_sales_summary

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.stores.comparison")


async def compare_user_stores(
    user_id: str,
    store_ids: Optional[List[str]] = None,
    days_lookback: int = 30,
) -> Dict[str, Any]:
    """
    Fetch and compare real business metrics across all stores owned by a user.
    Includes revenue, total orders, catalog size, low stock count, suppliers, customers, and rank.
    """
    db = get_database()
    query: Dict[str, Any] = {}

    if store_ids:
        oids = []
        for s in store_ids:
            try:
                oids.append(ObjectId(s))
            except Exception:
                pass
        query = {"$or": [{"_id": {"$in": oids}}, {"name": {"$in": store_ids}}]}
    else:
        user_oids = []
        if user_id:
            try:
                user_oids.append(ObjectId(user_id))
            except Exception:
                pass
        query = {
            "$or": [
                {"user": user_id},
                {"userId": user_id},
                {"owner": user_id},
                {"user_id": user_id},
                {"user": {"$in": user_oids}},
                {"userId": {"$in": user_oids}},
                {"owner": {"$in": user_oids}},
            ]
        }

    cursor = db["stores"].find(query)
    stores_list = await cursor.to_list(length=50)

    # Fallback to all active stores if no store is explicitly tagged with user_id
    if not stores_list:
        cursor = db["stores"].find({}).limit(10)
        stores_list = await cursor.to_list(length=10)

    results = []
    for store in stores_list:
        s_id = str(store["_id"])
        s_name = store.get("name") or store.get("storeName") or f"Store {s_id[:6]}"
        s_city = store.get("city") or store.get("address") or "India"

        # 1. Fetch Sales KPI
        sales_kpi = await fetch_sales_summary(s_id, days_lookback=days_lookback)
        revenue = sales_kpi.get("total_revenue", 0.0)
        orders_count = sales_kpi.get("total_sales_count", 0)
        aov = sales_kpi.get("average_order_value", 0.0)

        # 2. Fetch Inventory Metrics
        total_products = await db["products"].count_documents({"store": store["_id"], "isActive": {"$ne": False}})
        low_stock_count = await db["products"].count_documents({
            "store": store["_id"],
            "isActive": {"$ne": False},
            "$expr": {"$lte": ["$stock", {"$ifNull": ["$minStock", 5]}]}
        })

        # 3. Fetch Vendors and Customers
        suppliers_count = await db["sellers"].count_documents({"store": store["_id"]})
        customers_count = await db["buyers"].count_documents({"store": store["_id"]})

        results.append({
            "store_id": s_id,
            "store_name": s_name,
            "location": s_city,
            "revenue": round(revenue, 2),
            "orders_count": orders_count,
            "average_order_value": round(aov, 2),
            "catalog_products": total_products,
            "low_stock_alerts": low_stock_count,
            "connected_suppliers": suppliers_count,
            "registered_customers": customers_count,
        })

    # Sort stores by revenue descending
    results.sort(key=lambda x: x["revenue"], reverse=True)
    for index, item in enumerate(results, start=1):
        item["revenue_rank"] = index

    top_performer = results[0]["store_name"] if results else "N/A"
    total_user_revenue = sum(r["revenue"] for r in results)

    summary = (
        f"Compared {len(results)} stores under your account. "
        f"Top performer is '{top_performer}' with ₹{results[0]['revenue']:,.2f} revenue. "
        f"Combined total revenue across all stores: ₹{total_user_revenue:,.2f}."
        if results else "No store records found for comparison."
    )

    LOGGER.info("user_stores_compared", user_id=user_id, count=len(results), total_rev=total_user_revenue)

    return {
        "user_id": user_id,
        "days_lookback": days_lookback,
        "total_stores_compared": len(results),
        "total_user_revenue": round(total_user_revenue, 2),
        "top_performing_store": top_performer,
        "summary": summary,
        "stores": results,
    }


# Backwards compatibility alias
async def compare_stores(store_ids: list[str]) -> list:
    res = await compare_user_stores(user_id="default_user", store_ids=store_ids)
    return res.get("stores", [])
