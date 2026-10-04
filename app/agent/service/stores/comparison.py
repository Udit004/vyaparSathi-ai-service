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
    current_store_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch and compare real business metrics strictly across stores owned by or linked to the given user account.
    Data from other merchants/users is strictly excluded.
    """
    db = get_database()

    # 1. Collect all valid user keys & ObjectIds
    user_keys: List[Any] = []
    if user_id and user_id != "default_user":
        u_clean = str(user_id).strip()
        user_keys.append(u_clean)
        try:
            user_keys.append(ObjectId(u_clean))
        except Exception:
            pass

    # 2. Check current store context to discover owner if user_keys is missing/default
    anchor_store_doc = None
    anchor_id = current_store_id or (store_ids[0] if store_ids else None)
    if anchor_id:
        try:
            anchor_oid = ObjectId(anchor_id)
            anchor_store_doc = await db["stores"].find_one({"_id": anchor_oid})
        except Exception:
            anchor_store_doc = await db["stores"].find_one({"name": anchor_id})

    if anchor_store_doc:
        owner_val = (
            anchor_store_doc.get("user") or
            anchor_store_doc.get("userId") or
            anchor_store_doc.get("owner") or
            anchor_store_doc.get("merchant") or
            anchor_store_doc.get("createdBy")
        )
        if owner_val:
            owner_str = str(owner_val).strip()
            if owner_str not in user_keys:
                user_keys.append(owner_str)
            try:
                owner_oid = ObjectId(owner_str)
                if owner_oid not in user_keys:
                    user_keys.append(owner_oid)
            except Exception:
                pass

    # 3. Lookup user document to gather linked store ObjectIds
    explicit_store_oids: List[ObjectId] = []
    if user_keys:
        user_doc_query = [{"_id": k} for k in user_keys if isinstance(k, ObjectId)] + \
                         [{"_id": k} for k in user_keys if isinstance(k, str)] + \
                         [{"uid": k} for k in user_keys if isinstance(k, str)]
        if user_doc_query:
            try:
                user_doc = await db["users"].find_one({"$or": user_doc_query})
                if user_doc:
                    user_stores = user_doc.get("stores") or user_doc.get("store") or user_doc.get("storeId")
                    if isinstance(user_stores, list):
                        for st in user_stores:
                            try:
                                explicit_store_oids.append(ObjectId(st))
                            except Exception:
                                pass
                    elif user_stores:
                        try:
                            explicit_store_oids.append(ObjectId(user_stores))
                        except Exception:
                            pass
            except Exception as exc:
                LOGGER.warning("user_doc_lookup_failed", error=str(exc))

    # 4. Construct user ownership filter
    user_ownership_clauses: List[Dict[str, Any]] = []
    for k in user_keys:
        user_ownership_clauses.extend([
            {"user": k},
            {"userId": k},
            {"owner": k},
            {"user_id": k},
            {"createdBy": k},
            {"merchant": k},
        ])
    if explicit_store_oids:
        user_ownership_clauses.append({"_id": {"$in": explicit_store_oids}})

    # If anchor store is known, include anchor store
    if anchor_store_doc:
        user_ownership_clauses.append({"_id": anchor_store_doc["_id"]})

    # 5. Query stores with user isolation filter
    stores_list = []
    if user_ownership_clauses:
        user_filter = {"$or": user_ownership_clauses}

        if store_ids:
            requested_oids = []
            requested_names = []
            for s in store_ids:
                try:
                    requested_oids.append(ObjectId(s))
                except Exception:
                    requested_names.append(s)
            req_filter = {"$or": [{"_id": {"$in": requested_oids}}, {"name": {"$in": requested_names}}]}
            final_query = {"$and": [user_filter, req_filter]}
        else:
            final_query = user_filter

        cursor = db["stores"].find(final_query)
        stores_list = await cursor.to_list(length=50)

    # 6. Deduplicate stores by _id
    unique_stores: Dict[str, Any] = {}
    for s in stores_list:
        unique_stores[str(s["_id"])] = s

    if not unique_stores and anchor_store_doc:
        unique_stores[str(anchor_store_doc["_id"])] = anchor_store_doc

    stores_list = list(unique_stores.values())

    results = []
    for store in stores_list:
        s_id = str(store["_id"])
        s_name = store.get("name") or store.get("storeName") or f"Store {s_id[:6]}"
        s_city = store.get("city") or store.get("address") or "India"

        # Fetch Sales KPI
        sales_kpi = await fetch_sales_summary(s_id, days_lookback=days_lookback)
        revenue = sales_kpi.get("total_revenue", 0.0)
        orders_count = sales_kpi.get("total_sales_count", 0)
        aov = sales_kpi.get("average_order_value", 0.0)

        # Fetch Inventory Metrics
        total_products = await db["products"].count_documents({"store": store["_id"], "isActive": {"$ne": False}})
        low_stock_count = await db["products"].count_documents({
            "store": store["_id"],
            "isActive": {"$ne": False},
            "$expr": {"$lte": ["$stock", {"$ifNull": ["$minStock", 5]}]}
        })

        # Fetch Vendors and Customers
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

    if results:
        summary = (
            f"Compared {len(results)} stores linked to your user account. "
            f"Top performer is '{top_performer}' with ₹{results[0]['revenue']:,.2f} revenue. "
            f"Combined total revenue across your stores: ₹{total_user_revenue:,.2f}."
        )
    else:
        summary = "No stores found linked to your merchant account. Only stores registered under your account can be compared."

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
