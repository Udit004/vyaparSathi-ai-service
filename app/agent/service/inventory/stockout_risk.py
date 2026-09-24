"""
app/agent/service/inventory/stockout_risk.py
=============================================
Discovery service: products at risk of stocking out within N days.

This is a pure database-side discovery operation:
  MongoDB
    → filter active products with quantity > 0
    → join with 30-day sales aggregation (avg daily sales)
    → compute days_to_stockout = current_qty / avg_daily_sales
    → filter where days_to_stockout <= days_threshold
    → sort by days_to_stockout asc (most urgent first)
    → limit to max_results

Formula:
    avg_daily_sales = total_qty_sold_last_30d / 30
    days_to_stockout = current_qty / avg_daily_sales

Products with no sales history (avg_daily_sales = 0) are excluded — they
carry no stockout risk by definition.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from bson import ObjectId
from bson.errors import InvalidId
import structlog

from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.inventory.stockout_risk")

SALES_LOOKBACK_DAYS = 30
MAX_PRODUCTS_SCAN = 5000


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    LOGGER.warning("stockout_risk_store_not_found", store_id=store_id)
    return None


async def fetch_stockout_risk_products(
    store_id: str,
    days: int = 7,
    limit: int = 50,
) -> list[dict]:
    """
    Returns products likely to run out within ``days`` days.

    Discovery Logic:
        1. Aggregate avg daily sales from the last 30 days (MongoDB)
        2. Fetch all active products with stock > 0 (MongoDB)
        3. Compute days_to_stockout = current_qty / avg_daily_sales
        4. Filter where days_to_stockout <= days (Python business filter)
        5. Sort ascending by days_to_stockout (most urgent first)
        6. Return top ``limit`` candidates

    Store isolation: query is always scoped to store_id.

    Args:
        store_id: Authenticated store identifier.
        days:     Stockout risk horizon in days (default 7).
        limit:    Maximum products to return (default 50, max 200).

    Returns:
        List of compact candidate dicts:
            {product_id, name, category, current_quantity,
             avg_daily_sales, days_to_stockout, price}
    """
    # Guard: prevent absurd limits
    days = max(1, min(days, 90))
    limit = max(1, min(limit, 200))

    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return []

    cutoff = datetime.utcnow() - timedelta(days=SALES_LOOKBACK_DAYS)

    LOGGER.info(
        "discovery_started",
        service="stockout_risk",
        store_id=store_id,
        days=days,
        limit=limit,
    )

    # Stage 1: avg daily sales per product (MongoDB aggregation)
    sales_pipeline = [
        {
            "$match": {
                "store": store_oid,
                "completedAt": {"$gte": cutoff},
            }
        },
        {"$unwind": "$items"},
        {
            "$group": {
                "_id": "$items.productId",
                "total_qty_sold": {"$sum": "$items.quantity"},
            }
        },
    ]
    sales_cursor = db["sales"].aggregate(sales_pipeline)
    sales_list = await sales_cursor.to_list(length=MAX_PRODUCTS_SCAN)
    sales_map: dict[str, float] = {
        str(s["_id"]): s["total_qty_sold"] / SALES_LOOKBACK_DAYS
        for s in sales_list
    }

    # Stage 2: active products with stock > 0 (MongoDB)
    prod_cursor = db["products"].find(
        {"store": store_oid, "isActive": True, "quantity": {"$gt": 0}},
        {"_id": 1, "name": 1, "category": 1, "quantity": 1, "price": 1},
    )
    products = await prod_cursor.to_list(length=MAX_PRODUCTS_SCAN)

    products_scanned = len(products)
    LOGGER.info(
        "discovery_completed",
        service="stockout_risk",
        store_id=store_id,
        products_scanned=products_scanned,
        products_with_sales=len(sales_map),
    )

    # Stage 3–4: compute days_to_stockout + filter (Python business logic)
    candidates: list[dict] = []
    for p in products:
        pid = str(p["_id"])
        avg_daily = sales_map.get(pid, 0.0)
        if avg_daily <= 0:
            continue  # No sales → no stockout risk

        current_qty = p.get("quantity", 0)
        days_to_stockout = round(current_qty / avg_daily, 1)

        if days_to_stockout <= days:
            candidates.append(
                {
                    "product_id": pid,
                    "name": p.get("name", "Unknown"),
                    "category": p.get("category", "General"),
                    "current_quantity": current_qty,
                    "avg_daily_sales": round(avg_daily, 2),
                    "days_to_stockout": days_to_stockout,
                    "price": p.get("price", 0.0),
                }
            )

    candidates_found = len(candidates)

    # Stage 5: sort by urgency (lowest days_to_stockout first)
    candidates.sort(key=lambda x: x["days_to_stockout"])

    # Stage 6: limit
    final_candidates = candidates[:limit]

    LOGGER.info(
        "candidates_ranked",
        service="stockout_risk",
        store_id=store_id,
        candidates_found=candidates_found,
        after_limit=len(final_candidates),
        discovery_metrics={
            "products_scanned": products_scanned,
            "candidates_found": candidates_found,
            "final_candidates": len(final_candidates),
        },
    )

    return final_candidates
