"""
app/agent/service/sales/fast_moving.py
=======================================
Discovery service: fast-moving products ranked by sales velocity.

"Fast-moving" = products sold most frequently in the last N days,
ranked by average daily sales descending.

This is a pure MongoDB aggregation — no full product scan in Python.

MongoDB pipeline:
  → filter sales within lookback window
  → unwind items
  → group by productId + sum quantity
  → sort by total_qty desc
  → limit to max_results
  → lookup product name/category/current_stock from products
"""
from __future__ import annotations

from datetime import datetime, timedelta
from bson import ObjectId
from bson.errors import InvalidId
import structlog

from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.sales.fast_moving")

DEFAULT_LOOKBACK_DAYS = 30


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    LOGGER.warning("fast_moving_store_not_found", store_id=store_id)
    return None


async def fetch_fast_moving_products(
    store_id: str,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    min_daily_sales: float = 0.1,
    limit: int = 30,
) -> list[dict]:
    """
    Returns top fast-moving products by sales velocity.

    Discovery Logic (all in MongoDB aggregation):
        1. Filter sales within lookback_days
        2. Unwind items, group by productId, sum quantity
        3. Sort by total_qty_sold descending
        4. Limit to max candidates before lookup
        5. Lookup product details (name, category, current_quantity)
        6. Filter by min_daily_sales (business rule)
        7. Return top ``limit`` products

    Store isolation: query is always scoped to store_id.

    Args:
        store_id:        Authenticated store identifier.
        lookback_days:   How many days of sales history to consider (default 30).
        min_daily_sales: Minimum avg daily sales to be considered fast-moving (default 0.1).
        limit:           Maximum products to return (default 30, max 100).

    Returns:
        List of compact candidate dicts:
            {product_id, name, category, current_quantity,
             avg_daily_sales, total_qty_sold, sales_velocity_rank}
    """
    lookback_days = max(1, min(lookback_days, 365))
    limit = max(1, min(limit, 100))
    min_daily_sales = max(0.0, min_daily_sales)

    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return []

    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    LOGGER.info(
        "discovery_started",
        service="fast_moving",
        store_id=store_id,
        lookback_days=lookback_days,
        limit=limit,
    )

    # Single aggregation pipeline — all filtering/sorting at MongoDB level
    pipeline = [
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
        # Sort by sales volume descending before lookup (cheaper)
        {"$sort": {"total_qty_sold": -1}},
        # Fetch slightly more than limit so min_daily_sales filter still gives limit results
        {"$limit": limit * 3},
        # Lookup product details
        {
            "$lookup": {
                "from": "products",
                "let": {"pid": {"$toObjectId": "$_id"}},
                "pipeline": [
                    {"$match": {"$expr": {"$eq": ["$_id", "$$pid"]}}},
                    {"$project": {"name": 1, "category": 1, "quantity": 1, "price": 1}},
                ],
                "as": "product_info",
            }
        },
        {"$unwind": {"path": "$product_info", "preserveNullAndEmpty": False}},
    ]

    cursor = db["sales"].aggregate(pipeline)
    results = await cursor.to_list(length=limit * 3)

    candidates_found = len(results)
    LOGGER.info(
        "discovery_completed",
        service="fast_moving",
        store_id=store_id,
        candidates_found=candidates_found,
    )

    candidates: list[dict] = []
    for rank, r in enumerate(results, start=1):
        pid = str(r["_id"])
        total_qty = r.get("total_qty_sold", 0)
        avg_daily = round(total_qty / lookback_days, 2)

        # Business filter: skip products below min_daily_sales threshold
        if avg_daily < min_daily_sales:
            continue

        prod = r.get("product_info", {})
        candidates.append(
            {
                "product_id": pid,
                "name": prod.get("name", "Unknown"),
                "category": prod.get("category", "General"),
                "current_quantity": prod.get("quantity", 0),
                "avg_daily_sales": avg_daily,
                "total_qty_sold": total_qty,
                "price": prod.get("price", 0.0),
                "sales_velocity_rank": rank,
            }
        )

        if len(candidates) >= limit:
            break

    LOGGER.info(
        "candidates_ranked",
        service="fast_moving",
        store_id=store_id,
        candidates_found=candidates_found,
        final_candidates=len(candidates),
    )

    return candidates
