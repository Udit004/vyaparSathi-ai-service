"""
app/agent/service/sales/profit_margin.py
=========================================
Per-category price margin analysis.

The schema now has:
  - products.buyingPrice  → cost / purchase price per unit
  - products.sellingPrice → current selling price
  - inventory.sellingPrice → store-specific selling price (if overridden)

We compute:
  gross_margin_pct = (sellingPrice - buyingPrice) / sellingPrice * 100

If inventory.sellingPrice is set, we use that; otherwise we fall back to
products.sellingPrice.
"""
from __future__ import annotations

from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.sales.profit_margin")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    LOGGER.warning("profit_margin_store_not_found", store_id=store_id)
    return None


async def fetch_profit_margin_analysis(store_id: str) -> list[dict]:
    """
    Returns per-category estimated margin analysis, sorted by margin asc
    (lowest margin categories first — most actionable).

    Each item:
        category, product_count,
        avg_cost_price, avg_selling_price,
        avg_margin_pct, total_potential_profit
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return []

    # Join products with inventory to get effective selling price
    pipeline = [
        {"$match": {"store": store_oid, "isActive": True}},
        {
            "$lookup": {
                "from": "inventories",
                "localField": "_id",
                "foreignField": "product",
                "as": "inv",
            }
        },
        {
            "$addFields": {
                "effective_selling_price": {
                    "$ifNull": [
                        {"$arrayElemAt": ["$inv.sellingPrice", 0]},
                        "$sellingPrice",
                    ]
                }
            }
        },
        {
            "$group": {
                "_id": "$category",
                "product_count": {"$sum": 1},
                "avg_cost_price": {"$avg": "$buyingPrice"},
                "avg_selling_price": {"$avg": "$effective_selling_price"},
                "total_qty": {"$sum": "$quantity"},
            }
        },
        {"$sort": {"product_count": -1}},
    ]

    cursor = db["products"].aggregate(pipeline)
    results = await cursor.to_list(length=100)
    LOGGER.debug("fetch_profit_margin_analysis", store_id=store_id, categories=len(results))

    output = []
    for r in results:
        cost = r.get("avg_cost_price") or 0.0
        sell = r.get("avg_selling_price") or 0.0
        if sell > 0 and sell >= cost:
            margin_pct = round(((sell - cost) / sell) * 100, 2)
        else:
            margin_pct = 0.0

        total_qty = r.get("total_qty", 0)
        potential_profit = round((sell - cost) * total_qty, 2) if sell > cost else 0.0

        output.append(
            {
                "category": r.get("_id", "Uncategorized"),
                "product_count": r.get("product_count", 0),
                "avg_cost_price": round(cost, 2),
                "avg_selling_price": round(sell, 2),
                "avg_margin_pct": margin_pct,
                "total_potential_profit": potential_profit,
            }
        )

    output.sort(key=lambda x: x["avg_margin_pct"])
    return output
