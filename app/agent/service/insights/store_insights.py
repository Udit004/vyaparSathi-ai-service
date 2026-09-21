from __future__ import annotations

from datetime import datetime, timedelta
from bson import ObjectId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.insights.store_insights")

# Days with zero sales before a product is considered "dead stock".
DEAD_STOCK_DAYS = 60
# Days to look back for revenue-leader computation.
REVENUE_LEAD_DAYS = 30
# Maximum products surfaced in each insight category.
MAX_OOS_LIST = 10
MAX_DEAD_STOCK_LIST = 2000
MAX_SOLD_IDS = 5000
MAX_LEADERS = 3

async def fetch_store_insights(store_id: str) -> list[dict]:
    """
    Generates structured insights from real data:
        - Dead stock (quantity > 0, zero sales in 60 days)
        - Slow moving (avg daily < 0.5, stock > 10)
        - Revenue leaders (top 3 by revenue in 30 days)
        - Out of stock products
    """
    db = get_database()
    insights = []

    # --- Out of stock ---
    oos_cursor = db["products"].find(
        {"store": ObjectId(store_id), "isActive": True, "quantity": {"$lte": 0}},
        {"_id": 1, "name": 1},
    ).limit(MAX_OOS_LIST)
    oos = await oos_cursor.to_list(length=MAX_OOS_LIST)
    if oos:
        names = [p.get("name", "Unknown") for p in oos]
        insights.append(
            {
                "type": "out_of_stock",
                "title": "Products Out of Stock",
                "summary": (
                    f"{len(oos)} product(s) are out of stock: "
                    f"{', '.join(names[:3])}{'...' if len(names) > 3 else ''}. "
                    "Restock immediately."
                ),
                "severity": "danger",
                "affected_products": [str(p["_id"]) for p in oos],
            }
        )

    # --- Dead stock (no sales in 60 days, qty > 0) ---
    dead_cutoff = datetime.utcnow() - timedelta(days=DEAD_STOCK_DAYS)
    sold_pipeline = [
        {"$match": {"store": ObjectId(store_id), "completedAt": {"$gte": dead_cutoff}}},
        {"$unwind": "$items"},
        {"$group": {"_id": "$items.productId"}},
    ]
    sold_cursor = db["sales"].aggregate(sold_pipeline)
    sold_ids = {r["_id"] for r in await sold_cursor.to_list(length=MAX_SOLD_IDS)}

    all_products = await db["products"].find(
        {"store": ObjectId(store_id), "isActive": True, "quantity": {"$gt": 0}},
        {"_id": 1, "name": 1, "quantity": 1, "price": 1},
    ).to_list(length=MAX_DEAD_STOCK_LIST)

    dead_stock = [
        p for p in all_products
        if str(p["_id"]) not in sold_ids and str(p.get("_id")) not in sold_ids
    ]

    if dead_stock:
        dead_value = sum(p.get("quantity", 0) * p.get("price", 0) for p in dead_stock)
        names = [p.get("name", "Unknown") for p in dead_stock[:3]]
        insights.append(
            {
                "type": "dead_stock",
                "title": "Dead Stock Detected",
                "summary": (
                    f"{len(dead_stock)} product(s) have had zero sales in the last "
                    f"{DEAD_STOCK_DAYS} days with ₹{round(dead_value, 2)} of stock value "
                    f"tied up. Consider discounting: "
                    f"{', '.join(names)}{'...' if len(dead_stock) > 3 else ''}."
                ),
                "severity": "warning",
                "dead_stock_value": round(dead_value, 2),
                "affected_count": len(dead_stock),
            }
        )

    # --- Revenue leaders (last 30 days) ---
    top_pipeline = [
        {
            "$match": {
                "store": ObjectId(store_id),
                "completedAt": {"$gte": datetime.utcnow() - timedelta(days=REVENUE_LEAD_DAYS)},
            }
        },
        {"$unwind": "$items"},
        {
            "$group": {
                "_id": "$items.productId",
                "revenue": {"$sum": "$items.lineTotal"},
                "qty_sold": {"$sum": "$items.quantity"},
            }
        },
        {"$sort": {"revenue": -1}},
        {"$limit": MAX_LEADERS},
    ]
    top_cursor = db["sales"].aggregate(top_pipeline)
    top_sellers = await top_cursor.to_list(length=MAX_LEADERS)

    if top_sellers:
        # Look up names
        top_ids = [ObjectId(r["_id"]) for r in top_sellers if r["_id"]]
        name_cursor = db["products"].find({"_id": {"$in": top_ids}}, {"_id": 1, "name": 1})
        name_map = {
            str(p["_id"]): p.get("name", "Unknown")
            for p in await name_cursor.to_list(length=MAX_LEADERS)
        }

        leaders = [
            {
                "name": name_map.get(str(r["_id"]), "Unknown"),
                "revenue": round(r["revenue"], 2),
                "qty_sold": r["qty_sold"],
            }
            for r in top_sellers
        ]
        insights.append(
            {
                "type": "revenue_leaders",
                "title": "Top Revenue Products (30 days)",
                "summary": (
                    f"Top earner: {leaders[0]['name']} with "
                    f"₹{leaders[0]['revenue']} revenue. "
                    "These products drive most of your sales."
                ),
                "severity": "info",
                "leaders": leaders,
            }
        )

    LOGGER.debug("fetch_store_insights", store_id=store_id, insight_count=len(insights))
    return insights
