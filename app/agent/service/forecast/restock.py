from __future__ import annotations

from datetime import datetime, timedelta
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.forecast.restock")

# Days between ordering and receiving stock.
LEAD_TIME_DAYS = 3
# Extra safety-stock buffer (days of average sales).
SAFETY_STOCK_DAYS = 7
# Lookback window for computing average daily sales.
SALES_LOOKBACK_DAYS = 30
# Maximum products returned by the sales aggregation step.
MAX_SALES_RESULTS = 5000
# Maximum products fetched for restock computation.
MAX_PRODUCTS = 2000


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    """
    Resolve a store identifier to an ObjectId.

    Accepts either a valid 24-char hex ObjectId string or a store name.
    Returns None if not found.
    """
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass

    doc = await db["stores"].find_one(
        {"name": store_id},
        {"_id": 1},
    )
    if doc:
        return doc["_id"]

    LOGGER.warning("restock_store_not_found", store_id=store_id)
    return None


async def fetch_restock_priorities(store_id: str) -> list[dict]:
    """
    Computes restock priority per product using actual stock + real avg daily sales.

    Priority:
        RED    — out of stock OR days_to_stockout <= lead_time
        YELLOW — days_to_stockout <= lead_time * 2
        GREEN  — healthy stock
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return []
    cutoff = datetime.utcnow() - timedelta(days=SALES_LOOKBACK_DAYS)

    # 1. Avg daily sales per product over last 30 days
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
    sales_list = await sales_cursor.to_list(length=MAX_SALES_RESULTS)
    sales_map: dict[str, float] = {
        str(s["_id"]): s["total_qty_sold"] / SALES_LOOKBACK_DAYS for s in sales_list
    }

    # 2. All active products
    prod_cursor = db["products"].find(
        {"store": store_oid, "isActive": True},
        {"_id": 1, "name": 1, "quantity": 1, "price": 1},
    )
    products = await prod_cursor.to_list(length=MAX_PRODUCTS)
    LOGGER.debug("fetch_restock_priorities", store_id=store_id, product_count=len(products))

    output = []
    for p in products:
        pid = str(p["_id"])
        current_qty = p.get("quantity", 0)
        avg_daily = sales_map.get(pid, 0.0)

        if avg_daily > 0:
            days_to_stockout = round(current_qty / avg_daily, 1)
        else:
            days_to_stockout = None  # No sales data — can't predict

        # Suggested qty to cover lead time + safety stock buffer
        target_stock = avg_daily * (LEAD_TIME_DAYS + SAFETY_STOCK_DAYS)
        suggested_qty = max(0, round(target_stock - current_qty))

        if current_qty <= 0:
            priority = "RED"
        elif days_to_stockout is not None and days_to_stockout <= LEAD_TIME_DAYS:
            priority = "RED"
        elif days_to_stockout is not None and days_to_stockout <= LEAD_TIME_DAYS * 2:
            priority = "YELLOW"
        elif suggested_qty > 0:
            priority = "YELLOW"
        else:
            priority = "GREEN"

        output.append(
            {
                "product_id": pid,
                "name": p.get("name", "Unknown"),
                "current_quantity": current_qty,
                "avg_daily_sales": round(avg_daily, 2),
                "days_to_stockout": days_to_stockout,
                "suggested_restock_quantity": suggested_qty,
                "priority": priority,
            }
        )

    # Sort: RED first, then YELLOW, then GREEN, then by days_to_stockout asc
    priority_rank = {"RED": 0, "YELLOW": 1, "GREEN": 2}
    output.sort(
        key=lambda x: (
            priority_rank.get(x["priority"], 3),
            x["days_to_stockout"] if x["days_to_stockout"] is not None else 9999,
        )
    )

    return output
