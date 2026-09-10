"""
app/services/agent_data_service.py
====================================
Real MongoDB queries used exclusively by the agent tools.

All functions are async and use the shared motor client from
app.config.database. Each function is designed to be fast,
targeted, and return only the data the agent actually needs.

Collections used:
    products  — { store: ObjectId, name, category, quantity, price, isActive }
    sales     — { store: ObjectId, completedAt, items: [{productId, quantity, lineTotal}] }
"""

from __future__ import annotations

from datetime import datetime, timedelta
from bson import ObjectId

import structlog

from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.data_service")


# ---------------------------------------------------------------------------
# Inventory queries
# ---------------------------------------------------------------------------

async def fetch_inventory_summary(store_id: str) -> dict:
    """
    Returns a high-level inventory summary:
        total_products, low_stock_count, out_of_stock_count,
        total_inventory_value, low_stock_threshold_used
    """
    db = get_database()
    THRESHOLD = 10

    pipeline = [
        {"$match": {"store": ObjectId(store_id), "isActive": True}},
        {
            "$group": {
                "_id": None,
                "total_products": {"$sum": 1},
                "low_stock_count": {
                    "$sum": {
                        "$cond": [
                            {"$and": [
                                {"$gt": ["$quantity", 0]},
                                {"$lte": ["$quantity", THRESHOLD]}
                            ]},
                            1, 0
                        ]
                    }
                },
                "out_of_stock_count": {
                    "$sum": {"$cond": [{"$lte": ["$quantity", 0]}, 1, 0]}
                },
                "total_inventory_value": {
                    "$sum": {"$multiply": ["$quantity", "$price"]}
                },
            }
        },
    ]

    cursor = db["products"].aggregate(pipeline)
    result = await cursor.to_list(length=1)
    LOGGER.debug("fetch_inventory_summary", store_id=store_id, result=result)

    if not result:
        return {
            "total_products": 0,
            "low_stock_count": 0,
            "out_of_stock_count": 0,
            "total_inventory_value": 0.0,
            "low_stock_threshold_used": THRESHOLD,
        }

    r = result[0]
    return {
        "total_products": r.get("total_products", 0),
        "low_stock_count": r.get("low_stock_count", 0),
        "out_of_stock_count": r.get("out_of_stock_count", 0),
        "total_inventory_value": round(r.get("total_inventory_value", 0.0), 2),
        "low_stock_threshold_used": THRESHOLD,
    }


async def fetch_low_stock_products(store_id: str, threshold: int = 10) -> list[dict]:
    """
    Returns products whose quantity is > 0 but <= threshold, sorted by quantity asc.
    Each item: { product_id, name, category, current_quantity, price }
    """
    db = get_database()

    cursor = db["products"].find(
        {
            "store": ObjectId(store_id),
            "isActive": True,
            "quantity": {"$gt": 0, "$lte": threshold},
        },
        {"_id": 1, "name": 1, "category": 1, "quantity": 1, "price": 1},
    ).sort("quantity", 1)

    products = await cursor.to_list(length=200)
    LOGGER.debug("fetch_low_stock_products", store_id=store_id, count=len(products))

    return [
        {
            "product_id": str(p["_id"]),
            "name": p.get("name", "Unknown"),
            "category": p.get("category", "General"),
            "current_quantity": p.get("quantity", 0),
            "price": p.get("price", 0.0),
        }
        for p in products
    ]


# ---------------------------------------------------------------------------
# Sales queries
# ---------------------------------------------------------------------------

async def fetch_sales_summary(store_id: str, days_lookback: int = 30) -> dict:
    """
    Returns sales KPIs for the last N days:
        total_revenue, total_sales_count, average_order_value, days_covered
    """
    db = get_database()
    cutoff = datetime.utcnow() - timedelta(days=days_lookback)

    pipeline = [
        {
            "$match": {
                "store": ObjectId(store_id),
                "completedAt": {"$gte": cutoff},
            }
        },
        {
            "$project": {
                "order_total": {"$sum": "$items.lineTotal"},
            }
        },
        {
            "$group": {
                "_id": None,
                "total_revenue": {"$sum": "$order_total"},
                "total_sales_count": {"$sum": 1},
            }
        },
    ]

    cursor = db["sales"].aggregate(pipeline)
    result = await cursor.to_list(length=1)
    LOGGER.debug("fetch_sales_summary", store_id=store_id, days=days_lookback, result=result)

    if not result:
        return {
            "total_revenue": 0.0,
            "total_sales_count": 0,
            "average_order_value": 0.0,
            "days_covered": days_lookback,
        }

    r = result[0]
    count = r.get("total_sales_count", 0)
    revenue = r.get("total_revenue", 0.0)

    return {
        "total_revenue": round(revenue, 2),
        "total_sales_count": count,
        "average_order_value": round(revenue / count, 2) if count > 0 else 0.0,
        "days_covered": days_lookback,
    }


async def fetch_top_selling_products(
    store_id: str, limit: int = 5, days_lookback: int = 30
) -> list[dict]:
    """
    Returns the top N products by total quantity sold in the last N days.
    Each item: { product_id, name, total_quantity_sold, revenue_generated }
    """
    db = get_database()
    cutoff = datetime.utcnow() - timedelta(days=days_lookback)

    pipeline = [
        {
            "$match": {
                "store": ObjectId(store_id),
                "completedAt": {"$gte": cutoff},
            }
        },
        {"$unwind": "$items"},
        {
            "$group": {
                "_id": "$items.productId",
                "total_quantity_sold": {"$sum": "$items.quantity"},
                "revenue_generated": {"$sum": "$items.lineTotal"},
            }
        },
        {"$sort": {"total_quantity_sold": -1}},
        {"$limit": limit},
        # Join with products to get name
        {
            "$lookup": {
                "from": "products",
                "let": {"pid": {"$toObjectId": "$_id"}},
                "pipeline": [
                    {"$match": {"$expr": {"$eq": ["$_id", "$$pid"]}}},
                    {"$project": {"name": 1}},
                ],
                "as": "product_info",
            }
        },
    ]

    cursor = db["sales"].aggregate(pipeline)
    results = await cursor.to_list(length=limit)
    LOGGER.debug("fetch_top_selling_products", store_id=store_id, count=len(results))

    output = []
    for r in results:
        name = "Unknown"
        if r.get("product_info"):
            name = r["product_info"][0].get("name", "Unknown")
        output.append(
            {
                "product_id": str(r["_id"]),
                "name": name,
                "total_quantity_sold": r.get("total_quantity_sold", 0),
                "revenue_generated": round(r.get("revenue_generated", 0.0), 2),
            }
        )

    return output


# ---------------------------------------------------------------------------
# Restock / Forecast queries
# ---------------------------------------------------------------------------

async def fetch_restock_priorities(store_id: str) -> list[dict]:
    """
    Computes restock priority per product using actual stock + real avg daily sales.
    Priority:
        RED    — out of stock OR days_to_stockout <= 3
        YELLOW — days_to_stockout <= 7
        GREEN  — healthy stock

    Each item: { product_id, name, current_quantity, avg_daily_sales,
                 days_to_stockout, suggested_restock_quantity, priority }
    """
    db = get_database()
    LEAD_TIME = 3
    SAFETY_STOCK = 7
    LOOKBACK_DAYS = 30
    cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)

    # 1. Avg daily sales per product over last 30 days
    sales_pipeline = [
        {
            "$match": {
                "store": ObjectId(store_id),
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
    sales_list = await sales_cursor.to_list(length=5000)
    sales_map: dict[str, float] = {
        str(s["_id"]): s["total_qty_sold"] / LOOKBACK_DAYS for s in sales_list
    }

    # 2. All active products
    prod_cursor = db["products"].find(
        {"store": ObjectId(store_id), "isActive": True},
        {"_id": 1, "name": 1, "quantity": 1, "price": 1},
    )
    products = await prod_cursor.to_list(length=2000)
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
        target_stock = (avg_daily * (LEAD_TIME + SAFETY_STOCK))
        suggested_qty = max(0, round(target_stock - current_qty))

        if current_qty <= 0:
            priority = "RED"
        elif days_to_stockout is not None and days_to_stockout <= LEAD_TIME:
            priority = "RED"
        elif days_to_stockout is not None and days_to_stockout <= LEAD_TIME * 2:
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


async def fetch_forecast_summary(store_id: str, horizon_days: int = 7) -> list[dict]:
    """
    Returns a demand forecast per product using avg daily sales * horizon_days.
    Reuses existing aggregation_service logic but returns a clean dict list.
    Each item: { product_id, name, current_stock, predicted_demand, days_to_stockout }
    """
    from app.services.aggregation_service import get_forecast_for_store

    try:
        forecasts = await get_forecast_for_store(store_id)
    except Exception as e:
        LOGGER.error("fetch_forecast_summary_error", store_id=store_id, error=str(e))
        return []

    return [
        {
            "product_id": f["productId"],
            "name": f["productName"],
            "current_stock": f["currentStock"],
            "predicted_demand": round(f["predictedDailyDemand"] * horizon_days, 1),
            "predicted_daily": f["predictedDailyDemand"],
            "trend_percent": f["trendPercent"],
            "days_to_stockout": f["daysToStockout"],
            "horizon_days": horizon_days,
        }
        for f in forecasts
    ]


# ---------------------------------------------------------------------------
# Insights queries
# ---------------------------------------------------------------------------

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
    ).limit(10)
    oos = await oos_cursor.to_list(length=10)
    if oos:
        names = [p.get("name", "Unknown") for p in oos]
        insights.append(
            {
                "type": "out_of_stock",
                "title": "Products Out of Stock",
                "summary": f"{len(oos)} product(s) are out of stock: {', '.join(names[:3])}{'...' if len(names) > 3 else ''}. Restock immediately.",
                "severity": "danger",
                "affected_products": [str(p["_id"]) for p in oos],
            }
        )

    # --- Dead stock (no sales in 60 days, qty > 0) ---
    dead_cutoff = datetime.utcnow() - timedelta(days=60)
    sold_pipeline = [
        {"$match": {"store": ObjectId(store_id), "completedAt": {"$gte": dead_cutoff}}},
        {"$unwind": "$items"},
        {"$group": {"_id": "$items.productId"}},
    ]
    sold_cursor = db["sales"].aggregate(sold_pipeline)
    sold_ids = {r["_id"] for r in await sold_cursor.to_list(length=5000)}

    all_products = await db["products"].find(
        {"store": ObjectId(store_id), "isActive": True, "quantity": {"$gt": 0}},
        {"_id": 1, "name": 1, "quantity": 1, "price": 1},
    ).to_list(length=2000)

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
                    f"{len(dead_stock)} product(s) have had zero sales in the last 60 days "
                    f"with ₹{round(dead_value, 2)} of stock value tied up. "
                    f"Consider discounting: {', '.join(names)}{'...' if len(dead_stock) > 3 else ''}."
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
                "completedAt": {"$gte": datetime.utcnow() - timedelta(days=30)},
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
        {"$limit": 3},
    ]
    top_cursor = db["sales"].aggregate(top_pipeline)
    top_sellers = await top_cursor.to_list(length=3)

    if top_sellers:
        # Look up names
        top_ids = [ObjectId(r["_id"]) for r in top_sellers if r["_id"]]
        name_cursor = db["products"].find({"_id": {"$in": top_ids}}, {"_id": 1, "name": 1})
        name_map = {str(p["_id"]): p.get("name", "Unknown") for p in await name_cursor.to_list(length=3)}

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
                    f"Top earner: {leaders[0]['name']} with ₹{leaders[0]['revenue']} revenue. "
                    f"These products drive most of your sales."
                ),
                "severity": "info",
                "leaders": leaders,
            }
        )

    LOGGER.debug("fetch_store_insights", store_id=store_id, insight_count=len(insights))
    return insights
