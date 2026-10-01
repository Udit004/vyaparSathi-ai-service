"""
app/agent/service/profit_loss/report.py
=========================================
True Profit & Loss report using:
  Revenue   = sum of sales.totalAmount
  COGS      = sum of (sale_item.unitBuyingPrice * quantity)  [with product.buyingPrice fallback]
  Gross Profit = Revenue - COGS
  Expenses  = sum of expenses.amount
  Net Profit = Gross Profit - Expenses
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.profit_loss.report")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    LOGGER.warning("profit_loss_store_not_found", store_id=store_id)
    return None


async def fetch_profit_loss_report(store_id: str, days_lookback: int = 30) -> dict:
    """
    Returns a complete P&L report for the store over the given period.
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return {}

    since = datetime.now(timezone.utc) - timedelta(days=days_lookback)

    # --- 1. Build product buyingPrice fallback map ---
    prod_cursor = db["products"].find(
        {"store": store_oid, "isActive": True},
        {"buyingPrice": 1}
    )
    product_buying_map: dict[str, float] = {}
    async for p in prod_cursor:
        product_buying_map[str(p["_id"])] = float(p.get("buyingPrice", 0))

    # --- 2. Aggregate sales in range ---
    sales_cursor = db["sales"].find(
        {"store": store_oid, "completedAt": {"$gte": since}},
        {"totalAmount": 1, "items": 1, "discount": 1}
    )

    total_revenue = 0.0
    total_cogs = 0.0
    total_units_sold = 0
    total_orders = 0
    total_discount = 0.0

    async for sale in sales_cursor:
        total_revenue += float(sale.get("totalAmount", 0))
        total_orders += 1
        disc = sale.get("discount", {})
        total_discount += float(disc.get("amount", 0)) if isinstance(disc, dict) else 0.0

        for item in sale.get("items", []):
            qty = float(item.get("quantity", 0))
            total_units_sold += int(qty)
            unit_buying = item.get("unitBuyingPrice")
            if unit_buying is not None:
                total_cogs += float(unit_buying) * qty
            else:
                pid = str(item.get("productId", ""))
                fallback = product_buying_map.get(pid, 0.0)
                total_cogs += fallback * qty

    # --- 3. Aggregate expenses in range ---
    exp_pipeline = [
        {"$match": {"store": store_oid, "date": {"$gte": since}}},
        {"$group": {"_id": None, "total": {"$sum": "$amount"}}},
    ]
    exp_cursor = db["expenses"].aggregate(exp_pipeline)
    exp_results = await exp_cursor.to_list(length=1)
    total_expenses = round(exp_results[0]["total"], 2) if exp_results else 0.0

    gross_profit = total_revenue - total_cogs
    net_profit = gross_profit - total_expenses
    gross_margin = round((gross_profit / total_revenue) * 100, 2) if total_revenue > 0 else 0.0
    net_margin = round((net_profit / total_revenue) * 100, 2) if total_revenue > 0 else 0.0

    LOGGER.debug(
        "fetch_profit_loss_report",
        store_id=store_id,
        days=days_lookback,
        revenue=total_revenue,
        cogs=total_cogs,
        gross_profit=gross_profit,
        net_profit=net_profit,
    )

    return {
        "revenue": round(total_revenue, 2),
        "total_discount": round(total_discount, 2),
        "cogs": round(total_cogs, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_margin_pct": gross_margin,
        "total_expenses": total_expenses,
        "net_profit": round(net_profit, 2),
        "net_margin_pct": net_margin,
        "total_orders": total_orders,
        "total_units_sold": total_units_sold,
        "days_covered": days_lookback,
        "is_profitable": net_profit > 0,
    }
