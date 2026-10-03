"""
app/agent/tools/insights/daily_action_checklist.py
===================================================
Generates a prioritized, daily morning action checklist for the retail store owner.
Combines:
1. Urgent low-stock restock actions.
2. High-priority customer credit/udhaar collections.
3. Expiry alerts for perishable stock.
4. Progress tracking towards the owner's monthly business targets.
"""

from __future__ import annotations

import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import structlog

from app.agent.service.inventory.low_stock import fetch_low_stock_products
from app.agent.service.inventory.expiry_alerts import fetch_expiry_alerts
from app.agent.service.buyers.dues import fetch_buyer_dues
from app.agent.service.sales.summary import fetch_sales_summary
from app.agent.tools.memory.search import get_owner_goals_and_preferences

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.daily_action_checklist")


class DailyChecklistInput(BaseModel):
    store_id: Optional[str] = Field(
        default=None,
        description="The ID of the store (automatically injected if omitted).",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="The store owner / user ID (automatically injected if omitted).",
    )


class RestockActionItem(BaseModel):
    product_name: str = Field(..., description="Name of the low stock product.")
    current_stock: int = Field(..., description="Units remaining in inventory.")
    threshold: int = Field(..., description="Minimum safety stock threshold.")
    urgency: str = Field(..., description="Urgency level: CRITICAL or HIGH.")
    action: str = Field(..., description="Recommended reorder action.")


class CollectionActionItem(BaseModel):
    customer_name: str = Field(..., description="Customer name.")
    phone: str = Field(..., description="Customer phone number.")
    amount_due: float = Field(..., description="Outstanding credit amount.")
    action: str = Field(..., description="Recommended collection action.")


class ExpiryWarningItem(BaseModel):
    product_name: str = Field(..., description="Product nearing expiration.")
    stock_quantity: int = Field(..., description="Units remaining.")
    expiry_date: str = Field(..., description="Expiry date.")
    days_left: int = Field(..., description="Days remaining until expiry.")
    action: str = Field(..., description="Recommended clearance action.")


class DailyChecklistOutput(BaseModel):
    summary_headline: str = Field(..., description="High-level executive morning summary.")
    urgent_restocks: List[RestockActionItem] = Field(default_factory=list, description="Top urgent restocks needed today.")
    pending_collections: List[CollectionActionItem] = Field(default_factory=list, description="Top customer credit collections to follow up today.")
    expiry_warnings: List[ExpiryWarningItem] = Field(default_factory=list, description="Products requiring immediate promotional clearance.")
    owner_goal_status: Dict[str, Any] = Field(default_factory=dict, description="Active merchant revenue/profit targets and current status.")
    total_action_items: int = Field(..., description="Total count of actions requiring merchant attention today.")
    generated_at: str = Field(..., description="Timestamp when the checklist was generated.")


@tool("get_daily_action_checklist", args_schema=DailyChecklistInput)
async def get_daily_action_checklist(
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Generate an intelligent, prioritized Daily Action Checklist for the store owner.
    Synthesizes critical stockout risks, pending udhaar/credit collections, imminent product expiries,
    and monthly revenue goal progress into a clear 1-minute morning action briefing.

    Use this when:
    - The merchant asks "Aaj mujhe kya karna chahiye?", "What are my priorities today?", "Morning briefing".
    - The owner wants a quick overview of urgent operational tasks for the day.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    LOGGER.info("daily_action_checklist_invoked", store_id=store_id, user_id=user_id)

    if not store_id:
        return {
            "summary_headline": "Store ID missing from context.",
            "urgent_restocks": [],
            "pending_collections": [],
            "expiry_warnings": [],
            "owner_goal_status": {},
            "total_action_items": 0,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    # 1. Fetch Low Stock Items
    restock_actions: List[Dict[str, Any]] = []
    try:
        raw_low_stock = await fetch_low_stock_products(store_id, threshold=15, limit=5)
        if isinstance(raw_low_stock, list):
            for item in raw_low_stock[:4]:
                name = item.get("name") or item.get("product_name", "Unknown Item")
                qty = int(item.get("current_quantity") or item.get("quantity", 0))
                thresh = 15
                urgency = "CRITICAL" if qty <= 3 else "HIGH"
                restock_actions.append({
                    "product_name": name,
                    "current_stock": qty,
                    "threshold": thresh,
                    "urgency": urgency,
                    "action": f"Order {max(10, thresh * 2)} units from distributor immediately (Current: {qty} units left)",
                })
    except Exception as exc:
        LOGGER.warning("daily_checklist_low_stock_error", error=str(exc))

    # 2. Fetch Pending Credit (Udhaar) Collections
    collection_actions: List[Dict[str, Any]] = []
    try:
        dues_data = await fetch_buyer_dues(store_id, min_due=100.0, limit=5)
        raw_buyers = dues_data.get("buyers", []) if isinstance(dues_data, dict) else []
        for b in raw_buyers[:4]:
            b_name = b.get("name", "Customer")
            b_phone = b.get("phone", "N/A")
            b_due = float(b.get("total_due", 0.0))
            if b_due > 0:
                collection_actions.append({
                    "customer_name": b_name,
                    "phone": b_phone,
                    "amount_due": b_due,
                    "action": f"Send friendly payment reminder for ₹{b_due:,.2f}",
                })
    except Exception as exc:
        LOGGER.warning("daily_checklist_buyer_dues_error", error=str(exc))

    # 3. Fetch Expiry Warnings
    expiry_actions: List[Dict[str, Any]] = []
    try:
        expiry_data = await fetch_expiry_alerts(store_id, alert_days=30)
        critical_alerts = (expiry_data.get("critical", []) if isinstance(expiry_data, dict) else []) + (expiry_data.get("warning", []) if isinstance(expiry_data, dict) else [])
        for exp in critical_alerts[:3]:
            p_name = exp.get("product_name") or exp.get("name", "Item")
            p_qty = int(exp.get("quantity") or exp.get("stock_quantity", 0))
            p_date = str(exp.get("exp_date", "Soon"))
            days_left = int(exp.get("days_left", 15))
            expiry_actions.append({
                "product_name": p_name,
                "stock_quantity": p_qty,
                "expiry_date": p_date,
                "days_left": days_left,
                "action": f"Put on {15 if days_left > 10 else 25}% clearance discount or bundle deal ({days_left} days left)",
            })
    except Exception as exc:
        LOGGER.warning("daily_checklist_expiry_error", error=str(exc))

    # 4. Fetch Owner Targets & Sales Status
    owner_goal_status: Dict[str, Any] = {}
    try:
        goals_data = await get_owner_goals_and_preferences.ainvoke({"store_id": store_id, "user_id": user_id})
        sales_data = await fetch_sales_summary(store_id, days_lookback=30)
        mtd_sales = float(sales_data.get("total_revenue", 0.0) if isinstance(sales_data, dict) else 0.0)
        goals_list = goals_data.get("owner_goals", []) if isinstance(goals_data, dict) else []

        owner_goal_status = {
            "recorded_goals": goals_list,
            "current_month_to_date_sales": f"₹{mtd_sales:,.2f}",
            "summary": "Keep focus on fast-moving categories to maintain progress towards targets.",
        }
    except Exception as exc:
        LOGGER.warning("daily_checklist_goals_error", error=str(exc))

    total_tasks = len(restock_actions) + len(collection_actions) + len(expiry_actions)

    headline = f"Today's Action Plan: {len(restock_actions)} restock alerts, {len(collection_actions)} payment collections, and {len(expiry_actions)} expiry clearances."

    return {
        "summary_headline": headline,
        "urgent_restocks": restock_actions,
        "pending_collections": collection_actions,
        "expiry_warnings": expiry_actions,
        "owner_goal_status": owner_goal_status,
        "total_action_items": total_tasks,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
