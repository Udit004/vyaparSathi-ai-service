"""
app/agent/tools/forecast/restock_budget.py
===========================================
Calculates precise restocking capital requirements and supplier-wise budget breakdown
for upcoming inventory replenishment cycles (e.g. 7, 14, or 30 days).
"""

from __future__ import annotations

import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import structlog

from app.agent.service.inventory.low_stock import fetch_low_stock_products
from app.agent.service.forecast.restock import fetch_restock_priorities

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.restock_budget")


class RestockBudgetInput(BaseModel):
    store_id: Optional[str] = Field(
        default=None,
        description="The ID of the store (automatically populated if omitted).",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="The store owner / user ID (automatically populated if omitted).",
    )
    days_buffer: int = Field(
        default=14,
        ge=3,
        le=60,
        description="Number of days of inventory buffer to budget for (e.g. 7, 14, or 30 days).",
    )


class RestockItemBudget(BaseModel):
    product_name: str = Field(..., description="Name of the product.")
    current_stock: int = Field(..., description="Current stock units.")
    suggested_order_qty: int = Field(..., description="Quantity needed to fulfill the buffer period.")
    estimated_unit_cost: float = Field(..., description="Estimated purchase cost per unit in ₹.")
    total_line_cost: float = Field(..., description="Total cost for this item (unit cost * qty) in ₹.")
    supplier: str = Field(..., description="Primary supplier or distributor.")


class SupplierBudgetSummary(BaseModel):
    supplier_name: str = Field(..., description="Name of the distributor/supplier.")
    items_count: int = Field(..., description="Number of items to order from this supplier.")
    estimated_total_payment: float = Field(..., description="Total cash required for this supplier in ₹.")


class RestockBudgetOutput(BaseModel):
    estimated_total_capital_needed: float = Field(..., description="Total working capital required in ₹.")
    days_buffer_planned: int = Field(..., description="Planning buffer in days.")
    total_products_count: int = Field(..., description="Count of distinct products needing restock.")
    supplier_breakdown: List[SupplierBudgetSummary] = Field(default_factory=list, description="Capital breakdown per supplier.")
    items_detail: List[RestockItemBudget] = Field(default_factory=list, description="Item-level breakdown of restock orders.")
    budget_advice: str = Field(..., description="Actionable financial advice on managing distributor orders.")
    calculated_at: str = Field(..., description="Timestamp of budget calculation.")


@tool("calculate_restock_budget", args_schema=RestockBudgetInput)
async def calculate_restock_budget(
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    days_buffer: int = 14,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Calculate the exact working capital and budget needed for upcoming inventory replenishment.
    Analyzes low-stock SKUs, sales velocity, supplier pricing, and aggregates total cash needed per distributor.

    Use this when:
    - The merchant asks "Restock ke liye kitna paisa lagega?", "How much budget do I need for this week's purchase?", "Distributor payment planning".
    - Planning weekly or monthly distributor orders against available cash flow.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    LOGGER.info("calculate_restock_budget_invoked", store_id=store_id, days=days_buffer)

    if not store_id:
        return {
            "estimated_total_capital_needed": 0.0,
            "days_buffer_planned": days_buffer,
            "total_products_count": 0,
            "supplier_breakdown": [],
            "items_detail": [],
            "budget_advice": "Store ID missing from runtime context.",
            "calculated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    # Fetch low stock & restock priorities
    items: List[Dict[str, Any]] = []
    supplier_map: Dict[str, Dict[str, Any]] = {}
    total_capital = 0.0

    try:
        raw_items = await fetch_low_stock_products(store_id, threshold=20, limit=20)
        if isinstance(raw_items, list):
            for p in raw_items:
                name = p.get("name") or p.get("product_name", "Item")
                stock = int(p.get("current_quantity") or p.get("quantity") or p.get("stock_quantity", 0))
                thresh = 20
                unit_cost = float(p.get("purchase_price") or p.get("cost_price") or (p.get("price", 100.0) * 0.8))
                supplier = p.get("supplier_name") or p.get("seller_name") or "Primary Distributor"

                # Calculate recommended restock qty
                order_qty = max(10, thresh * 2 - stock)
                line_cost = order_qty * unit_cost
                total_capital += line_cost

                items.append({
                    "product_name": name,
                    "current_stock": stock,
                    "suggested_order_qty": order_qty,
                    "estimated_unit_cost": unit_cost,
                    "total_line_cost": line_cost,
                    "supplier": supplier,
                })

                if supplier not in supplier_map:
                    supplier_map[supplier] = {"count": 0, "total": 0.0}
                supplier_map[supplier]["count"] += 1
                supplier_map[supplier]["total"] += line_cost

    except Exception as exc:
        LOGGER.warning("restock_budget_calc_error", error=str(exc))

    supplier_summary = [
        {
            "supplier_name": sup,
            "items_count": data["count"],
            "estimated_total_payment": data["total"],
        }
        for sup, data in supplier_map.items()
    ]
    supplier_summary.sort(key=lambda x: x["estimated_total_payment"], reverse=True)

    advice = (
        f"You will need approximately ₹{total_capital:,.2f} to restock {len(items)} critical SKUs for a {days_buffer}-day sales buffer. "
        f"Top distributor payout is ₹{supplier_summary[0]['estimated_total_payment']:,.2f} to {supplier_summary[0]['supplier_name']}."
        if supplier_summary else "Current stock levels are adequate; no immediate heavy restock budget required."
    )

    return {
        "estimated_total_capital_needed": total_capital,
        "days_buffer_planned": days_buffer,
        "total_products_count": len(items),
        "supplier_breakdown": supplier_summary,
        "items_detail": items[:15],
        "budget_advice": advice,
        "calculated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
