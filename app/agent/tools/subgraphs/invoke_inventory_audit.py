"""
app/agent/tools/subgraphs/invoke_inventory_audit.py
====================================================
Stub tool for the deep_inventory_subgraph.
"""

from pydantic import BaseModel, Field
from langchain_core.tools import tool


class InvokeDeepInventoryInput(BaseModel):
    store_id: str = Field(..., description="The store ID to audit.")
    expiry_alert_days: int = Field(30, description="Days ahead to check for expiry alerts (default 30).")


@tool("invoke_deep_inventory_audit", args_schema=InvokeDeepInventoryInput)
async def invoke_deep_inventory_audit(store_id: str, expiry_alert_days: int = 30) -> dict:
    """
    [SUBGRAPH TOOL] Run a full deep inventory audit for the store.

    This runs the deep_inventory subgraph which performs (in parallel):
      1. Inventory overview: total products, value, low/out-of-stock counts
      2. Category health: per-category health percentage breakdown
      3. Dead stock analysis: products with 0 quantity
      4. Expiry risk: products expiring within expiry_alert_days days
      5. Slow-moving stock: products with quantity > 0 but 0 sales in 30 days
      6. Restock priorities: top 10 RED-priority items
      7. Composite risk score: 'low' | 'medium' | 'high'

    Use when the owner asks:
      - "How is my inventory?"
      - "Give me an inventory report / audit"
      - "What are my inventory risks?"
      - "Show me a full stock analysis"
      - "Which categories are struggling?"

    OUTPUT (DeepInventoryOutput):
      total_products      : int
      total_value         : float   — total inventory value
      low_stock_count     : int
      out_of_stock_count  : int
      dead_stock_count    : int
      slow_moving_count   : int
      expiring_soon       : list    — [{name, exp_date, days_left, quantity}]
      category_health     : list    — [{category, total, low_stock, out_of_stock, health_pct}]
      red_restock_items   : list    — top 10 RED items
      risk_score          : str     — 'low' | 'medium' | 'high'
      audit_at            : str     — ISO timestamp
    """
    return {
        "__subgraph__": "deep_inventory",
        "store_id": store_id,
        "expiry_alert_days": expiry_alert_days,
    }
