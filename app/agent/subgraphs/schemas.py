"""
app/agent/subgraphs/schemas.py
================================
Pydantic Input/Output schemas for all Vyapar Copilot subgraphs.

These are the contracts between the main graph and each subgraph.
The main graph's subgraph_router node passes an Input model to
the subgraph and receives an Output model back, which is then
written to ``VyaparAgentState.subgraph_result`` for the LLM to
synthesize into the final response.

Schema summary
--------------

MorningBriefingInput / MorningBriefingOutput
    Daily digest: 7-day KPIs + low-stock alerts + top RED restocks
    + LLM-generated briefing text.

DeepInventoryInput / DeepInventoryOutput
    Full inventory audit: category health, dead/slow/expiry risks,
    restock priorities, computed risk score.

SmartRestockInput / SmartRestockOutput
    Prioritised restock order plan: ranked order items with estimated
    cost, supplier hints, and LLM-formatted order memo.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Morning Briefing Subgraph
# ---------------------------------------------------------------------------

class MorningBriefingInput(BaseModel):
    """
    Input for the morning_briefing subgraph.
    Passed by the subgraph_router node from VyaparAgentState.
    """
    store_id: str = Field(..., description="MongoDB ObjectId of the store.")
    owner_name: str = Field("", description="Owner's display name for personalised greeting.")
    currency: str = Field("INR", description="Store currency code.")
    low_stock_threshold: int = Field(10, description="Units at/below which stock is flagged low.")


class MorningBriefingOutput(BaseModel):
    """
    Output of the morning_briefing subgraph.
    Written to VyaparAgentState.subgraph_result['output'].
    """
    revenue_7d: float = Field(..., description="Total revenue in the last 7 days.")
    sales_count_7d: int = Field(..., description="Number of transactions in the last 7 days.")
    low_stock_count: int = Field(..., description="Products currently at or below low-stock threshold.")
    out_of_stock_count: int = Field(..., description="Products currently out of stock.")
    expiring_soon_count: int = Field(..., description="Products expiring within 7 days.")
    top_red_restock: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top 5 RED-priority products needing immediate restocking. "
                    "Each: {name, current_quantity, suggested_restock_quantity, priority}."
    )
    briefing_text: str = Field(..., description="LLM-generated concise morning briefing for the owner.")
    generated_at: str = Field(..., description="ISO timestamp of generation.")


# ---------------------------------------------------------------------------
# Deep Inventory Audit Subgraph
# ---------------------------------------------------------------------------

class DeepInventoryInput(BaseModel):
    """Input for the deep_inventory subgraph."""
    store_id: str = Field(..., description="MongoDB ObjectId of the store.")
    currency: str = Field("INR", description="Store currency code.")
    low_stock_threshold: int = Field(10, description="Low-stock threshold in units.")
    expiry_alert_days: int = Field(30, description="Days ahead to look for expiry alerts.")


class DeepInventoryOutput(BaseModel):
    """
    Output of the deep_inventory subgraph.
    Written to VyaparAgentState.subgraph_result['output'].
    """
    total_products: int = Field(..., description="Total active products in inventory.")
    total_value: float = Field(..., description="Total inventory value (quantity * price).")
    low_stock_count: int
    out_of_stock_count: int
    dead_stock_count: int = Field(..., description="Products with 0 quantity.")
    slow_moving_count: int = Field(..., description="Products with stock but 0 sales in 30 days.")
    expiring_soon: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Products expiring within 7 days. Each: {name, exp_date, days_left, quantity}."
    )
    category_health: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Per-category health breakdown. Each: {category, total, low_stock, out_of_stock, health_pct}."
    )
    red_restock_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top 10 RED-priority restock items."
    )
    risk_score: str = Field(..., description="Overall inventory risk: 'low' | 'medium' | 'high'.")
    audit_at: str = Field(..., description="ISO timestamp of audit.")


# ---------------------------------------------------------------------------
# Smart Restock Order Subgraph
# ---------------------------------------------------------------------------

class SmartRestockInput(BaseModel):
    """Input for the smart_restock subgraph."""
    store_id: str = Field(..., description="MongoDB ObjectId of the store.")
    currency: str = Field("INR", description="Store currency code.")
    lead_time_days: int = Field(3, description="Supplier lead time in days.")
    include_yellow: bool = Field(True, description="Include YELLOW priority items in the order plan.")


class SmartRestockOutput(BaseModel):
    """
    Output of the smart_restock subgraph.
    Written to VyaparAgentState.subgraph_result['output'].
    """
    order_items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Ranked order items. Each: {name, priority, current_quantity, "
                    "suggested_restock_quantity, estimated_cost, category, days_to_stockout}."
    )
    total_estimated_cost: float = Field(..., description="Sum of (suggested_qty * price) for all order items.")
    red_count: int = Field(..., description="Number of RED priority items in the order.")
    yellow_count: int = Field(..., description="Number of YELLOW priority items in the order.")
    order_text: str = Field(..., description="LLM-formatted human-readable restock order memo.")
    generated_at: str = Field(..., description="ISO timestamp of generation.")
