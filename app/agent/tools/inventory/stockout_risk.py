"""
app/agent/tools/inventory/stockout_risk.py
===========================================
Discovery tool: products likely to run out within N days.

This is a Stage 1 discovery tool — it narrows the full product catalogue
to a small set of urgent candidates. Use it before calling detailed
tools like get_product_details or get_demand_forecast.

Example flow:
    User: "Which products will run out in the next 7 days?"
    Think → get_stockout_risk_products(days=7, limit=50)
           → [20 candidates]
    Observe → Context Builder → LLM
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

from app.agent.service.inventory.stockout_risk import fetch_stockout_risk_products


class StockoutRiskInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    days: int = Field(
        7,
        description=(
            "Risk horizon in days. Products running out within this many days "
            "are returned. Default: 7. Range: 1–90."
        ),
    )
    limit: int = Field(
        50,
        description="Maximum number of products to return. Default: 50. Max: 200.",
    )


class StockoutRiskItem(BaseModel):
    product_id: str
    name: str
    category: str
    current_quantity: float
    avg_daily_sales: float
    days_to_stockout: float
    price: float


class StockoutRiskOutput(BaseModel):
    items: List[StockoutRiskItem]
    count: int
    risk_horizon_days: int
    fetched_at: str


@tool("get_stockout_risk_products", args_schema=StockoutRiskInput)
async def get_stockout_risk_products(
    store_id: str,
    days: int = 7,
    limit: int = 50,
) -> StockoutRiskOutput:
    """
    [DISCOVERY TOOL] Find products likely to run out within the given number of days.

    Use this as a Stage 1 discovery step to identify urgent restock candidates
    before running detailed analysis. Returns at most `limit` products sorted by
    urgency (most critical first).

    Key behaviour:
    - Only products with actual sales history are included (no hallucinated risk)
    - Products with no sales in 30 days are excluded (no velocity = no risk)
    - Results are sorted by days_to_stockout ascending (most urgent first)
    - Store isolation: always scoped to the authenticated store_id

    When to use:
    - "Which products will run out soon?"
    - "What should I restock urgently?"
    - "Show me stockout risks for the next 7 / 14 / 30 days"

    Formula: days_to_stockout = current_quantity / avg_daily_sales (last 30 days)
    """
    items = await fetch_stockout_risk_products(store_id, days=days, limit=limit)
    return StockoutRiskOutput(
        items=[StockoutRiskItem(**i) for i in items],
        count=len(items),
        risk_horizon_days=days,
        fetched_at=datetime.utcnow().isoformat(),
    )
