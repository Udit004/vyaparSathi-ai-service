"""app/agent/tools/sales/profit_margin.py"""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service.sales.profit_margin import fetch_profit_margin_analysis


class ProfitMarginInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")


class MarginCategoryItem(BaseModel):
    category: str
    product_count: int
    avg_cost_price: float       # product.price (base/cost price)
    avg_selling_price: float    # inventory.sellingPrice
    avg_margin_pct: float       # estimated gross margin %
    total_potential_profit: float


class ProfitMarginOutput(BaseModel):
    categories: List[MarginCategoryItem]
    total_categories: int
    fetched_at: str


@tool("get_profit_margin_analysis", args_schema=ProfitMarginInput)
async def get_profit_margin_analysis(store_id: str) -> ProfitMarginOutput:
    """
    Get estimated profit margin analysis per product category. Compares the
    base product price (cost) against the inventory selling price to estimate
    gross margin percentage. Categories with lowest margins appear first.
    Use when the owner asks about profitability, which categories make the most
    money, or where to focus pricing strategy.

    Note: margin estimates depend on inventory.sellingPrice being set. If not
    set, product.price is used as a proxy (margin will show 0%).
    """
    data = await fetch_profit_margin_analysis(store_id)
    return ProfitMarginOutput(
        categories=[MarginCategoryItem(**i) for i in data],
        total_categories=len(data),
        fetched_at=datetime.utcnow().isoformat(),
    )
