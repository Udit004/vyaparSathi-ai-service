"""app/agent/tools/sales/discount_impact.py"""
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service.sales.discount_impact import fetch_discount_impact


class DiscountImpactInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    days: int = Field(30, description="Number of past days to analyze (default 30).")


class DiscountImpactOutput(BaseModel):
    total_gross_revenue: float      # Revenue before discounts
    total_discount_given: float     # Total discount amount applied
    total_net_revenue: float        # Actual revenue collected
    discount_rate_pct: float        # discount / gross * 100
    discounted_sales_count: int     # Sales that had a discount
    total_sales_count: int
    days_covered: int
    fetched_at: str


@tool("get_discount_impact", args_schema=DiscountImpactInput)
async def get_discount_impact(store_id: str, days: int = 30) -> DiscountImpactOutput:
    """
    Analyze the impact of discounts on revenue. Returns gross revenue (before
    discounts), total discount amount given, net revenue collected, and the
    effective discount rate percentage. Use when the owner asks about how much
    revenue is being lost to discounts, or whether discounting strategy is healthy.
    """
    data = await fetch_discount_impact(store_id, days)
    return DiscountImpactOutput(
        total_gross_revenue=data["total_gross_revenue"],
        total_discount_given=data["total_discount_given"],
        total_net_revenue=data["total_net_revenue"],
        discount_rate_pct=data["discount_rate_pct"],
        discounted_sales_count=data["discounted_sales_count"],
        total_sales_count=data["total_sales_count"],
        days_covered=data["days_covered"],
        fetched_at=datetime.utcnow().isoformat(),
    )
