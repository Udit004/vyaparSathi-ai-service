from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

from app.agent.service import fetch_sales_summary, fetch_top_selling_products

# --- Schemas ---

class SalesSummaryInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    days_lookback: int = Field(30, description="Number of days to look back for the summary.")

class SalesSummaryOutput(BaseModel):
    total_revenue: float
    total_sales_count: int
    average_order_value: float
    days_covered: int
    fetched_at: str

class TopSellingInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    limit: int = Field(5, description="Maximum number of products to return.")
    days_lookback: int = Field(30, description="Number of days to look back.")

class TopSellingItem(BaseModel):
    product_id: str
    name: str
    total_quantity_sold: float
    revenue_generated: float

class TopSellingOutput(BaseModel):
    items: List[TopSellingItem]
    fetched_at: str


# --- Tools ---

@tool("get_sales_summary", args_schema=SalesSummaryInput)
async def get_sales_summary(store_id: str, days_lookback: int = 30) -> SalesSummaryOutput:
    """
    Get a high-level summary of the store's sales over a specified period,
    including total revenue, total sales count, and average order value.
    """
    data = await fetch_sales_summary(store_id, days_lookback)
    return SalesSummaryOutput(
        total_revenue=data["total_revenue"],
        total_sales_count=data["total_sales_count"],
        average_order_value=data["average_order_value"],
        days_covered=data["days_covered"],
        fetched_at=datetime.utcnow().isoformat()
    )


@tool("get_top_selling_products", args_schema=TopSellingInput)
async def get_top_selling_products(store_id: str, limit: int = 5, days_lookback: int = 30) -> TopSellingOutput:
    """
    Get a list of the top selling products by revenue and quantity over a specified period.
    """
    items = await fetch_top_selling_products(store_id, limit, days_lookback)
    return TopSellingOutput(
        items=[TopSellingItem(**i) for i in items],
        fetched_at=datetime.utcnow().isoformat()
    )
