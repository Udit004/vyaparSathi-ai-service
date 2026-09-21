from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

from app.agent.service import fetch_top_selling_products

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
