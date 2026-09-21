from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import fetch_category_performance

class CategoryPerformanceInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")

class CategoryPerformanceItem(BaseModel):
    category: str
    revenue: float
    units_sold: int

class CategoryPerformanceOutput(BaseModel):
    categories: List[CategoryPerformanceItem]
    fetched_at: str

@tool("get_category_performance", args_schema=CategoryPerformanceInput)
async def get_category_performance(store_id: str) -> CategoryPerformanceOutput:
    """
    Get sales metrics aggregated by category.
    """
    items = await fetch_category_performance(store_id)
    return CategoryPerformanceOutput(
        categories=[CategoryPerformanceItem(**i) for i in items],
        fetched_at=datetime.utcnow().isoformat()
    )
