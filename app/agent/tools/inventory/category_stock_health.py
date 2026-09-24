"""app/agent/tools/inventory/category_stock_health.py"""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service.inventory.category_stock_health import fetch_category_stock_health


class CategoryStockHealthInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    threshold: int = Field(10, description="Low-stock threshold units (default 10).")


class CategoryHealthItem(BaseModel):
    category: str
    total: int
    low_stock: int
    out_of_stock: int
    healthy: int
    health_pct: float    # % of products in healthy state
    total_value: float


class CategoryStockHealthOutput(BaseModel):
    categories: List[CategoryHealthItem]
    total_categories: int
    fetched_at: str


@tool("get_category_stock_health", args_schema=CategoryStockHealthInput)
async def get_category_stock_health(store_id: str, threshold: int = 10) -> CategoryStockHealthOutput:
    """
    Get a per-category breakdown of inventory health showing total products,
    low stock count, out-of-stock count, health percentage, and total value.
    Unhealthiest categories appear first. Use to identify which product categories
    need immediate attention.
    """
    data = await fetch_category_stock_health(store_id, threshold)
    return CategoryStockHealthOutput(
        categories=[CategoryHealthItem(**i) for i in data],
        total_categories=len(data),
        fetched_at=datetime.utcnow().isoformat(),
    )
