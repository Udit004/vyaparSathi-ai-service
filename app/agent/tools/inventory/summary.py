from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

from app.agent.service import fetch_inventory_summary

class InventorySummaryInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")

class InventorySummaryOutput(BaseModel):
    total_products: int
    low_stock_count: int
    out_of_stock_count: int
    total_inventory_value: float
    low_stock_threshold_used: int
    fetched_at: str

@tool("get_inventory_summary", args_schema=InventorySummaryInput)
async def get_inventory_summary(store_id: str) -> InventorySummaryOutput:
    """
    Get a high-level summary of the store's inventory, including total products,
    low stock counts, out of stock counts, and total inventory value.
    """
    data = await fetch_inventory_summary(store_id)
    return InventorySummaryOutput(
        total_products=data["total_products"],
        low_stock_count=data["low_stock_count"],
        out_of_stock_count=data["out_of_stock_count"],
        total_inventory_value=data["total_inventory_value"],
        low_stock_threshold_used=data["low_stock_threshold_used"],
        fetched_at=datetime.utcnow().isoformat()
    )
