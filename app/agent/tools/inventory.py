from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

# --- Schemas ---

class InventorySummaryInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")

class InventorySummaryOutput(BaseModel):
    total_products: int
    low_stock_count: int
    out_of_stock_count: int
    total_inventory_value: float
    low_stock_threshold_used: int
    fetched_at: str

class LowStockInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    threshold: int = Field(10, description="The quantity threshold below which a product is considered low stock.")

class LowStockItem(BaseModel):
    product_id: str
    name: str
    category: str
    current_quantity: float
    price: float

class LowStockOutput(BaseModel):
    items: List[LowStockItem]
    count: int
    fetched_at: str


# --- Tools ---

@tool("get_inventory_summary", args_schema=InventorySummaryInput)
async def get_inventory_summary(store_id: str) -> InventorySummaryOutput:
    """
    Get a high-level summary of the store's inventory, including total products,
    low stock counts, out of stock counts, and total inventory value.
    """
    from app.services.agent_data_service import fetch_inventory_summary
    data = await fetch_inventory_summary(store_id)
    return InventorySummaryOutput(
        total_products=data["total_products"],
        low_stock_count=data["low_stock_count"],
        out_of_stock_count=data["out_of_stock_count"],
        total_inventory_value=data["total_inventory_value"],
        low_stock_threshold_used=data["low_stock_threshold_used"],
        fetched_at=datetime.utcnow().isoformat()
    )


@tool("get_low_stock_products", args_schema=LowStockInput)
async def get_low_stock_products(store_id: str, threshold: int = 10) -> LowStockOutput:
    """
    Get a list of specific products that are running low on stock (below the given threshold).
    Useful for identifying exactly what needs to be ordered.
    """
    from app.services.agent_data_service import fetch_low_stock_products
    items = await fetch_low_stock_products(store_id, threshold)
    return LowStockOutput(
        items=[LowStockItem(**i) for i in items],
        count=len(items),
        fetched_at=datetime.utcnow().isoformat()
    )
