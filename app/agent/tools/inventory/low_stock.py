from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

from app.agent.service import fetch_low_stock_products

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

@tool("get_low_stock_products", args_schema=LowStockInput)
async def get_low_stock_products(store_id: str, threshold: int = 10) -> LowStockOutput:
    """
    Get a list of specific products that are running low on stock (below the given threshold).
    Useful for identifying exactly what needs to be ordered.
    """
    items = await fetch_low_stock_products(store_id, threshold)
    return LowStockOutput(
        items=[LowStockItem(**i) for i in items],
        count=len(items),
        fetched_at=datetime.utcnow().isoformat()
    )
