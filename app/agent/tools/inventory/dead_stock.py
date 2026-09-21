from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import fetch_dead_stock

class DeadStockInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    days_inactive: int = Field(90, description="Minimum number of days with no sales to be considered dead stock.")

class DeadStockItem(BaseModel):
    product_id: str
    name: str
    current_quantity: float
    last_sold_date: str

class DeadStockOutput(BaseModel):
    items: List[DeadStockItem]
    count: int
    fetched_at: str

@tool("get_dead_stock", args_schema=DeadStockInput)
async def get_dead_stock(store_id: str, days_inactive: int = 90) -> DeadStockOutput:
    """
    Identify dead stock - products that have not had any sales for a specified number of days.
    """
    items = await fetch_dead_stock(store_id, days_inactive)
    return DeadStockOutput(
        items=[DeadStockItem(**i) for i in items],
        count=len(items),
        fetched_at=datetime.utcnow().isoformat()
    )
