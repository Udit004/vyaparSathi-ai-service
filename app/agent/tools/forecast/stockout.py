from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import fetch_stockout_estimate

class StockoutInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    product_id: str = Field(..., description="The ID of the product.")

class StockoutOutput(BaseModel):
    product_id: str
    estimated_stockout_date: str
    days_remaining: int
    fetched_at: str

@tool("get_stockout_estimate", args_schema=StockoutInput)
async def get_stockout_estimate(store_id: str, product_id: str) -> StockoutOutput:
    """
    Estimate when a specific product will run out of stock based on current sales velocity.
    """
    data = await fetch_stockout_estimate(store_id, product_id)
    return StockoutOutput(
        **data,
        fetched_at=datetime.utcnow().isoformat()
    )
