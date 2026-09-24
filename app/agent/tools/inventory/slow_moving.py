"""app/agent/tools/inventory/slow_moving.py"""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service.inventory.slow_moving import fetch_slow_moving_products


class SlowMovingInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    lookback_days: int = Field(30, description="Days to look back for sales activity (default 30).")


class SlowMovingItem(BaseModel):
    product_id: str
    name: str
    category: str
    quantity: int
    tied_up_value: float   # quantity * price — capital locked in unsold stock
    lookback_days: int


class SlowMovingOutput(BaseModel):
    products: List[SlowMovingItem]
    total_count: int
    total_tied_up_value: float
    fetched_at: str


@tool("get_slow_moving_products", args_schema=SlowMovingInput)
async def get_slow_moving_products(store_id: str, lookback_days: int = 30) -> SlowMovingOutput:
    """
    Get products that have stock (quantity > 0) but ZERO sales in the last
    lookback_days days. These are dormant products tying up working capital.
    Different from dead stock (which has 0 quantity). Sorted by tied-up value desc.
    Use to recommend clearance pricing or promotional strategies.
    """
    data = await fetch_slow_moving_products(store_id, lookback_days)
    total_value = round(sum(p["tied_up_value"] for p in data), 2)
    return SlowMovingOutput(
        products=[SlowMovingItem(**i) for i in data],
        total_count=len(data),
        total_tied_up_value=total_value,
        fetched_at=datetime.utcnow().isoformat(),
    )
