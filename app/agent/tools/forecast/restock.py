from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

from app.agent.service import fetch_restock_priorities

class RestockPriorityInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")

class RestockItem(BaseModel):
    product_id: str
    name: str
    current_quantity: float
    avg_daily_sales: float
    days_to_stockout: Optional[float] = None
    suggested_restock_quantity: float
    priority: Literal["RED", "YELLOW", "GREEN"]

class RestockPriorityOutput(BaseModel):
    priorities: List[RestockItem]
    red_count: int
    yellow_count: int
    fetched_at: str

@tool("get_restock_priorities", args_schema=RestockPriorityInput)
async def get_restock_priorities(store_id: str) -> RestockPriorityOutput:
    """
    Get a prioritized list of products that need restocking based on real sales velocity.
    RED means urgent (out of stock or days_to_stockout <= lead_time),
    YELLOW means restock soon, GREEN means healthy.
    Includes avg daily sales and days-to-stockout estimates.
    """
    items = await fetch_restock_priorities(store_id)
    priorities = [RestockItem(**i) for i in items]
    return RestockPriorityOutput(
        priorities=priorities,
        red_count=sum(1 for p in priorities if p.priority == "RED"),
        yellow_count=sum(1 for p in priorities if p.priority == "YELLOW"),
        fetched_at=datetime.utcnow().isoformat()
    )
