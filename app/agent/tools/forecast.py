from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

from app.agent.service import fetch_restock_priorities, fetch_forecast_summary

# --- Schemas ---

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


class ForecastSummaryInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    horizon_days: int = Field(7, description="Number of days to forecast into the future.")

class ForecastItem(BaseModel):
    product_id: str
    name: str
    current_stock: float
    predicted_demand: float
    predicted_daily: float
    trend_percent: float
    days_to_stockout: Optional[float] = None
    horizon_days: int

class ForecastSummaryOutput(BaseModel):
    horizon_days: int
    items: List[ForecastItem]
    fetched_at: str


# --- Tools ---

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


@tool("get_forecast_summary", args_schema=ForecastSummaryInput)
async def get_forecast_summary(store_id: str, horizon_days: int = 7) -> ForecastSummaryOutput:
    """
    Get a demand forecast summary for all products in the store over the given horizon.
    Includes predicted demand, current stock, and days-to-stockout estimates.
    """
    items = await fetch_forecast_summary(store_id, horizon_days)
    return ForecastSummaryOutput(
        horizon_days=horizon_days,
        items=[ForecastItem(**i) for i in items],
        fetched_at=datetime.utcnow().isoformat()
    )
