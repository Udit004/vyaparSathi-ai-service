from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import fetch_stock_history

class StockHistoryInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    product_id: str = Field(..., description="The ID of the product.")
    days: int = Field(30, description="Number of days of history to fetch.")

class StockHistoryEntry(BaseModel):
    date: str
    quantity: float

class StockHistoryOutput(BaseModel):
    product_id: str
    history: List[StockHistoryEntry]
    fetched_at: str

@tool("get_stock_history", args_schema=StockHistoryInput)
async def get_stock_history(store_id: str, product_id: str, days: int = 30) -> StockHistoryOutput:
    """
    Fetch the historical stock levels for a specific product over a given number of days.
    Useful for tracking how stock depletes over time.
    """
    history_data = await fetch_stock_history(store_id, product_id, days)
    return StockHistoryOutput(
        product_id=product_id,
        history=[StockHistoryEntry(**h) for h in history_data],
        fetched_at=datetime.utcnow().isoformat()
    )
