from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import fetch_product_history

class ProductHistoryInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    product_id: str = Field(..., description="The ID of the product.")

class HistoryEvent(BaseModel):
    date: str
    event_type: str
    details: str

class ProductHistoryOutput(BaseModel):
    events: List[HistoryEvent]
    fetched_at: str

@tool("get_product_history", args_schema=ProductHistoryInput)
async def get_product_history(store_id: str, product_id: str) -> ProductHistoryOutput:
    """
    Get price or update history for a product.
    """
    items = await fetch_product_history(store_id, product_id)
    return ProductHistoryOutput(
        events=[HistoryEvent(**i) for i in items],
        fetched_at=datetime.utcnow().isoformat()
    )
