from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service.stores.summary import fetch_store_summary


class StoreSummaryInput(BaseModel):
    store_id: str = Field(..., description="The ID or name of the store.")


class StoreSummaryOutput(BaseModel):
    store_id: str
    name: str
    owner_name: Optional[str] = None
    owner_email: Optional[str] = None
    location: Optional[str] = None
    phone: Optional[str] = None
    business_type: Optional[str] = None
    total_products: Optional[int] = 0
    low_stock_count: Optional[int] = 0
    status: str
    fetched_at: str


@tool("get_store_summary", args_schema=StoreSummaryInput)
async def get_store_summary(store_id: str) -> StoreSummaryOutput:
    """
    Get detailed information for a store, including store name, merchant/owner name,
    official contact email, phone, location, business category, and catalog product counts.
    """
    data = await fetch_store_summary(store_id)
    return StoreSummaryOutput(
        **data,
        fetched_at=datetime.utcnow().isoformat()
    )
