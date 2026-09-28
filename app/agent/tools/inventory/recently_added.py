"""app/agent/tools/inventory/recently_added.py"""
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service.inventory.recently_added import fetch_recently_added_products


class RecentlyAddedInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    lookback_days: int = Field(7, description="Number of days to look back for recently added products (default 7).")


class RecentlyAddedItem(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    quantity: int
    sku: str
    created_at: Optional[str] = None
    days_ago: Optional[float] = None


class RecentlyAddedOutput(BaseModel):
    products: List[RecentlyAddedItem]
    total_count: int
    lookback_days: int
    fetched_at: str


@tool("get_recently_added_products", args_schema=RecentlyAddedInput)
async def get_recently_added_products(store_id: str, lookback_days: int = 7) -> RecentlyAddedOutput:
    """
    Get products that were added to the inventory within the last `lookback_days`
    days (default 7). Returns newly created products sorted by creation date
    (newest first), along with their SKU, price, quantity, category, and how
    many days ago they were added.
    Use this to track new inventory arrivals, verify recently stocked items,
    or audit product onboarding activity.
    """
    data = await fetch_recently_added_products(store_id, lookback_days)
    return RecentlyAddedOutput(
        products=[RecentlyAddedItem(**i) for i in data],
        total_count=len(data),
        lookback_days=lookback_days,
        fetched_at=datetime.utcnow().isoformat(),
    )
