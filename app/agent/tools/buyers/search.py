"""app/agent/tools/buyers/search.py"""
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service.buyers.search import fetch_buyers


class BuyerSearchInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    query: str = Field("", description="Optional search term to filter by buyer name or phone.")
    status: str = Field("active", description="Filter by buyer status: 'active' or 'inactive'.")
    limit: int = Field(20, description="Maximum number of buyers to return.")


class BuyerItem(BaseModel):
    buyer_id: str
    name: str
    phone: str
    email: Optional[str]
    address: Optional[str]
    gstin: Optional[str]
    total_sales: float
    total_paid: float
    total_due: float
    status: str


class BuyerSearchOutput(BaseModel):
    buyers: List[BuyerItem]
    total_count: int
    fetched_at: str


@tool("search_buyers", args_schema=BuyerSearchInput)
async def search_buyers(store_id: str, query: str = "", status: str = "active", limit: int = 20) -> BuyerSearchOutput:
    """
    Search and list buyers (customers) for the store. Returns contact details
    and financial summary (total sales, amount paid, amount due) for each buyer.
    Use when the owner asks about customers, who owes money, or to look up a specific buyer.
    """
    data = await fetch_buyers(store_id, query=query, status=status, limit=limit)
    return BuyerSearchOutput(
        buyers=[BuyerItem(**b) for b in data],
        total_count=len(data),
        fetched_at=datetime.utcnow().isoformat(),
    )
