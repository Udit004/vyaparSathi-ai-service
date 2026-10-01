"""app/agent/tools/buyers/dues.py"""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service.buyers.dues import fetch_buyer_dues


class BuyerDuesInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    min_due: float = Field(0.01, description="Minimum due amount to include.")
    limit: int = Field(20, description="Maximum number of buyers to return.")


class BuyerDueItem(BaseModel):
    buyer_id: str
    name: str
    phone: str
    total_sales: float
    total_paid: float
    total_due: float


class BuyerDuesOutput(BaseModel):
    buyers: List[BuyerDueItem]
    total_outstanding: float
    buyers_with_dues: int
    fetched_at: str


@tool("get_buyer_dues", args_schema=BuyerDuesInput)
async def get_buyer_dues(store_id: str, min_due: float = 0.01, limit: int = 20) -> BuyerDuesOutput:
    """
    Get a list of buyers with outstanding dues (unpaid amounts). Returns buyers
    sorted by highest due amount first, with store-level total outstanding.
    Use when the owner asks about pending payments, who owes money, or recovery of dues.
    """
    data = await fetch_buyer_dues(store_id, min_due=min_due, limit=limit)
    return BuyerDuesOutput(
        buyers=[BuyerDueItem(**b) for b in data.get("buyers", [])],
        total_outstanding=data.get("total_outstanding", 0.0),
        buyers_with_dues=data.get("buyers_with_dues", 0),
        fetched_at=datetime.utcnow().isoformat(),
    )
