"""app/agent/tools/purchases/summary.py"""
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service.purchases.summary import fetch_purchase_summary


class PurchaseSummaryInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    days_lookback: int = Field(30, description="Number of days to look back for purchase data.")


class PurchaseSummaryOutput(BaseModel):
    total_spend: float
    total_paid: float
    total_due: float
    order_count: int
    average_order_value: float
    paid_count: int
    partial_count: int
    unpaid_count: int
    days_covered: int
    fetched_at: str


@tool("get_purchase_summary", args_schema=PurchaseSummaryInput)
async def get_purchase_summary(store_id: str, days_lookback: int = 30) -> PurchaseSummaryOutput:
    """
    Get a summary of all purchase orders made by the store from suppliers/sellers
    over a given period. Includes total spend, payment status breakdown, and
    outstanding dues to sellers. Use when the owner asks about buying costs,
    how much was spent on restocking, or pending payments to suppliers.
    """
    data = await fetch_purchase_summary(store_id, days_lookback=days_lookback)
    return PurchaseSummaryOutput(
        **data,
        fetched_at=datetime.utcnow().isoformat(),
    )
