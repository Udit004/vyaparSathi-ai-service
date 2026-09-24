"""app/agent/tools/sales/daily_trend.py"""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service.sales.daily_trend import fetch_daily_sales_trend


class DailySalesTrendInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    days: int = Field(30, description="Number of past days to include in trend (default 30).")


class DailySalesTrendOutput(BaseModel):
    days: List[str]           # date strings YYYY-MM-DD
    revenue: List[float]
    transactions: List[int]
    peak_day: dict            # {date, revenue}
    avg_daily_revenue: float
    trend: str                # "up" | "down" | "flat"
    lookback_days: int
    fetched_at: str


@tool("get_daily_sales_trend", args_schema=DailySalesTrendInput)
async def get_daily_sales_trend(store_id: str, days: int = 30) -> DailySalesTrendOutput:
    """
    Get day-by-day sales revenue and transaction count for the last N days.
    Returns a trend direction (up/down/flat), peak revenue day, and average
    daily revenue. Use when the owner asks about sales trends, growth, or
    specific day-level performance. Data is suitable for chart rendering.
    """
    data = await fetch_daily_sales_trend(store_id, days)
    return DailySalesTrendOutput(
        days=data.get("days", []),
        revenue=data.get("revenue", []),
        transactions=data.get("transactions", []),
        peak_day=data.get("peak_day", {}),
        avg_daily_revenue=data.get("avg_daily_revenue", 0.0),
        trend=data.get("trend", "flat"),
        lookback_days=data.get("lookback_days", days),
        fetched_at=datetime.utcnow().isoformat(),
    )
