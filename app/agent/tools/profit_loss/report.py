"""app/agent/tools/profit_loss/report.py"""
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service.profit_loss.report import fetch_profit_loss_report


class ProfitLossInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    days_lookback: int = Field(30, description="Number of days to look back for the P&L report.")


class ProfitLossOutput(BaseModel):
    revenue: float
    total_discount: float
    cogs: float                # Cost of Goods Sold
    gross_profit: float
    gross_margin_pct: float
    total_expenses: float
    net_profit: float
    net_margin_pct: float
    total_orders: int
    total_units_sold: int
    days_covered: int
    is_profitable: bool
    fetched_at: str


@tool("get_profit_loss_report", args_schema=ProfitLossInput)
async def get_profit_loss_report(store_id: str, days_lookback: int = 30) -> ProfitLossOutput:
    """
    Get a complete Profit & Loss (P&L) report for the store.
    - Revenue = total sales
    - COGS = sum of (buying price × units sold) per sale item
    - Gross Profit = Revenue - COGS
    - Net Profit = Gross Profit - Operating Expenses (rent, salary, etc.)
    - Margin % is shown at both gross and net level.

    Use this when the owner asks about profit, loss, net income, whether the store
    made money, margins, or needs a financial summary.
    Supports `days_lookback` (e.g. 7, 30, 90) for flexible time windows.
    """
    data = await fetch_profit_loss_report(store_id, days_lookback=days_lookback)
    return ProfitLossOutput(
        **data,
        fetched_at=datetime.utcnow().isoformat(),
    )
