"""app/agent/tools/expenses/summary.py"""
from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service.expenses.summary import fetch_expense_summary


class ExpenseSummaryInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    days_lookback: int = Field(30, description="Number of days to look back for expense data.")


class ExpenseCategoryItem(BaseModel):
    category: str
    total_amount: float
    count: int


class ExpenseSummaryOutput(BaseModel):
    total_expenses: float
    expense_count: int
    categories: List[ExpenseCategoryItem]
    days_covered: int
    fetched_at: str


@tool("get_expense_summary", args_schema=ExpenseSummaryInput)
async def get_expense_summary(store_id: str, days_lookback: int = 30) -> ExpenseSummaryOutput:
    """
    Get a breakdown of all store expenses (Rent, Electricity, Salary, Transport,
    Maintenance, Marketing, etc.) over a given period. Returns total expense amount
    and a per-category breakdown. Use when the owner asks about operating costs,
    overheads, or how much was spent on a specific expense category.
    """
    data = await fetch_expense_summary(store_id, days_lookback=days_lookback)
    return ExpenseSummaryOutput(
        total_expenses=data.get("total_expenses", 0.0),
        expense_count=data.get("expense_count", 0),
        categories=[ExpenseCategoryItem(**c) for c in data.get("categories", [])],
        days_covered=data.get("days_covered", days_lookback),
        fetched_at=datetime.utcnow().isoformat(),
    )
