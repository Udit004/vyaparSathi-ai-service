"""
app/agent/tools/expenses/write.py
===================================
LangChain tools for writing expenses.
"""
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app.agent.service.expenses.write import create_expense, update_expense, delete_expense

class CreateExpenseSchema(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    title: str = Field(..., description="Expense title or reason")
    category: str = Field(..., description="Expense category (e.g. 'Rent', 'Salary', 'Utility')")
    amount: float = Field(..., description="Expense amount")
    payment_method: Optional[str] = Field("Cash", description="Payment method (Cash, UPI, Bank)")
    description: Optional[str] = Field(None, description="Detailed description")
    date: Optional[str] = Field(None, description="Expense date in ISO format (YYYY-MM-DD)")

@tool(args_schema=CreateExpenseSchema)
async def tool_create_expense(store_id: str, title: str, category: str, amount: float, payment_method: str = "Cash", description: str = None, date: str = None) -> str:
    """Record a new business expense."""
    if not store_id: return "Error: store_id is required"
        
    data = {"title": title, "category": category, "amount": amount, "payment_method": payment_method, "description": description, "date": date}
    try:
        expense_id = await create_expense(store_id, data)
        return f"Successfully created expense '{title}' for amount {amount}. ID: {expense_id}"
    except Exception as e:
        return f"Failed to create expense: {str(e)}"

class UpdateExpenseSchema(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    expense_id: str = Field(..., description="The ID of the expense to update")
    title: Optional[str] = Field(None, description="Expense title")
    amount: Optional[float] = Field(None, description="Expense amount")

@tool(args_schema=UpdateExpenseSchema)
async def tool_update_expense(store_id: str, expense_id: str, title: str = None, amount: float = None) -> str:
    """Update fields of an existing expense."""
    if not store_id: return "Error: store_id is required"
        
    data = {}
    if title is not None: data["title"] = title
    if amount is not None: data["amount"] = amount
    
    try:
        await update_expense(store_id, expense_id, data)
        return f"Successfully updated expense {expense_id}."
    except Exception as e:
        return f"Failed to update expense: {str(e)}"

class DeleteExpenseSchema(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    expense_id: str = Field(..., description="The ID of the expense to delete")

@tool(args_schema=DeleteExpenseSchema)
async def tool_delete_expense(store_id: str, expense_id: str) -> str:
    """Delete an expense record."""
    if not store_id: return "Error: store_id is required"
    try:
        await delete_expense(store_id, expense_id)
        return f"Successfully deleted expense {expense_id}."
    except Exception as e:
        return f"Failed to delete expense: {str(e)}"
