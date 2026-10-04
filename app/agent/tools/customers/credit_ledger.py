"""
Tool for querying customer credit balances (Udhar / Khata) and identifying high-risk pending dues.
Supported for both Text Copilot and Voice Agent.
"""
from __future__ import annotations

from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import structlog

from app.agent.service.customers.credit_ledger import get_customer_credit_ledger as fetch_credit_ledger

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.credit_ledger")


class CreditLedgerInput(BaseModel):
    user_id: Optional[str] = Field(
        default=None,
        description="ID of the merchant user. Injected automatically if available.",
    )
    store_id: Optional[str] = Field(
        default=None,
        description="ID of the store. Injected automatically if available.",
    )
    min_due_days: Optional[int] = Field(
        default=0,
        description="Minimum number of days credit has been pending (e.g. 30 to show dues > 30 days).",
    )


@tool("get_customer_credit_ledger", args_schema=CreditLedgerInput)
async def get_customer_credit_ledger(
    user_id: Optional[str] = None,
    store_id: Optional[str] = None,
    min_due_days: Optional[int] = 0,
    **kwargs,
) -> dict:
    """
    Fetch pending customer credit balances (Udhar / Khata), customer contact details, and payment due timelines.
    Use this tool to advise merchants on overdue balances or draft payment reminder messages.
    """
    effective_user_id = user_id or kwargs.get("configurable", {}).get("user_id") or kwargs.get("user_id")
    effective_store_id = store_id or kwargs.get("configurable", {}).get("store_id") or kwargs.get("store_id")

    if not effective_user_id:
        return {"error": "Merchant user_id is required to fetch credit ledger."}

    ledger = await fetch_credit_ledger(
        user_id=effective_user_id,
        store_id=effective_store_id,
        min_due_days=min_due_days or 0
    )

    total_pending = sum(item.get("outstanding_balance", 0.0) for item in ledger)

    return {
        "status": "success",
        "customers_count": len(ledger),
        "total_outstanding_credit": round(total_pending, 2),
        "credit_ledger": ledger
    }
