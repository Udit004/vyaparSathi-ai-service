"""
app/agent/tools/purchases/search.py
===================================
Search and inspect purchase orders / invoices by seller, invoice number, status, or date range.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import structlog

from app.agent.service.purchases.search import fetch_purchases

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.purchases.search")


class PurchaseSearchInput(BaseModel):
    query: str = Field(
        default="",
        description="Search query by invoice number, bill number, or notes.",
    )
    seller_name: Optional[str] = Field(
        default=None,
        description="Filter purchases by seller/distributor name (e.g. 'Global Traders').",
    )
    payment_status: Optional[str] = Field(
        default=None,
        description="Filter by payment status: 'paid', 'unpaid', or 'partial'.",
    )
    days_lookback: int = Field(
        default=60,
        ge=1,
        le=365,
        description="Number of days to search back (default 60 days).",
    )
    limit: int = Field(default=20, ge=1, le=50, description="Max purchases to return.")
    store_id: Optional[str] = Field(default=None, description="The store ID.")
    user_id: Optional[str] = Field(default=None, description="The store owner / user ID.")


class PurchaseItemSummary(BaseModel):
    product_name: str
    quantity: float
    purchase_price: float
    total_cost: float


class PurchaseOrderRecord(BaseModel):
    purchase_id: str
    invoice_number: str
    seller_name: str
    purchase_date: str
    total_amount: float
    paid_amount: float
    due_amount: float
    payment_status: str
    items_count: int
    items: List[PurchaseItemSummary]


class PurchaseSearchOutput(BaseModel):
    purchases: List[PurchaseOrderRecord]
    count: int
    found: bool
    total_spend: float
    total_due: float
    summary: str
    fetched_at: str


@tool("search_purchases", args_schema=PurchaseSearchInput)
async def search_purchases(
    query: str = "",
    seller_name: Optional[str] = None,
    payment_status: Optional[str] = None,
    days_lookback: int = 60,
    limit: int = 20,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Search and retrieve purchase orders, distributor bills, and supplier invoices.
    Supports filtering by distributor/seller name, invoice number, payment status (paid/unpaid/partial), and date.

    Use this when:
    - The merchant asks "Show me past orders from Global Traders", "Purchases summary for this month", "Find invoice #1024", "Unpaid purchase orders".
    - Checking supplier order history, bill amounts, and pending payables.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    LOGGER.info("search_purchases_invoked", store_id=store_id, seller=seller_name, query=query)

    if not store_id:
        return {
            "purchases": [],
            "count": 0,
            "found": False,
            "total_spend": 0.0,
            "total_due": 0.0,
            "summary": "Store ID missing from runtime context.",
            "fetched_at": datetime.utcnow().isoformat(),
        }

    raw_purchases = await fetch_purchases(
        store_id=store_id,
        query=query,
        seller_name=seller_name or "",
        payment_status=payment_status or "",
        days_lookback=days_lookback,
        limit=limit,
    )

    records = [
        PurchaseOrderRecord(
            purchase_id=p["purchase_id"],
            invoice_number=p["invoice_number"],
            seller_name=p["seller_name"],
            purchase_date=p["purchase_date"],
            total_amount=p["total_amount"],
            paid_amount=p["paid_amount"],
            due_amount=p["due_amount"],
            payment_status=p["payment_status"],
            items_count=p["items_count"],
            items=[PurchaseItemSummary(**it) for it in p.get("items", [])],
        )
        for p in raw_purchases
    ]

    found = len(records) > 0
    total_spend = sum(r.total_amount for r in records)
    total_due = sum(r.due_amount for r in records)

    if found:
        summary = f"Found {len(records)} purchase order(s) totaling ₹{total_spend:,.2f} (Outstanding Due: ₹{total_due:,.2f})."
    else:
        summary = "No matching purchase orders found."

    return {
        "purchases": [r.model_dump() for r in records],
        "count": len(records),
        "found": found,
        "total_spend": total_spend,
        "total_due": total_due,
        "summary": summary,
        "fetched_at": datetime.utcnow().isoformat(),
    }
