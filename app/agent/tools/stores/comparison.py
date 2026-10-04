"""
app/agent/tools/stores/comparison.py
======================================
Multi-Store Comparison Tool for Text Copilot & Realtime Voice Agent.
Compares sales revenue, catalog size, low-stock warnings, and vendor counts
across all stores registered under a merchant user account.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

from app.agent.service.stores.comparison import compare_user_stores


class StoreComparisonInput(BaseModel):
    user_id: Optional[str] = Field(
        default=None,
        description="The user's ID to compare all stores under their account. Automatically injected if omitted."
    )
    store_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of specific store IDs or store names to compare. If omitted, compares all stores owned by the merchant."
    )
    days_lookback: int = Field(
        default=30,
        description="Lookback period in days for sales and revenue comparison (default: 30 days)."
    )
    store_id: Optional[str] = Field(default=None, description="Current store ID context")

    class Config:
        extra = "ignore"


@tool("compare_stores", args_schema=StoreComparisonInput)
async def compare_stores(
    user_id: Optional[str] = None,
    store_ids: Optional[List[str]] = None,
    days_lookback: int = 30,
    store_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Compare performance metrics (sales revenue, order count, catalog size, low stock alerts, connected suppliers, customers)
    across all stores owned by the merchant user account.
    """
    target_user = user_id or kwargs.get("user_id") or "default_user"
    target_store = store_id or kwargs.get("store_id")
    return await compare_user_stores(
        user_id=target_user,
        store_ids=store_ids,
        days_lookback=days_lookback,
        current_store_id=target_store,
    )
