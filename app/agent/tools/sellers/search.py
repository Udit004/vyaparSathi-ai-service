"""
app/agent/tools/sellers/search.py
=================================
Search and lookup sellers / distributors / vendors connected to the store.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import structlog

from app.agent.service.sellers.search import fetch_sellers

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.sellers.search")


class SellerSearchInput(BaseModel):
    query: str = Field(
        default="",
        description="Search query to filter sellers by contact name, business name, phone, email, or GSTIN (e.g. 'Global Traders', 'Udit').",
    )
    status: str = Field(
        default="active",
        description="Filter by seller status: 'active', 'inactive', or 'all'.",
    )
    limit: int = Field(default=20, ge=1, le=50, description="Max sellers to return.")
    store_id: Optional[str] = Field(default=None, description="The store ID.")
    user_id: Optional[str] = Field(default=None, description="The store owner / user ID.")


class SellerItem(BaseModel):
    seller_id: str
    name: str
    business_name: str
    phone: str
    email: Optional[str] = None
    address: Optional[str] = None
    gstin: Optional[str] = None
    total_purchase: float
    total_paid: float
    total_due: float
    status: str


class SellerSearchOutput(BaseModel):
    sellers: List[SellerItem]
    count: int
    found: bool
    summary: str
    fetched_at: str


@tool("search_sellers", args_schema=SellerSearchInput)
async def search_sellers(
    query: str = "",
    status: str = "active",
    limit: int = 20,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Search for sellers, vendors, or wholesale distributors connected to your store.
    Returns seller contact info, business name, phone number, GSTIN, and current purchase/due balance.

    Use this when:
    - The merchant asks "Kaun kaun se sellers connected hain mere se?", "Who are my distributors?", "Search for seller Global Traders / Udit Tiwari".
    - Checking vendor contact details, pending payments, or supplier list.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    LOGGER.info("search_sellers_invoked", store_id=store_id, query=query)

    if not store_id:
        return {
            "sellers": [],
            "count": 0,
            "found": False,
            "summary": "Store ID missing from runtime context.",
            "fetched_at": datetime.utcnow().isoformat(),
        }

    raw_sellers = await fetch_sellers(store_id=store_id, query=query, status=status, limit=limit)
    sellers = [SellerItem(**s) for s in raw_sellers]
    found = len(sellers) > 0

    if found:
        names_list = ", ".join([s.business_name or s.name for s in sellers[:5]])
        summary = f"Found {len(sellers)} seller(s): {names_list}."
    else:
        summary = f"No sellers found matching '{query}' for this store." if query else "No sellers are currently connected to this store."

    return {
        "sellers": [s.model_dump() for s in sellers],
        "count": len(sellers),
        "found": found,
        "summary": summary,
        "fetched_at": datetime.utcnow().isoformat(),
    }
