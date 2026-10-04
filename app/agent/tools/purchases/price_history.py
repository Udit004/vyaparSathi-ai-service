"""
Tool for checking supplier price trends and detecting wholesale price hikes.
Supported for both Text Copilot and Voice Agent.
"""
from __future__ import annotations

from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import structlog

from app.agent.service.purchases.price_history import get_supplier_price_trends

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.price_history")


class SupplierPriceTrendInput(BaseModel):
    user_id: Optional[str] = Field(
        default=None,
        description="ID of the merchant user. Injected automatically if available.",
    )
    product_name: Optional[str] = Field(
        default=None,
        description="Optional product name to filter price trends (e.g. 'Basmati Rice', 'Fortune Oil').",
    )
    supplier_name: Optional[str] = Field(
        default=None,
        description="Optional supplier/distributor name to filter price history.",
    )


@tool("check_supplier_price_trends", args_schema=SupplierPriceTrendInput)
async def check_supplier_price_trends(
    user_id: Optional[str] = None,
    product_name: Optional[str] = None,
    supplier_name: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Check historical supplier wholesale prices, price changes, and inflation trends across purchase bills.
    Use this tool to compare wholesale purchase costs or verify if a supplier increased prices recently.
    """
    effective_user_id = user_id or kwargs.get("configurable", {}).get("user_id") or kwargs.get("user_id")

    if not effective_user_id:
        return {"error": "Merchant user_id is required to check supplier price trends."}

    trends = await get_supplier_price_trends(
        user_id=effective_user_id,
        product_name=product_name,
        supplier_name=supplier_name
    )

    return {
        "status": "success",
        "records_found": len(trends),
        "price_history": trends
    }
