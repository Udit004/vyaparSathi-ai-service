from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import fetch_supplier_pricing

class SupplierPricingInput(BaseModel):
    supplier_id: str = Field(..., description="The ID of the supplier.")
    product_id: Optional[str] = Field(None, description="Optional specific product ID to check.")

class PricingItem(BaseModel):
    product_id: str
    unit_price: float
    bulk_discount_threshold: Optional[int]
    bulk_price: Optional[float]

class SupplierPricingOutput(BaseModel):
    pricing: List[PricingItem]
    fetched_at: str

@tool("get_supplier_pricing", args_schema=SupplierPricingInput)
async def get_supplier_pricing(supplier_id: str, product_id: Optional[str] = None) -> SupplierPricingOutput:
    """
    Get supplier pricing and bulk discount information for products.
    """
    items = await fetch_supplier_pricing(supplier_id, product_id)
    return SupplierPricingOutput(
        pricing=[PricingItem(**i) for i in items],
        fetched_at=datetime.utcnow().isoformat()
    )
