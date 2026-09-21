from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import fetch_product_details

class ProductDetailsInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    product_id: str = Field(..., description="The ID of the product.")

class ProductDetailsOutput(BaseModel):
    product_id: str
    name: str
    category: str
    description: Optional[str]
    price: float
    sku: Optional[str]
    fetched_at: str

@tool("get_product_details", args_schema=ProductDetailsInput)
async def get_product_details(store_id: str, product_id: str) -> ProductDetailsOutput:
    """
    Get full details and metadata for a single product.
    """
    data = await fetch_product_details(store_id, product_id)
    return ProductDetailsOutput(
        **data,
        fetched_at=datetime.utcnow().isoformat()
    )
