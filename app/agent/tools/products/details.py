from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

# Mock service function
async def mock_fetch_product_details(store_id: str, product_id: str) -> dict:
    return {
        "product_id": product_id,
        "name": "Apple",
        "category": "Fruits",
        "description": "A red apple",
        "price": 1.5,
        "sku": "APP-001"
    }

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
    data = await mock_fetch_product_details(store_id, product_id)
    return ProductDetailsOutput(
        **data,
        fetched_at=datetime.utcnow().isoformat()
    )
