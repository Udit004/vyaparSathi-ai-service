from typing import Optional, Any
from pydantic import BaseModel, Field, model_validator
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import fetch_product_details

class ProductDetailsInput(BaseModel):
    product_id: str = Field(..., description="The ID, name, or SKU of the product.")
    store_id: Optional[str] = Field(default=None, description="The ID of the store.")

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "product_id" not in data or not data.get("product_id"):
                for alias in ("product_name", "name", "id", "product", "item_name", "sku", "query"):
                    if alias in data and data.get(alias):
                        data["product_id"] = str(data[alias])
                        break
        return data

class ProductDetailsOutput(BaseModel):
    product_id: str
    name: str
    category: str
    description: Optional[str]
    price: float
    sku: Optional[str]
    fetched_at: str

@tool("get_product_details", args_schema=ProductDetailsInput)
async def get_product_details(product_id: str, store_id: Optional[str] = None) -> ProductDetailsOutput:
    """
    Get full details and metadata for a single product.
    """
    data = await fetch_product_details(store_id or "", product_id)
    return ProductDetailsOutput(
        **data,
        fetched_at=datetime.utcnow().isoformat()
    )
