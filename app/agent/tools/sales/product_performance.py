from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

# Mock service function
async def mock_fetch_product_performance(store_id: str, product_id: str) -> dict:
    return {
        "product_id": product_id,
        "total_revenue": 1500.0,
        "units_sold": 150,
        "conversion_rate": 0.05
    }

class ProductPerformanceInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    product_id: str = Field(..., description="The ID of the product.")

class ProductPerformanceOutput(BaseModel):
    product_id: str
    total_revenue: float
    units_sold: int
    conversion_rate: float
    fetched_at: str

@tool("get_product_performance", args_schema=ProductPerformanceInput)
async def get_product_performance(store_id: str, product_id: str) -> ProductPerformanceOutput:
    """
    Get detailed sales performance metrics for a specific product.
    """
    data = await mock_fetch_product_performance(store_id, product_id)
    return ProductPerformanceOutput(
        **data,
        fetched_at=datetime.utcnow().isoformat()
    )
