from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

# Mock service function
async def mock_compare_stores(store_ids: List[str]) -> list:
    return [
        {"store_id": "store-1", "revenue": 10000, "profit_margin": 0.2},
        {"store_id": "store-2", "revenue": 12000, "profit_margin": 0.25}
    ]

class StoreComparisonInput(BaseModel):
    store_ids: List[str] = Field(..., description="List of store IDs to compare.")

class ComparisonMetric(BaseModel):
    store_id: str
    revenue: float
    profit_margin: float

class StoreComparisonOutput(BaseModel):
    comparisons: List[ComparisonMetric]
    fetched_at: str

@tool("compare_stores", args_schema=StoreComparisonInput)
async def compare_stores(store_ids: List[str]) -> StoreComparisonOutput:
    """
    Compare key metrics between multiple stores.
    """
    items = await mock_compare_stores(store_ids)
    return StoreComparisonOutput(
        comparisons=[ComparisonMetric(**i) for i in items],
        fetched_at=datetime.utcnow().isoformat()
    )
