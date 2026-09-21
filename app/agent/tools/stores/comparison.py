from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import compare_stores as fetch_compare_stores

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
    items = await fetch_compare_stores(store_ids)
    return StoreComparisonOutput(
        comparisons=[ComparisonMetric(**i) for i in items],
        fetched_at=datetime.utcnow().isoformat()
    )
