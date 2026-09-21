from typing import List, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

from app.agent.service import fetch_store_insights

class StoreInsightsInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")

class InsightResult(BaseModel):
    type: str
    title: str
    summary: str
    severity: str
    affected_products: Optional[List[str]] = None
    affected_count: Optional[int] = None
    dead_stock_value: Optional[float] = None
    leaders: Optional[List[dict]] = None

class StoreInsightsOutput(BaseModel):
    insights: List[InsightResult]
    insight_count: int
    fetched_at: str

@tool("get_store_insights", args_schema=StoreInsightsInput)
async def get_store_insights(store_id: str) -> StoreInsightsOutput:
    """
    Get AI-driven operational insights for the store derived from real sales and inventory data.
    Includes: out-of-stock alerts, dead stock detection, and top revenue-generating products.
    Use this to understand the overall health of the store.
    """
    items = await fetch_store_insights(store_id)
    insights = [InsightResult(**i) for i in items]
    return StoreInsightsOutput(
        insights=insights,
        insight_count=len(insights),
        fetched_at=datetime.utcnow().isoformat()
    )
