from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import fetch_demand

class DemandForecastInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    product_id: str = Field(..., description="The ID of the product.")
    horizon_days: int = Field(7, description="Days to forecast.")

class DemandForecastOutput(BaseModel):
    product_id: str
    predicted_demand: float
    confidence_score: float
    fetched_at: str

@tool("get_demand_forecast", args_schema=DemandForecastInput)
async def get_demand_forecast(store_id: str, product_id: str, horizon_days: int = 7) -> DemandForecastOutput:
    """
    Predict future demand for a specific product over a given horizon.
    """
    data = await fetch_demand(store_id, product_id, horizon_days)
    return DemandForecastOutput(
        **data,
        fetched_at=datetime.utcnow().isoformat()
    )
