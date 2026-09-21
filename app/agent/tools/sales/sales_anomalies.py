from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

# Mock service function
async def mock_fetch_sales_anomalies(store_id: str) -> list:
    return [
        {"product_id": "prod-x", "anomaly_type": "SPIKE", "description": "Sales spiked by 300% today."}
    ]

class SalesAnomaliesInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")

class AnomalyItem(BaseModel):
    product_id: str
    anomaly_type: str
    description: str

class SalesAnomaliesOutput(BaseModel):
    anomalies: List[AnomalyItem]
    fetched_at: str

@tool("get_sales_anomalies", args_schema=SalesAnomaliesInput)
async def get_sales_anomalies(store_id: str) -> SalesAnomaliesOutput:
    """
    Detect unusual sales spikes or drops in the store.
    """
    items = await mock_fetch_sales_anomalies(store_id)
    return SalesAnomaliesOutput(
        anomalies=[AnomalyItem(**i) for i in items],
        fetched_at=datetime.utcnow().isoformat()
    )
