from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

# Mock service function
async def mock_fetch_inventory_risk(store_id: str) -> list:
    return [
        {"product_id": "prod-1", "name": "Laptop", "risk_level": "HIGH", "risk_type": "STOCKOUT", "description": "High demand, low stock"},
        {"product_id": "prod-2", "name": "Fidget Spinner", "risk_level": "MEDIUM", "risk_type": "OVERSTOCK", "description": "Low demand, high stock"}
    ]

class InventoryRiskInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")

class RiskItem(BaseModel):
    product_id: str
    name: str
    risk_level: str
    risk_type: str
    description: str

class InventoryRiskOutput(BaseModel):
    risks: List[RiskItem]
    fetched_at: str

@tool("get_inventory_risk", args_schema=InventoryRiskInput)
async def get_inventory_risk(store_id: str) -> InventoryRiskOutput:
    """
    Assess inventory risks such as impending stockouts or severe overstocking for the store.
    """
    items = await mock_fetch_inventory_risk(store_id)
    return InventoryRiskOutput(
        risks=[RiskItem(**i) for i in items],
        fetched_at=datetime.utcnow().isoformat()
    )
