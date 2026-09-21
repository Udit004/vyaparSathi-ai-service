from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import fetch_inventory_risk

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
    items = await fetch_inventory_risk(store_id)
    return InventoryRiskOutput(
        risks=[RiskItem(**i) for i in items],
        fetched_at=datetime.utcnow().isoformat()
    )
