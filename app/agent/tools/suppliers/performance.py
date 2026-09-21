from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import fetch_supplier_performance

class SupplierPerformanceInput(BaseModel):
    supplier_id: str = Field(..., description="The ID of the supplier.")

class SupplierPerformanceOutput(BaseModel):
    supplier_id: str
    on_time_delivery_rate: float
    defect_rate: float
    average_lead_time_days: int
    fetched_at: str

@tool("get_supplier_performance", args_schema=SupplierPerformanceInput)
async def get_supplier_performance(supplier_id: str) -> SupplierPerformanceOutput:
    """
    Evaluate supplier reliability, delivery times, and defect rates.
    """
    data = await fetch_supplier_performance(supplier_id)
    return SupplierPerformanceOutput(
        **data,
        fetched_at=datetime.utcnow().isoformat()
    )
