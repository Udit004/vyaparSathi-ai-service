from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

# Mock service function
async def mock_fetch_supplier_performance(supplier_id: str) -> dict:
    return {
        "supplier_id": supplier_id,
        "on_time_delivery_rate": 0.95,
        "defect_rate": 0.02,
        "average_lead_time_days": 14
    }

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
    data = await mock_fetch_supplier_performance(supplier_id)
    return SupplierPerformanceOutput(
        **data,
        fetched_at=datetime.utcnow().isoformat()
    )
