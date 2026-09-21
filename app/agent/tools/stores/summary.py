from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime

# Mock service function
async def mock_fetch_store_summary(store_id: str) -> dict:
    return {
        "store_id": store_id,
        "name": "Main Store",
        "location": "New York",
        "status": "ACTIVE"
    }

class StoreSummaryInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")

class StoreSummaryOutput(BaseModel):
    store_id: str
    name: str
    location: Optional[str]
    status: str
    fetched_at: str

@tool("get_store_summary", args_schema=StoreSummaryInput)
async def get_store_summary(store_id: str) -> StoreSummaryOutput:
    """
    Get general information and high-level stats for a store.
    """
    data = await mock_fetch_store_summary(store_id)
    return StoreSummaryOutput(
        **data,
        fetched_at=datetime.utcnow().isoformat()
    )
