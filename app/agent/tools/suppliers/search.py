from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import search_suppliers as fetch_search_suppliers

class SupplierSearchInput(BaseModel):
    query: str = Field(..., description="Search query for suppliers.")

class SupplierSearchResult(BaseModel):
    supplier_id: str
    name: str
    rating: float

class SupplierSearchOutput(BaseModel):
    results: List[SupplierSearchResult]
    fetched_at: str

@tool("search_suppliers", args_schema=SupplierSearchInput)
async def search_suppliers(query: str) -> SupplierSearchOutput:
    """
    Search for suppliers by name or category.
    """
    items = await fetch_search_suppliers(query)
    return SupplierSearchOutput(
        results=[SupplierSearchResult(**i) for i in items],
        fetched_at=datetime.utcnow().isoformat()
    )
