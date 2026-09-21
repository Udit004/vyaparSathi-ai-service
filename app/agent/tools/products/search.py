from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import search_products as fetch_search_products

class ProductSearchInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    query: str = Field(..., description="Search query string.")

class ProductSearchResult(BaseModel):
    product_id: str
    name: str
    category: str
    price: float

class ProductSearchOutput(BaseModel):
    results: List[ProductSearchResult]
    fetched_at: str

@tool("search_products", args_schema=ProductSearchInput)
async def search_products(store_id: str, query: str) -> ProductSearchOutput:
    """
    Search for products in the store by name or query.
    """
    items = await fetch_search_products(store_id, query)
    return ProductSearchOutput(
        results=[ProductSearchResult(**i) for i in items],
        fetched_at=datetime.utcnow().isoformat()
    )
