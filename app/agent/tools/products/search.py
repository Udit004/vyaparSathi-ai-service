from typing import List, Optional, Any
from pydantic import BaseModel, Field, model_validator
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service import search_products as fetch_search_products

class ProductSearchInput(BaseModel):
    query: str = Field(default="", description="Search query string or product name.")
    store_id: Optional[str] = Field(default=None, description="The ID of the store.")

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "query" not in data or not data.get("query"):
                for alias in ("product_name", "name", "search", "search_query", "product", "item_name", "q", "title"):
                    if alias in data and data.get(alias):
                        data["query"] = str(data[alias])
                        break
        return data

class ProductSearchResult(BaseModel):
    product_id: str
    name: str
    category: str
    price: float

class ProductSearchOutput(BaseModel):
    results: List[ProductSearchResult]
    fetched_at: str

@tool("search_products", args_schema=ProductSearchInput)
async def search_products(query: str = "", store_id: Optional[str] = None) -> ProductSearchOutput:
    """
    Search for products in the store by name or query.
    """
    items = await fetch_search_products(store_id or "", query)
    return ProductSearchOutput(
        results=[ProductSearchResult(**i) for i in items],
        fetched_at=datetime.utcnow().isoformat()
    )
