from __future__ import annotations
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.products.search")

async def search_products(store_id: str, query: str) -> list:
    """Mock implementation for product search."""
    LOGGER.debug("mock_search_products", store_id=store_id, query=query)
    return [
        {"product_id": "prod-1", "name": "Apple", "category": "Fruits", "price": 1.5}
    ]
