from __future__ import annotations
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.products.details")

async def fetch_product_details(store_id: str, product_id: str) -> dict:
    """Mock implementation for fetching product details."""
    LOGGER.debug("mock_fetch_product_details", store_id=store_id, product_id=product_id)
    return {
        "product_id": product_id,
        "name": "Apple",
        "category": "Fruits",
        "description": "A red apple",
        "price": 1.5,
        "sku": "APP-001"
    }
