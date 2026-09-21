from __future__ import annotations
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.products.history")

async def fetch_product_history(store_id: str, product_id: str) -> list:
    """Mock implementation for fetching product history."""
    LOGGER.debug("mock_fetch_product_history", store_id=store_id, product_id=product_id)
    return [
        {"date": "2023-01-01", "event_type": "PRICE_CHANGE", "details": "Price changed from 1.0 to 1.5"}
    ]
