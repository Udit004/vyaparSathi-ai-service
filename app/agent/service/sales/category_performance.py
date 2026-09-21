from __future__ import annotations
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.sales.category_performance")

async def fetch_category_performance(store_id: str) -> list:
    """Mock implementation for sales metrics aggregated by category."""
    LOGGER.debug("mock_fetch_category_performance", store_id=store_id)
    return [
        {"category": "Electronics", "revenue": 5000.0, "units_sold": 200},
        {"category": "Accessories", "revenue": 1000.0, "units_sold": 500}
    ]
