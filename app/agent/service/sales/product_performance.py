from __future__ import annotations
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.sales.product_performance")

async def fetch_product_performance(store_id: str, product_id: str) -> dict:
    """Mock implementation for detailed sales performance metrics."""
    LOGGER.debug("mock_fetch_product_performance", store_id=store_id, product_id=product_id)
    return {
        "product_id": product_id,
        "total_revenue": 1500.0,
        "units_sold": 150,
        "conversion_rate": 0.05
    }
