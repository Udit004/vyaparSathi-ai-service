from __future__ import annotations
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.inventory.stock_history")

async def fetch_stock_history(store_id: str, product_id: str, days: int) -> list:
    """Mock implementation for fetching historical stock levels."""
    LOGGER.debug("mock_fetch_stock_history", store_id=store_id, product_id=product_id, days=days)
    return [
        {"date": "2023-10-01", "quantity": 100},
        {"date": "2023-10-02", "quantity": 95},
        {"date": "2023-10-03", "quantity": 80},
    ]
