from __future__ import annotations
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.inventory.dead_stock")

async def fetch_dead_stock(store_id: str, days_inactive: int) -> list:
    """Mock implementation for identifying dead stock."""
    LOGGER.debug("mock_fetch_dead_stock", store_id=store_id, days_inactive=days_inactive)
    return [
        {"product_id": "prod-4", "name": "Old Phone Case", "current_quantity": 50, "last_sold_date": "2023-01-15"}
    ]
