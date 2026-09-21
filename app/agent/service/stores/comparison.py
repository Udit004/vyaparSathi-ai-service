from __future__ import annotations
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.stores.comparison")

async def compare_stores(store_ids: list[str]) -> list:
    """Mock implementation for comparing stores."""
    LOGGER.debug("mock_compare_stores", store_ids=store_ids)
    return [
        {"store_id": "store-1", "revenue": 10000, "profit_margin": 0.2},
        {"store_id": "store-2", "revenue": 12000, "profit_margin": 0.25}
    ]
