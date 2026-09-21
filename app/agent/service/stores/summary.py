from __future__ import annotations
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.stores.summary")

async def fetch_store_summary(store_id: str) -> dict:
    """Mock implementation for fetching store summary."""
    LOGGER.debug("mock_fetch_store_summary", store_id=store_id)
    return {
        "store_id": store_id,
        "name": "Main Store",
        "location": "New York",
        "status": "ACTIVE"
    }
