from __future__ import annotations
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.suppliers.search")

async def search_suppliers(query: str) -> list:
    """Mock implementation for supplier search."""
    LOGGER.debug("mock_search_suppliers", query=query)
    return [
        {"supplier_id": "sup-1", "name": "Global Traders", "rating": 4.5}
    ]
