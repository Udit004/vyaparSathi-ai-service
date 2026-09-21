from __future__ import annotations
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.inventory.inventory_risk")

async def fetch_inventory_risk(store_id: str) -> list:
    """Mock implementation for assessing inventory risks."""
    LOGGER.debug("mock_fetch_inventory_risk", store_id=store_id)
    return [
        {"product_id": "prod-1", "name": "Laptop", "risk_level": "HIGH", "risk_type": "STOCKOUT", "description": "High demand, low stock"},
        {"product_id": "prod-2", "name": "Fidget Spinner", "risk_level": "MEDIUM", "risk_type": "OVERSTOCK", "description": "Low demand, high stock"}
    ]
