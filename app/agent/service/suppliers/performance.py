from __future__ import annotations
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.suppliers.performance")

async def fetch_supplier_performance(supplier_id: str) -> dict:
    """Mock implementation for evaluating supplier performance."""
    LOGGER.debug("mock_fetch_supplier_performance", supplier_id=supplier_id)
    return {
        "supplier_id": supplier_id,
        "on_time_delivery_rate": 0.95,
        "defect_rate": 0.02,
        "average_lead_time_days": 14
    }
