from __future__ import annotations
from typing import Optional
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.suppliers.pricing")

async def fetch_supplier_pricing(supplier_id: str, product_id: Optional[str] = None) -> list:
    """Mock implementation for getting supplier pricing."""
    LOGGER.debug("mock_fetch_supplier_pricing", supplier_id=supplier_id, product_id=product_id)
    return [
        {"product_id": "prod-1", "unit_price": 10.0, "bulk_discount_threshold": 100, "bulk_price": 9.0}
    ]
