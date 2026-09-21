from __future__ import annotations
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.sales.sales_anomalies")

async def fetch_sales_anomalies(store_id: str) -> list:
    """Mock implementation for detecting unusual sales spikes or drops."""
    LOGGER.debug("mock_fetch_sales_anomalies", store_id=store_id)
    return [
        {"product_id": "prod-x", "anomaly_type": "SPIKE", "description": "Sales spiked by 300% today."}
    ]
