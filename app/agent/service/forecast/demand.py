from __future__ import annotations
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.forecast.demand")

async def fetch_demand(store_id: str, product_id: str, horizon_days: int) -> dict:
    """Mock implementation for demand forecasting."""
    LOGGER.debug("mock_fetch_demand", store_id=store_id, product_id=product_id, horizon_days=horizon_days)
    return {
        "product_id": product_id,
        "predicted_demand": 50.0,
        "confidence_score": 0.85
    }
