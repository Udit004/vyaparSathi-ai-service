"""app/agent/tools/inventory/expiry_alerts.py"""
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from datetime import datetime
from app.agent.service.inventory.expiry_alerts import fetch_expiry_alerts


class ExpiryAlertsInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    alert_days: int = Field(30, description="Look-ahead window in days for expiry alerts (default 30).")


class ExpiryAlertItem(BaseModel):
    product_id: str
    name: str
    category: str
    quantity: int
    exp_date: str
    days_left: int


class ExpiryAlertsOutput(BaseModel):
    expired: List[ExpiryAlertItem]
    critical: List[ExpiryAlertItem]    # expiring within 7 days
    warning: List[ExpiryAlertItem]     # expiring within alert_days
    total_at_risk: int
    alert_days: int
    fetched_at: str


@tool("get_expiry_alerts", args_schema=ExpiryAlertsInput)
async def get_expiry_alerts(store_id: str, alert_days: int = 30) -> ExpiryAlertsOutput:
    """
    Get products that are expired or approaching their expiry date, grouped by
    urgency: 'expired' (already past), 'critical' (within 7 days), 'warning'
    (within alert_days). Use this to flag food, medicine, or perishable items at risk.
    """
    data = await fetch_expiry_alerts(store_id, alert_days)
    return ExpiryAlertsOutput(
        expired=[ExpiryAlertItem(**i) for i in data["expired"]],
        critical=[ExpiryAlertItem(**i) for i in data["critical"]],
        warning=[ExpiryAlertItem(**i) for i in data["warning"]],
        total_at_risk=data["total_at_risk"],
        alert_days=data["alert_days"],
        fetched_at=datetime.utcnow().isoformat(),
    )
