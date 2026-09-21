"""
app/agent/service/__init__.py
==============================
Agent data services — real MongoDB queries used exclusively by the agent tools.

The canonical implementations live under domain subfolders:

- inventory/
- sales/
- forecast/
- insights/
"""

from app.agent.service.inventory import fetch_inventory_summary, fetch_low_stock_products
from app.agent.service.sales import fetch_sales_summary, fetch_top_selling_products
from app.agent.service.forecast import fetch_restock_priorities, fetch_forecast_summary
from app.agent.service.insights import fetch_store_insights

__all__ = [
    "fetch_inventory_summary",
    "fetch_low_stock_products",
    "fetch_sales_summary",
    "fetch_top_selling_products",
    "fetch_restock_priorities",
    "fetch_forecast_summary",
    "fetch_store_insights",
]
