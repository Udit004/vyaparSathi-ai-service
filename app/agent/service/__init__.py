"""
app/agent/service/__init__.py
==============================
Agent data services — real MongoDB queries used exclusively by the agent tools.

Each module owns one domain (inventory, sales, forecast, insights) so
the codebase stays modular, readable, and independently testable.

Layout:

    app/agent/service/
    ├── __init__.py
    ├── inventory.py    # fetch_inventory_summary, fetch_low_stock_products
    ├── sales.py        # fetch_sales_summary, fetch_top_selling_products
    ├── forecast.py     # fetch_restock_priorities, fetch_forecast_summary
    └── insights.py     # fetch_store_insights
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