"""
app/agent/service/__init__.py
==============================
Agent data services — real MongoDB queries and mocks used exclusively by the agent tools.

The canonical implementations live under domain subfolders:
- inventory/
- sales/
- forecast/
- insights/
- products/
- stores/
- suppliers/
"""

# Inventory
from app.agent.service.inventory.summary import fetch_inventory_summary
from app.agent.service.inventory.low_stock import fetch_low_stock_products
from app.agent.service.inventory.stock_history import fetch_stock_history
from app.agent.service.inventory.dead_stock import fetch_dead_stock
from app.agent.service.inventory.inventory_risk import fetch_inventory_risk

# Sales
from app.agent.service.sales.summary import fetch_sales_summary
from app.agent.service.sales.top_products import fetch_top_selling_products
from app.agent.service.sales.product_performance import fetch_product_performance
from app.agent.service.sales.category_performance import fetch_category_performance
from app.agent.service.sales.sales_anomalies import fetch_sales_anomalies

# Forecast
from app.agent.service.forecast.demand import fetch_demand
from app.agent.service.forecast.restock import fetch_restock_priorities
from app.agent.service.forecast.stockout import fetch_stockout_estimate

# Insights
from app.agent.service.insights.store_insights import fetch_store_insights

# Products
from app.agent.service.products.search import search_products
from app.agent.service.products.details import fetch_product_details
from app.agent.service.products.history import fetch_product_history

# Stores
from app.agent.service.stores.summary import fetch_store_summary
from app.agent.service.stores.comparison import compare_stores

# Suppliers
from app.agent.service.suppliers.search import search_suppliers
from app.agent.service.suppliers.performance import fetch_supplier_performance
from app.agent.service.suppliers.pricing import fetch_supplier_pricing

__all__ = [
    # Inventory
    "fetch_inventory_summary",
    "fetch_low_stock_products",
    "fetch_stock_history",
    "fetch_dead_stock",
    "fetch_inventory_risk",
    
    # Sales
    "fetch_sales_summary",
    "fetch_top_selling_products",
    "fetch_product_performance",
    "fetch_category_performance",
    "fetch_sales_anomalies",
    
    # Forecast
    "fetch_demand",
    "fetch_restock_priorities",
    "fetch_stockout_estimate",
    
    # Insights
    "fetch_store_insights",
    
    # Products
    "search_products",
    "fetch_product_details",
    "fetch_product_history",
    
    # Stores
    "fetch_store_summary",
    "compare_stores",
    
    # Suppliers
    "search_suppliers",
    "fetch_supplier_performance",
    "fetch_supplier_pricing",
]
