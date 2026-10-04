"""
app/agent/tools/__init__.py
============================
Canonical exports for the domain-structured tool layer.

The live implementations are organized by domain package:
- clarification/
- inventory/
- sales/
- forecast/
- insights/
- products/
- stores/
- suppliers/
"""

from app.agent.tools.clarification import ask_for_clarification

# Inventory
from app.agent.tools.inventory.summary import get_inventory_summary
from app.agent.tools.inventory.low_stock import get_low_stock_products
from app.agent.tools.inventory.stock_history import get_stock_history
from app.agent.tools.inventory.dead_stock import get_dead_stock
from app.agent.tools.inventory.inventory_risk import get_inventory_risk

# Sales
from app.agent.tools.sales.summary import get_sales_summary
from app.agent.tools.sales.top_products import get_top_selling_products
from app.agent.tools.sales.product_performance import get_product_performance
from app.agent.tools.sales.category_performance import get_category_performance
from app.agent.tools.sales.sales_anomalies import get_sales_anomalies

# Forecast
from app.agent.tools.forecast.demand import get_demand_forecast
from app.agent.tools.forecast.restock import get_restock_priorities
from app.agent.tools.forecast.stockout import get_stockout_estimate

# Insights
from app.agent.tools.insights.store_insights import get_store_insights

# Web search
from app.agent.tools.web.search_web import search_web

# Products
from app.agent.tools.products.search import search_products
from app.agent.tools.products.details import get_product_details
from app.agent.tools.products.history import get_product_history

# Stores
from app.agent.tools.stores.summary import get_store_summary
from app.agent.tools.stores.comparison import compare_stores

# Personalization & Proactive Intelligence
from app.agent.tools.memory.merchant_memories import manage_merchant_memories
from app.agent.tools.insights.proactive_insights import manage_proactive_insights
from app.agent.tools.analytics.store_baselines import get_store_baselines
from app.agent.tools.purchases.price_history import check_supplier_price_trends
from app.agent.tools.customers.credit_ledger import get_customer_credit_ledger

__all__ = [
    "ask_for_clarification",
    
    # Inventory
    "get_inventory_summary",
    "get_low_stock_products",
    "get_stock_history",
    "get_dead_stock",
    "get_inventory_risk",
    
    # Sales
    "get_sales_summary",
    "get_top_selling_products",
    "get_product_performance",
    "get_category_performance",
    "get_sales_anomalies",
    
    # Forecast
    "get_demand_forecast",
    "get_restock_priorities",
    "get_stockout_estimate",
    
    # Insights
    "get_store_insights",

    # Web search
    "search_web",
    
    # Products
    "search_products",
    "get_product_details",
    "get_product_history",
    
    # Stores
    "get_store_summary",
    "compare_stores",
    
    # Suppliers
    "search_suppliers",
    "get_supplier_performance",
    "get_supplier_pricing",

    # Personalization & Proactive Intelligence
    "manage_merchant_memories",
    "manage_proactive_insights",
    "get_store_baselines",
    "check_supplier_price_trends",
    "get_customer_credit_ledger",
]

