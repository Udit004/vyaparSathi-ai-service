from app.agent.tools.clarification import ask_for_clarification

# Inventory
from app.agent.tools.inventory.summary import get_inventory_summary
from app.agent.tools.inventory.low_stock import get_low_stock_products
from app.agent.tools.inventory.stock_history import get_stock_history
from app.agent.tools.inventory.dead_stock import get_dead_stock
from app.agent.tools.inventory.inventory_risk import get_inventory_risk
from app.agent.tools.inventory.expiry_alerts import get_expiry_alerts
from app.agent.tools.inventory.category_stock_health import get_category_stock_health
from app.agent.tools.inventory.slow_moving import get_slow_moving_products
from app.agent.tools.inventory.stockout_risk import get_stockout_risk_products

# Sales
from app.agent.tools.sales.summary import get_sales_summary
from app.agent.tools.sales.top_products import get_top_selling_products
from app.agent.tools.sales.product_performance import get_product_performance
from app.agent.tools.sales.category_performance import get_category_performance
from app.agent.tools.sales.sales_anomalies import get_sales_anomalies
from app.agent.tools.sales.daily_trend import get_daily_sales_trend
from app.agent.tools.sales.discount_impact import get_discount_impact
from app.agent.tools.sales.profit_margin import get_profit_margin_analysis
from app.agent.tools.sales.fast_moving import get_fast_moving_products

# Forecast
from app.agent.tools.forecast.demand import get_demand_forecast
from app.agent.tools.forecast.restock import get_restock_priorities
from app.agent.tools.forecast.stockout import get_stockout_estimate

# Insights
from app.agent.tools.insights.store_insights import get_store_insights

# Products
from app.agent.tools.products.search import search_products
from app.agent.tools.products.details import get_product_details
from app.agent.tools.products.history import get_product_history

# Stores
from app.agent.tools.stores.summary import get_store_summary
from app.agent.tools.stores.comparison import compare_stores

# Suppliers
from app.agent.tools.suppliers.search import search_suppliers
from app.agent.tools.suppliers.performance import get_supplier_performance
from app.agent.tools.suppliers.pricing import get_supplier_pricing

# Subgraphs
from app.agent.tools.subgraphs.invoke_briefing import invoke_morning_briefing
from app.agent.tools.subgraphs.invoke_inventory_audit import invoke_deep_inventory_audit
from app.agent.tools.subgraphs.invoke_restock_order import invoke_smart_restock_order

# Memory
from app.agent.tools.memory.search import search_memory

# LangGraph ToolNode will use this list
VYAPAR_TOOLS = [
    ask_for_clarification,
    
    # Inventory
    get_inventory_summary,
    get_low_stock_products,
    get_stock_history,
    get_dead_stock,
    get_inventory_risk,
    get_expiry_alerts,
    get_category_stock_health,
    get_slow_moving_products,
    get_stockout_risk_products,
    
    # Sales
    get_sales_summary,
    get_top_selling_products,
    get_product_performance,
    get_category_performance,
    get_sales_anomalies,
    get_daily_sales_trend,
    get_discount_impact,
    get_profit_margin_analysis,
    get_fast_moving_products,
    
    # Forecast
    get_demand_forecast,
    get_restock_priorities,
    get_stockout_estimate,
    
    # Insights
    get_store_insights,
    
    # Products
    search_products,
    get_product_details,
    get_product_history,
    
    # Stores
    get_store_summary,
    compare_stores,
    
    # Suppliers
    search_suppliers,
    get_supplier_performance,
    get_supplier_pricing,
    
    # Subgraphs
    invoke_morning_briefing,
    invoke_deep_inventory_audit,
    invoke_smart_restock_order,

    # Memory search (agent-controlled long-term memory retrieval)
    search_memory,
]

def get_tool_by_name(name: str):
    """Helper to dispatch tool calls."""
    for tool in VYAPAR_TOOLS:
        if tool.name == name:
            return tool
    return None
