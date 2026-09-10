from .inventory import get_inventory_summary, get_low_stock_products
from .sales import get_sales_summary, get_top_selling_products
from .forecast import get_restock_priorities, get_forecast_summary
from .insights import get_store_insights

# LangGraph ToolNode will use this list
VYAPAR_TOOLS = [
    get_inventory_summary,
    get_low_stock_products,
    get_sales_summary,
    get_top_selling_products,
    get_restock_priorities,
    get_forecast_summary,
    get_store_insights,
]

def get_tool_by_name(name: str):
    """Helper to dispatch tool calls."""
    for tool in VYAPAR_TOOLS:
        if tool.name == name:
            return tool
    return None
