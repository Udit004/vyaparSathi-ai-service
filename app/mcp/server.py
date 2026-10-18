"""
app/mcp/server.py
=================
MCP (Model Context Protocol) Server for Vyapar Sathi.
Exposes read-only business tools to MCP clients like Claude, ChatGPT, and Cursor.

Supports:
- Local stdio transport (Claude Desktop / Cursor / local CLI)
- Remote HTTP SSE transport (ChatGPT / Web MCP clients / FastAPI mount)
"""

import os
import sys
import argparse
from datetime import datetime, date
from typing import Any, Dict, List, Optional
from bson import ObjectId
from dotenv import load_dotenv
import structlog

# Load environment configuration (.env)
load_dotenv()

from mcp.server.fastmcp import FastMCP

# Import core MongoDB read services
from app.agent.service.inventory.summary import fetch_inventory_summary
from app.agent.service.inventory.low_stock import fetch_low_stock_products
from app.agent.service.inventory.dead_stock import fetch_dead_stock
from app.agent.service.inventory.expiry_alerts import fetch_expiry_alerts
from app.agent.service.sales.summary import fetch_sales_summary
from app.agent.service.sales.top_products import fetch_top_selling_products
from app.agent.service.sales.daily_trend import fetch_daily_sales_trend
from app.agent.service.buyers.dues import fetch_buyer_dues
from app.agent.service.buyers.search import fetch_buyers
from app.agent.service.products.search import search_products
from app.agent.service.products.details import fetch_product_details
from app.agent.service.profit_loss.report import fetch_profit_loss_report
from app.agent.service.expenses.summary import fetch_expense_summary
from app.agent.service.insights.store_insights import fetch_store_insights

LOGGER = structlog.get_logger("vyaparsathi.mcp.server")

# Default store identifier fallback if not supplied by the caller
DEFAULT_STORE = os.getenv("VYAPAR_DEFAULT_STORE", "Grocery")


def _sanitize(data: Any) -> Any:
    """Recursively convert MongoDB ObjectIds, datetimes, and complex objects to JSON-serializable types."""
    if isinstance(data, dict):
        return {str(k): _sanitize(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize(v) for v in data]
    elif isinstance(data, (datetime, date)):
        return data.isoformat()
    elif isinstance(data, ObjectId):
        return str(data)
    return data


import contextvars
from app.config.database import get_database

# Per-request / Per-session store context (set via query param ?store=..., ?user=..., or header)
current_store_context: contextvars.ContextVar[str] = contextvars.ContextVar("current_store_context", default="")


def _get_target_store(store_id_or_name: Optional[str]) -> str:
    """Returns the requested store identifier, or the session context store, or fallback."""
    cleaned = (store_id_or_name or "").strip()
    if cleaned:
        return cleaned
    ctx = current_store_context.get("").strip()
    if ctx:
        return ctx
    return DEFAULT_STORE


# ---------------------------------------------------------------------------
# Initialize MCP Server
# ---------------------------------------------------------------------------
mcp = FastMCP("VyaparSathi-Assistant")


# ===========================================================================
# 0. STORE DIRECTORY & CONTEXT DISCOVERY
# ===========================================================================

@mcp.tool()
async def list_available_stores(user_query: str = "") -> List[Dict[str, Any]]:
    """
    List registered stores in Vyapar Sathi with their store names, IDs, and owners.
    Useful when a user has multiple stores (e.g. 'Udit Kumar Tiwari', 'RAM JI STORE', 'udit store')
    to determine which store they want to check.
    Args:
        user_query: Optional user name, email, or store name filter
    """
    db = get_database()
    query: Dict[str, Any] = {}
    if user_query:
        query["name"] = {"$regex": user_query.strip(), "$options": "i"}

    stores = await db["stores"].find(query, {"_id": 1, "name": 1, "owner": 1}).to_list(20)
    user_ids = [s["owner"] for s in stores if s.get("owner")]
    users = {}
    if user_ids:
        user_docs = await db["users"].find({"_id": {"$in": user_ids}}, {"_id": 1, "name": 1, "email": 1}).to_list(len(user_ids))
        users = {str(u["_id"]): u for u in user_docs}

    results = []
    for s in stores:
        owner_id = str(s.get("owner", ""))
        u = users.get(owner_id, {})
        results.append({
            "store_id": str(s["_id"]),
            "store_name": s.get("name", "Unnamed Store"),
            "owner_name": u.get("name", ""),
            "owner_email": u.get("email", ""),
        })
    return _sanitize(results)


@mcp.tool()
async def get_store_overview(store: str = "") -> Dict[str, Any]:
    """
    Get a comprehensive single-snapshot overview of the store's current business health:
    Includes inventory stats (total items, inventory valuation, low stock items), recent sales revenue and order counts,
    and outstanding customer khata dues.
    Use this tool whenever the user asks 'tell me about my store', 'store information', 'how is my store doing', or 'store overview'.
    Args:
        store: Store name or ObjectId (leave blank to query the authenticated store)
    """
    target = _get_target_store(store)
    inv = await fetch_inventory_summary(target)
    sales = await fetch_sales_summary(target, days_lookback=30)
    dues = await fetch_buyer_dues(target, min_due=1.0, limit=5)

    return _sanitize({
        "store": target,
        "inventory_status": inv,
        "sales_performance_30_days": sales,
        "customer_khata_dues": {
            "total_outstanding_amount": dues.get("total_due", 0),
            "customers_with_dues_count": len(dues.get("buyers", [])),
        },
    })


# ===========================================================================
# 1. INVENTORY TOOLS (READ-ONLY)
# ===========================================================================

@mcp.tool()
async def get_inventory_summary(store: str = "") -> Dict[str, Any]:
    """
    Get a high-level summary of the store's inventory.
    Returns:
        - total_products: Total number of active products
        - low_stock_count: Number of products below low stock threshold
        - out_of_stock_count: Number of items with zero stock
        - total_inventory_value: Total financial value of stock in store
        - low_stock_threshold_used: Stock alert threshold (e.g. 10)
    """
    target = _get_target_store(store)
    result = await fetch_inventory_summary(target)
    return _sanitize(result)


@mcp.tool()
async def get_low_stock_products(store: str = "", threshold: int = 10, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get products that are running low in stock and need replenishment.
    Args:
        store: Store name or ObjectId (e.g. 'Grocery')
        threshold: Products with quantity <= this number are returned (default: 10)
        limit: Maximum number of products to return (default: 50)
    Returns:
        List of low stock products with id, name, category, quantity, and price.
    """
    target = _get_target_store(store)
    result = await fetch_low_stock_products(target, threshold=threshold, limit=limit)
    return _sanitize(result)


@mcp.tool()
async def get_dead_stock(store: str = "", days_inactive: int = 60) -> List[Dict[str, Any]]:
    """
    Find slow-moving or dead stock products that have had zero sales for N days.
    Args:
        store: Store name or ObjectId
        days_inactive: Lookback window in days without sales (default: 60)
    """
    target = _get_target_store(store)
    result = await fetch_dead_stock(target, days_inactive=days_inactive)
    return _sanitize(result)


@mcp.tool()
async def get_expiry_alerts(store: str = "", alert_days: int = 30) -> Dict[str, Any]:
    """
    Retrieve products that are expired or expiring within the specified days.
    Returns urgency buckets:
        - expired: already passed expiry date
        - critical: expiring within 7 days
        - warning: expiring within alert_days
        - total_at_risk: total count
    """
    target = _get_target_store(store)
    result = await fetch_expiry_alerts(target, alert_days=alert_days)
    return _sanitize(result)


# ===========================================================================
# 2. PRODUCT CATALOG SEARCH & DETAILS (READ-ONLY)
# ===========================================================================

@mcp.tool()
async def search_product_catalog(query: str, store: str = "") -> List[Dict[str, Any]]:
    """
    Search products in the catalog by name, category, brand, or SKU/barcode.
    Args:
        query: Search term (e.g. 'Rice', 'Milk', 'Electronics')
        store: Optional store name or ObjectId
    """
    target = _get_target_store(store)
    result = await search_products(target, query=query)
    return _sanitize(result)


@mcp.tool()
async def get_product_details(product_id: str, store: str = "") -> Dict[str, Any]:
    """
    Get complete details for a specific product by its ID.
    Args:
        product_id: The 24-character hex ObjectId of the product
        store: Optional store identifier
    """
    target = _get_target_store(store)
    result = await fetch_product_details(target, product_id)
    return _sanitize(result)


# ===========================================================================
# 3. SALES & PERFORMANCE TOOLS (READ-ONLY)
# ===========================================================================

@mcp.tool()
async def get_sales_summary(store: str = "", days: int = 30) -> Dict[str, Any]:
    """
    Get overall sales performance and KPIs for the store over a given period.
    Returns:
        - total_revenue: Total sales revenue
        - total_sales_count: Number of completed orders / transactions
        - average_order_value: Average revenue per order
        - days_covered: Number of days in lookback period
    """
    target = _get_target_store(store)
    result = await fetch_sales_summary(target, days_lookback=days)
    return _sanitize(result)


@mcp.tool()
async def get_top_selling_products(store: str = "", limit: int = 10, days: int = 30) -> List[Dict[str, Any]]:
    """
    Get top selling products ranked by sales quantity and revenue.
    Args:
        store: Store identifier
        limit: Number of top items to return (default: 10)
        days: Period in days to analyze (default: 30)
    """
    target = _get_target_store(store)
    result = await fetch_top_selling_products(target, limit=limit, days_lookback=days)
    return _sanitize(result)


@mcp.tool()
async def get_daily_sales_trend(store: str = "", days: int = 7) -> List[Dict[str, Any]]:
    """
    Get day-by-day sales breakdown and order volume for recent days.
    Useful for seeing daily trends and weekly patterns.
    """
    target = _get_target_store(store)
    result = await fetch_daily_sales_trend(target, days=days)
    return _sanitize(result)


# ===========================================================================
# 4. CUSTOMER DUES & KHATA / LEDGER (READ-ONLY)
# ===========================================================================

@mcp.tool()
async def get_customer_dues(store: str = "", min_due: float = 0.01, limit: int = 20) -> Dict[str, Any]:
    """
    Get customers with outstanding unpaid credit (Udhar / Khata dues).
    Returns list of customers with contact info, total sales, amount paid, and amount due,
    sorted from highest due to lowest.
    """
    target = _get_target_store(store)
    result = await fetch_buyer_dues(target, min_due=min_due, limit=limit)
    return _sanitize(result)


@mcp.tool()
async def search_customers(query: str = "", store: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search customer list (buyers) by name or phone number.
    """
    target = _get_target_store(store)
    result = await fetch_buyers(target, query=query, limit=limit)
    return _sanitize(result)


# ===========================================================================
# 5. FINANCIAL REPORTS & INSIGHTS (READ-ONLY)
# ===========================================================================

@mcp.tool()
async def get_profit_loss_report(store: str = "", days: int = 30) -> Dict[str, Any]:
    """
    Get true Profit & Loss (P&L) statement calculated from:
        - Revenue (Sales total)
        - Cost of Goods Sold (COGS based on purchase prices)
        - Gross Profit & Gross Margin %
        - Operating Expenses
        - Net Profit & Net Margin %
    """
    target = _get_target_store(store)
    result = await fetch_profit_loss_report(target, days_lookback=days)
    return _sanitize(result)


@mcp.tool()
async def get_expense_summary(store: str = "", days: int = 30) -> Dict[str, Any]:
    """
    Get business expense summary grouped by expense categories (e.g. Rent, Utilities, Salaries)
    and total expense amount.
    """
    target = _get_target_store(store)
    result = await fetch_expense_summary(target, days_lookback=days)
    return _sanitize(result)


@mcp.tool()
async def get_store_insights(store: str = "") -> List[Dict[str, Any]]:
    """
    Get AI-driven proactive store insights, including stockout risks, revenue drivers,
    dead stock warnings, and daily recommended action items.
    """
    target = _get_target_store(store)
    result = await fetch_store_insights(target)
    return _sanitize(result)


# ===========================================================================
# Standalone CLI Runner (Stdio & SSE)
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Vyapar Sathi MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport type: 'stdio' for Claude Desktop/local, 'sse' or 'streamable-http' for web/ChatGPT.",
    )
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on for SSE transport (default: 8001)")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface (default: 0.0.0.0)")

    args = parser.parse_args()

    print(f"[VyaparSathi MCP] Starting server with transport: {args.transport}", file=sys.stderr)
    mcp.settings.host = args.host
    mcp.settings.port = args.port

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        print(f"[VyaparSathi MCP] Listening on http://{args.host}:{args.port}/sse", file=sys.stderr)
        mcp.run(transport="sse")
    elif args.transport == "streamable-http":
        print(f"[VyaparSathi MCP] Listening on http://{args.host}:{args.port}/mcp", file=sys.stderr)
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
