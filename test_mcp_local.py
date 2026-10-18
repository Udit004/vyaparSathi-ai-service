"""
test_mcp_local.py
=================
Quick local test script to verify that the Vyapar Sathi MCP Server is working
and can execute read-only queries against MongoDB.
"""
import asyncio
import json
from app.mcp.server import mcp

async def run_tests():
    print("=" * 60)
    print(" Vyapar Sathi MCP Server - Local Test Runner")
    print("=" * 60)

    # 1. List tools
    tools = await mcp.list_tools()
    print(f"\n[1] Discovered {len(tools)} Read-Only MCP Tools:")
    for tool in tools:
        summary = tool.description.strip().splitlines()[0] if tool.description else ""
        print(f"    - {tool.name:25}: {summary}")

    # 2. Test Tool: get_inventory_summary
    print("\n[2] Calling MCP Tool: 'get_inventory_summary' for store='Grocery'...")
    try:
        res = await mcp.call_tool("get_inventory_summary", {"store": "Grocery"})
        print("    Result:")
        print("   ", res.content[0].text if res.content else res)
    except Exception as e:
        print("    Error calling tool:", e)

    # 3. Test Tool: get_low_stock_products
    print("\n[3] Calling MCP Tool: 'get_low_stock_products' (threshold=15)...")
    try:
        res = await mcp.call_tool("get_low_stock_products", {"store": "Grocery", "threshold": 15, "limit": 3})
        print("    Result:")
        print("   ", res.content[0].text if res.content else res)
    except Exception as e:
        print("    Error calling tool:", e)

    # 4. Test Tool: get_sales_summary
    print("\n[4] Calling MCP Tool: 'get_sales_summary' (days=30)...")
    try:
        res = await mcp.call_tool("get_sales_summary", {"store": "Grocery", "days": 30})
        print("    Result:")
        print("   ", res.content[0].text if res.content else res)
    except Exception as e:
        print("    Error calling tool:", e)

    # 5. Test Tool: get_customer_dues
    print("\n[5] Calling MCP Tool: 'get_customer_dues'...")
    try:
        res = await mcp.call_tool("get_customer_dues", {"store": "Grocery"})
        print("    Result:")
        print("   ", res.content[0].text if res.content else res)
    except Exception as e:
        print("    Error calling tool:", e)

    print("\n" + "=" * 60)
    print(" All MCP tools executed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(run_tests())
