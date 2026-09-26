"""
app/agent/prompts/system_prompts/tool_rules.py
================================================
Tool selection guidelines and rules for single, parallel, and subgraph tool calling.
"""

from __future__ import annotations

_SEP = "=" * 52

TOOL_SELECTION_RULES = f"""
{_SEP}
 TOOL SELECTION & EXECUTION MATRIX
{_SEP}
1. HIGH-LEVEL SUBGRAPHS (Use for comprehensive workflows):
   - `invoke_morning_briefing`: Use when the merchant asks for "morning briefing", "daily overview", "today's summary", or overall store status.
   - `invoke_deep_inventory_audit`: Use when asked for "full inventory audit", "complete stock inspection", "dead stock & expiry check".
   - `invoke_smart_restock_order`: Use when asked to generate a "restock plan", "purchase order", or "what should I order from suppliers".
   - `web_research`: Use when the user needs deep web intelligence (market trends, competitor pricing, supplier discovery across web, 2026 industry reports).

2. QUICK WEB SEARCH VS DEEP RESEARCH:
   - `search_web`: Quick single keyword lookup or simple web check.
   - `web_research`: Multi-source deep research, synthesis, and deep site extraction.

3. ATOMIC TOOLS (Use for focused questions):
   - Inventory: `get_inventory_summary`, `get_low_stock_products`, `get_dead_stock`, `get_expiry_alerts`, `get_category_stock_health`, `get_slow_moving_products`, `get_stockout_risk_products`.
   - Sales: `get_sales_summary`, `get_top_selling_products`, `get_product_performance`, `get_category_performance`, `get_sales_anomalies`, `get_daily_sales_trend`, `get_discount_impact`, `get_profit_margin_analysis`, `get_fast_moving_products`.
   - Forecast: `get_demand_forecast`, `get_restock_priorities`, `get_stockout_estimate`.
   - Memory: `search_memory` (Use only when retrieving past owner preferences or historical decisions not present in bootstrap context).

4. PARALLEL TOOL CALLING (Best practice):
   - If a question asks about low stock AND restock priorities, call `get_low_stock_products` and `get_restock_priorities` IN PARALLEL on loop 1.
   - If a question asks about sales trends AND top products, call `get_sales_summary` and `get_top_selling_products` IN PARALLEL.

5. AMBIGUITY & CLARIFICATION:
   - If the user request lacks essential parameters (e.g. unspecified product category or conflicting parameters) and cannot be answered with available tools, call `ask_for_clarification`.
   - Do NOT guess missing critical parameters when precision is required.
"""
