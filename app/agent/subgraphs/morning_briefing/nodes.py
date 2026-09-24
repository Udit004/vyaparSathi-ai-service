"""
app/agent/subgraphs/morning_briefing/nodes.py
==============================================
Nodes for the Morning Briefing subgraph.

Flow: fetch_kpis_node --> fetch_alerts_node --> synthesize_node --> END

  fetch_kpis_node   — parallel fetch: 7d sales KPIs + inventory summary
                      + expiry alerts (7d window)
  fetch_alerts_node — parallel fetch: low-stock list + top-5 RED restocks
  synthesize_node   — LLM generates a concise morning briefing paragraph
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict

import structlog

from app.agent.service.inventory.expiry_alerts import fetch_expiry_alerts
from app.agent.service.inventory.summary import fetch_inventory_summary
from app.agent.service.forecast.restock import fetch_restock_priorities
from app.agent.service.sales.summary import fetch_sales_summary
from app.lib.llm import get_llm

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.subgraph.morning_briefing")


async def fetch_kpis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Concurrently fetch:
      - 7-day sales summary
      - Inventory summary
      - 7-day expiry alerts
    """
    store_id = state["store_id"]
    LOGGER.info("morning_briefing.fetch_kpis", store_id=store_id)

    sales_data, inv_data, expiry_data = await asyncio.gather(
        fetch_sales_summary(store_id, days_lookback=7),
        fetch_inventory_summary(store_id),
        fetch_expiry_alerts(store_id, alert_days=7),
        return_exceptions=True,
    )

    # Graceful degradation: replace exceptions with empty dicts
    if isinstance(sales_data, Exception):
        LOGGER.warning("morning_briefing.fetch_kpis.sales_error", error=str(sales_data))
        sales_data = {}
    if isinstance(inv_data, Exception):
        LOGGER.warning("morning_briefing.fetch_kpis.inv_error", error=str(inv_data))
        inv_data = {}
    if isinstance(expiry_data, Exception):
        LOGGER.warning("morning_briefing.fetch_kpis.expiry_error", error=str(expiry_data))
        expiry_data = {}

    return {
        "kpi_data": {
            "sales": sales_data,
            "inventory": inv_data,
            "expiry": expiry_data,
        }
    }


async def fetch_alerts_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Concurrently fetch restock priorities (TOP 5 RED only) for the alert section.
    """
    store_id = state["store_id"]
    LOGGER.info("morning_briefing.fetch_alerts", store_id=store_id)

    try:
        restock_all = await fetch_restock_priorities(store_id)
        red_items = [r for r in restock_all if r.get("priority") == "RED"][:5]
    except Exception as exc:
        LOGGER.warning("morning_briefing.fetch_alerts.error", error=str(exc))
        red_items = []

    return {"alert_data": {"top_red_restock": red_items}}


async def synthesize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call the LLM to generate a concise morning briefing paragraph using
    the fetched KPIs and alerts.
    """
    store_id = state["store_id"]
    owner_name = state.get("owner_name", "")
    currency = state.get("currency", "INR")
    kpi = state.get("kpi_data", {})
    alerts = state.get("alert_data", {})

    LOGGER.info("morning_briefing.synthesize", store_id=store_id)

    sales = kpi.get("sales", {})
    inv = kpi.get("inventory", {})
    expiry = kpi.get("expiry", {})
    red_items = alerts.get("top_red_restock", [])

    greeting = f"Good morning, {owner_name}!" if owner_name else "Good morning!"

    prompt = (
        f"{greeting} Here is a morning briefing for store {store_id}.\n\n"
        f"Currency: {currency}\n"
        f"7-day revenue: {currency} {sales.get('total_revenue', 0):,.2f} "
        f"({sales.get('total_sales_count', 0)} transactions)\n"
        f"Inventory: {inv.get('total_products', 0)} products, "
        f"{inv.get('low_stock_count', 0)} low stock, "
        f"{inv.get('out_of_stock_count', 0)} out of stock\n"
        f"Expiring in 7 days: {len(expiry.get('critical', []))} products\n"
        f"RED restock alerts: {len(red_items)} products\n"
    )

    if red_items:
        prompt += "\nTop urgent restocks:\n"
        for item in red_items[:3]:
            prompt += (
                f"  - {item.get('name', 'Unknown')}: "
                f"only {item.get('current_quantity', 0)} units left "
                f"(suggest ordering {item.get('suggested_restock_quantity', 0)} units)\n"
            )

    prompt += (
        "\nWrite a concise, action-oriented morning briefing (3-4 bullet points max) "
        "for the store owner. Lead with the most urgent items. Be specific with numbers. "
        "Use the owner's currency. Keep it under 150 words."
    )

    try:
        llm = get_llm()
        if llm:
            from langchain_core.messages import HumanMessage
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            briefing_text = response.content if hasattr(response, "content") else str(response)
            if isinstance(briefing_text, list):
                briefing_text = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in briefing_text
                )
        else:
            briefing_text = _fallback_briefing(greeting, sales, inv, red_items, currency)
    except Exception as exc:
        LOGGER.warning("morning_briefing.synthesize.llm_error", error=str(exc))
        briefing_text = _fallback_briefing(greeting, sales, inv, red_items, currency)

    # Build output dict
    output = {
        "revenue_7d": float(sales.get("total_revenue", 0)),
        "sales_count_7d": int(sales.get("total_sales_count", 0)),
        "low_stock_count": int(inv.get("low_stock_count", 0)),
        "out_of_stock_count": int(inv.get("out_of_stock_count", 0)),
        "expiring_soon_count": len(expiry.get("critical", [])),
        "top_red_restock": red_items,
        "briefing_text": briefing_text,
        "generated_at": datetime.utcnow().isoformat(),
    }

    return {"briefing_output": output}


def _fallback_briefing(greeting, sales, inv, red_items, currency) -> str:
    lines = [greeting]
    rev = sales.get("total_revenue", 0)
    lines.append(f"- 7-day revenue: {currency} {rev:,.2f} ({sales.get('total_sales_count', 0)} orders)")
    ls = inv.get("low_stock_count", 0)
    oos = inv.get("out_of_stock_count", 0)
    if oos > 0:
        lines.append(f"- {oos} products are OUT OF STOCK — restock immediately.")
    if ls > 0:
        lines.append(f"- {ls} products are running low on stock.")
    if red_items:
        names = ", ".join(r.get("name", "?") for r in red_items[:3])
        lines.append(f"- Urgent restock needed: {names}.")
    return "\n".join(lines)
