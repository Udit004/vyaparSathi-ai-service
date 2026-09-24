"""
app/agent/subgraphs/smart_restock/nodes.py
==========================================
Nodes for the Smart Restock Order subgraph.

Flow: fetch_priorities_node --> fetch_supplier_context_node --> build_order_node --> END

  fetch_priorities_node       — restock priorities + stockout estimates
  fetch_supplier_context_node — search suppliers by category for RED items
  build_order_node            — LLM generates ranked order memo
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict

import structlog

from app.agent.service.forecast.restock import fetch_restock_priorities
from app.agent.service.forecast.stockout import fetch_stockout_estimate
from app.agent.service.suppliers.search import search_suppliers
from app.lib.llm import get_llm

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.subgraph.smart_restock")


async def fetch_priorities_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch restock priorities and stockout estimates in parallel."""
    store_id = state["store_id"]
    include_yellow = state.get("include_yellow", True)
    LOGGER.info("smart_restock.fetch_priorities", store_id=store_id)

    try:
        restock_data = await fetch_restock_priorities(store_id)
    except Exception as exc:
        LOGGER.warning("smart_restock.priorities_error", error=str(exc))
        restock_data = []

    stockout_data = []

    # Filter by priority
    if include_yellow:
        order_items = [r for r in restock_data if r.get("priority") in ("RED", "YELLOW")]
    else:
        order_items = [r for r in restock_data if r.get("priority") == "RED"]

    return {
        "priority_data": {
            "order_items": order_items,
            "stockout": stockout_data,
        }
    }


async def fetch_supplier_context_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    For the top RED categories, try to find relevant suppliers.
    This is best-effort — failures are silently ignored.
    """
    store_id = state["store_id"]
    order_items = state.get("priority_data", {}).get("order_items", [])
    LOGGER.info("smart_restock.fetch_supplier_context", store_id=store_id, items=len(order_items))

    # Collect unique categories from RED items only
    red_categories = list({
        item.get("category", "General")
        for item in order_items
        if item.get("priority") == "RED" and item.get("category")
    })[:3]  # Max 3 category searches

    supplier_map: dict[str, list] = {}
    if red_categories:
        tasks = [search_suppliers(store_id, category) for category in red_categories]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for cat, result in zip(red_categories, results):
            if not isinstance(result, Exception) and result:
                supplier_map[cat] = result[:2]  # Top 2 suppliers per category

    return {"supplier_data": supplier_map}


async def build_order_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LLM generates a formatted restock order memo."""
    store_id = state["store_id"]
    currency = state.get("currency", "INR")
    lead_time = state.get("lead_time_days", 3)
    order_items = state.get("priority_data", {}).get("order_items", [])
    supplier_map = state.get("supplier_data", {})

    LOGGER.info("smart_restock.build_order", store_id=store_id, order_items=len(order_items))

    red_items = [i for i in order_items if i.get("priority") == "RED"]
    yellow_items = [i for i in order_items if i.get("priority") == "YELLOW"]

    # Compute estimated cost per item (suggested_qty * price proxy)
    enriched_items = []
    total_cost = 0.0
    for item in order_items:
        qty = item.get("suggested_restock_quantity", 0)
        # Use avg_daily_sales * lead_time as a price proxy if price not available
        # Products from fetch_restock_priorities have: name, current_quantity,
        # avg_daily_sales, days_to_stockout, suggested_restock_quantity, priority
        enriched = {
            "name": item.get("name", "Unknown"),
            "priority": item.get("priority", "YELLOW"),
            "current_quantity": item.get("current_quantity", 0),
            "suggested_restock_quantity": qty,
            "days_to_stockout": item.get("days_to_stockout"),
            "category": item.get("category", "General"),
            "estimated_cost": 0.0,  # No price in restock output; LLM will note this
        }
        enriched_items.append(enriched)

    # Build LLM prompt
    red_summary = "\n".join(
        f"  RED [{i+1}] {item['name']} — qty: {item['current_quantity']} left, "
        f"order {item['suggested_restock_quantity']} units"
        + (f", stockout in {item['days_to_stockout']:.0f}d" if item.get("days_to_stockout") else " (out of stock)")
        for i, item in enumerate(red_items[:10])
    ) or "  None"

    yellow_summary = "\n".join(
        f"  YELLOW [{i+1}] {item['name']} — qty: {item['current_quantity']} left, "
        f"order {item['suggested_restock_quantity']} units"
        for i, item in enumerate(yellow_items[:10])
    ) or "  None"

    prompt = (
        f"You are an inventory management assistant for an Indian store (Store ID: {store_id}).\n"
        f"Currency: {currency}. Supplier lead time: {lead_time} days.\n\n"
        f"RED PRIORITY (order IMMEDIATELY):\n{red_summary}\n\n"
        f"YELLOW PRIORITY (order this week):\n{yellow_summary}\n\n"
        "Write a concise, professional restock order memo (max 200 words) for the owner:\n"
        "1. Lead with the most urgent items\n"
        "2. Group by priority (RED first, then YELLOW)\n"
        "3. Include specific quantities for each item\n"
        "4. Add a closing action recommendation\n"
        "Keep it practical and action-oriented."
    )

    try:
        llm = get_llm()
        if llm:
            from langchain_core.messages import HumanMessage
            response = await llm.ainvoke([HumanMessage(content=prompt)])
            order_text = response.content if hasattr(response, "content") else str(response)
            if isinstance(order_text, list):
                order_text = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in order_text
                )
        else:
            order_text = _fallback_order_text(red_items, yellow_items, currency)
    except Exception as exc:
        LOGGER.warning("smart_restock.build_order.llm_error", error=str(exc))
        order_text = _fallback_order_text(red_items, yellow_items, currency)

    output = {
        "order_items": enriched_items,
        "total_estimated_cost": total_cost,
        "red_count": len(red_items),
        "yellow_count": len(yellow_items),
        "order_text": order_text,
        "generated_at": datetime.utcnow().isoformat(),
    }

    return {"restock_output": output}


def _fallback_order_text(red_items, yellow_items, currency) -> str:
    lines = ["RESTOCK ORDER PLAN", "=" * 40]
    if red_items:
        lines.append("\n[URGENT - Order Immediately]")
        for item in red_items[:5]:
            lines.append(f"  - {item['name']}: order {item['suggested_restock_quantity']} units")
    if yellow_items:
        lines.append("\n[This Week]")
        for item in yellow_items[:5]:
            lines.append(f"  - {item['name']}: order {item['suggested_restock_quantity']} units")
    lines.append("\nContact your suppliers and place orders immediately for RED items.")
    return "\n".join(lines)
