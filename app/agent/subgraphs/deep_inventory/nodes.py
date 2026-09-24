"""
app/agent/subgraphs/deep_inventory/nodes.py
============================================
Nodes for the Deep Inventory Audit subgraph.

Flow: fetch_overview_node --> fetch_risks_node --> fetch_restock_node --> END

  fetch_overview_node — inventory summary + category health (parallel)
  fetch_risks_node    — dead stock + expiry alerts + slow movers + risk
  fetch_restock_node  — restock priorities (top 10 RED)
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict

import structlog

from app.agent.service.inventory.summary import fetch_inventory_summary
from app.agent.service.inventory.category_stock_health import fetch_category_stock_health
from app.agent.service.inventory.dead_stock import fetch_dead_stock
from app.agent.service.inventory.expiry_alerts import fetch_expiry_alerts
from app.agent.service.inventory.slow_moving import fetch_slow_moving_products
from app.agent.service.inventory.inventory_risk import fetch_inventory_risk
from app.agent.service.forecast.restock import fetch_restock_priorities

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.subgraph.deep_inventory")


async def fetch_overview_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Parallel fetch: inventory summary + category health."""
    store_id = state["store_id"]
    threshold = state.get("low_stock_threshold", 10)
    LOGGER.info("deep_inventory.fetch_overview", store_id=store_id)

    inv_data, cat_data = await asyncio.gather(
        fetch_inventory_summary(store_id),
        fetch_category_stock_health(store_id, threshold),
        return_exceptions=True,
    )
    if isinstance(inv_data, Exception):
        LOGGER.warning("deep_inventory.overview.inv_error", error=str(inv_data))
        inv_data = {}
    if isinstance(cat_data, Exception):
        LOGGER.warning("deep_inventory.overview.cat_error", error=str(cat_data))
        cat_data = []

    return {"overview_data": {"inventory": inv_data, "categories": cat_data}}


async def fetch_risks_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Parallel fetch: dead stock + expiry alerts + slow movers + risk score."""
    store_id = state["store_id"]
    expiry_days = state.get("expiry_alert_days", 30)
    LOGGER.info("deep_inventory.fetch_risks", store_id=store_id)

    dead_data, expiry_data, slow_data, risk_data = await asyncio.gather(
        fetch_dead_stock(store_id, days_inactive=90),
        fetch_expiry_alerts(store_id, alert_days=expiry_days),
        fetch_slow_moving_products(store_id),
        fetch_inventory_risk(store_id),
        return_exceptions=True,
    )
    for name, val in [("dead", dead_data), ("expiry", expiry_data), ("slow", slow_data), ("risk", risk_data)]:
        if isinstance(val, Exception):
            LOGGER.warning(f"deep_inventory.risks.{name}_error", error=str(val))

    return {
        "risk_data": {
            "dead_stock": dead_data if not isinstance(dead_data, Exception) else [],
            "expiry": expiry_data if not isinstance(expiry_data, Exception) else {},
            "slow_moving": slow_data if not isinstance(slow_data, Exception) else [],
            "risk": risk_data if not isinstance(risk_data, Exception) else {},
        }
    }


async def fetch_restock_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch top 10 RED-priority restock items and compute risk score."""
    store_id = state["store_id"]
    LOGGER.info("deep_inventory.fetch_restock", store_id=store_id)

    try:
        restock_all = await fetch_restock_priorities(store_id)
        red_items = [r for r in restock_all if r.get("priority") == "RED"][:10]
    except Exception as exc:
        LOGGER.warning("deep_inventory.restock_error", error=str(exc))
        red_items = []

    # Compute composite risk score from all gathered data
    overview = state.get("overview_data", {})
    inv = overview.get("inventory", {})
    risk = state.get("risk_data", {}).get("risk", {})
    expiry = state.get("risk_data", {}).get("expiry", {})

    total = inv.get("total_products", 1) or 1
    oos_pct = (inv.get("out_of_stock_count", 0) / total) * 100
    low_pct = (inv.get("low_stock_count", 0) / total) * 100
    expiring_critical = len((expiry or {}).get("critical", []))

    if oos_pct > 20 or len(red_items) > 10 or expiring_critical > 5:
        risk_score = "high"
    elif oos_pct > 10 or len(red_items) > 3 or expiring_critical > 0:
        risk_score = "medium"
    else:
        risk_score = "low"

    # Build final output dict
    risk_data = state.get("risk_data", {})
    dead_stock = risk_data.get("dead_stock", [])
    slow_moving = risk_data.get("slow_moving", [])
    expiry_obj = risk_data.get("expiry", {})
    categories = overview.get("categories", [])

    output = {
        "total_products": inv.get("total_products", 0),
        "total_value": inv.get("total_inventory_value", 0.0),
        "low_stock_count": inv.get("low_stock_count", 0),
        "out_of_stock_count": inv.get("out_of_stock_count", 0),
        "dead_stock_count": len(dead_stock) if isinstance(dead_stock, list) else 0,
        "slow_moving_count": len(slow_moving) if isinstance(slow_moving, list) else 0,
        "expiring_soon": (expiry_obj.get("critical", []) if isinstance(expiry_obj, dict) else [])[:10],
        "category_health": categories[:15],
        "red_restock_items": red_items,
        "risk_score": risk_score,
        "audit_at": datetime.utcnow().isoformat(),
    }

    return {"audit_output": output}
