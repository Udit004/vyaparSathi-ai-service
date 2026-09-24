"""
app/agent/subgraphs/deep_inventory/graph.py
============================================
Compiled LangGraph for the Deep Inventory Audit subgraph.

State: DeepInventoryState (TypedDict)
  Input fields  : store_id, currency, low_stock_threshold, expiry_alert_days
  Internal fields: overview_data, risk_data
  Output fields  : audit_output

Graph:
  fetch_overview_node --> fetch_risks_node --> fetch_restock_node --> END
"""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from app.agent.subgraphs.deep_inventory.nodes import (
    fetch_overview_node,
    fetch_risks_node,
    fetch_restock_node,
)


class DeepInventoryState(TypedDict, total=False):
    # Input
    store_id: str
    currency: str
    low_stock_threshold: int
    expiry_alert_days: int
    # Internal
    overview_data: dict[str, Any]
    risk_data: dict[str, Any]
    # Output
    audit_output: dict[str, Any]


def build_deep_inventory_graph():
    wf = StateGraph(DeepInventoryState)
    wf.add_node("fetch_overview", fetch_overview_node)
    wf.add_node("fetch_risks", fetch_risks_node)
    wf.add_node("fetch_restock", fetch_restock_node)

    wf.set_entry_point("fetch_overview")
    wf.add_edge("fetch_overview", "fetch_risks")
    wf.add_edge("fetch_risks", "fetch_restock")
    wf.add_edge("fetch_restock", END)

    return wf.compile()


deep_inventory_graph = build_deep_inventory_graph()
