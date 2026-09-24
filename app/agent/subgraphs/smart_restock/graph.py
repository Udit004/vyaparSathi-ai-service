"""
app/agent/subgraphs/smart_restock/graph.py
==========================================
Compiled LangGraph for the Smart Restock Order subgraph.

State: SmartRestockState (TypedDict)
  Input fields  : store_id, currency, lead_time_days, include_yellow
  Internal fields: priority_data, supplier_data
  Output fields  : restock_output

Graph:
  fetch_priorities --> fetch_supplier_context --> build_order --> END
"""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from app.agent.subgraphs.smart_restock.nodes import (
    fetch_priorities_node,
    fetch_supplier_context_node,
    build_order_node,
)


class SmartRestockState(TypedDict, total=False):
    # Input
    store_id: str
    currency: str
    lead_time_days: int
    include_yellow: bool
    # Internal
    priority_data: dict[str, Any]
    supplier_data: dict[str, Any]
    # Output
    restock_output: dict[str, Any]


def build_smart_restock_graph():
    wf = StateGraph(SmartRestockState)
    wf.add_node("fetch_priorities", fetch_priorities_node)
    wf.add_node("fetch_supplier_context", fetch_supplier_context_node)
    wf.add_node("build_order", build_order_node)

    wf.set_entry_point("fetch_priorities")
    wf.add_edge("fetch_priorities", "fetch_supplier_context")
    wf.add_edge("fetch_supplier_context", "build_order")
    wf.add_edge("build_order", END)

    return wf.compile()


smart_restock_graph = build_smart_restock_graph()
