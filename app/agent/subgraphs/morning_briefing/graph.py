"""
app/agent/subgraphs/morning_briefing/graph.py
=============================================
Compiled LangGraph for the Morning Briefing subgraph.

State: MorningBriefingState (TypedDict)
  Input fields  : store_id, owner_name, currency, low_stock_threshold
  Internal fields: kpi_data, alert_data
  Output fields  : briefing_output

Graph:
  fetch_kpis_node --> fetch_alerts_node --> synthesize_node --> END
"""

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from app.agent.subgraphs.morning_briefing.nodes import (
    fetch_kpis_node,
    fetch_alerts_node,
    synthesize_node,
)


class MorningBriefingState(TypedDict, total=False):
    # Input
    store_id: str
    owner_name: str
    currency: str
    low_stock_threshold: int
    # Internal
    kpi_data: dict[str, Any]
    alert_data: dict[str, Any]
    # Output
    briefing_output: dict[str, Any]


def build_morning_briefing_graph():
    wf = StateGraph(MorningBriefingState)
    wf.add_node("fetch_kpis", fetch_kpis_node)
    wf.add_node("fetch_alerts", fetch_alerts_node)
    wf.add_node("synthesize", synthesize_node)

    wf.set_entry_point("fetch_kpis")
    wf.add_edge("fetch_kpis", "fetch_alerts")
    wf.add_edge("fetch_alerts", "synthesize")
    wf.add_edge("synthesize", END)

    return wf.compile()


morning_briefing_graph = build_morning_briefing_graph()
