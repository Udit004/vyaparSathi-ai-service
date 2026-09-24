"""
app/agent/subgraphs/__init__.py
================================
Subgraph package — purpose-built LangGraph subgraphs that the main
Vyapar Copilot agent can delegate to for deep analytical workflows.

Available subgraphs
-------------------
  morning_briefing    -- Daily store health digest (KPIs + alerts + briefing text)
  deep_inventory      -- Full inventory audit (risks, expiry, slow-movers, categories)
  smart_restock       -- Prioritised restock order plan with supplier hints

Each subgraph is a compiled LangGraph with typed Pydantic Input/Output schemas
documented in `schemas.py`. The main graph invokes them via the subgraph_router
node after the LLM calls the corresponding stub tool.
"""

from app.agent.subgraphs.morning_briefing.graph import morning_briefing_graph
from app.agent.subgraphs.deep_inventory.graph import deep_inventory_graph
from app.agent.subgraphs.smart_restock.graph import smart_restock_graph

__all__ = [
    "morning_briefing_graph",
    "deep_inventory_graph",
    "smart_restock_graph",
]
