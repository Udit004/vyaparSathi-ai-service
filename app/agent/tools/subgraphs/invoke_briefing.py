"""
app/agent/tools/subgraphs/invoke_briefing.py
=============================================
Stub tool that signals the subgraph_router to run the
morning_briefing_subgraph.

The tool does NOT run the subgraph itself — it just returns a typed
marker so the observe_node can set state['active_subgraph'] = "morning_briefing",
and the subgraph_router will invoke the compiled subgraph.
"""

from pydantic import BaseModel, Field
from langchain_core.tools import tool


class InvokeMorningBriefingInput(BaseModel):
    store_id: str = Field(..., description="The store ID to generate the briefing for.")


@tool("invoke_morning_briefing", args_schema=InvokeMorningBriefingInput)
async def invoke_morning_briefing(store_id: str) -> dict:
    """
    [SUBGRAPH TOOL] Generate a complete morning briefing for the store owner.

    This runs the morning_briefing subgraph which performs:
      1. 7-day sales KPI fetch (revenue, transaction count)
      2. Inventory snapshot (low stock, out of stock count)
      3. Expiry alerts (products expiring within 7 days)
      4. Top 5 RED-priority restock alerts
      5. LLM-synthesized concise morning briefing paragraph

    Use when the owner says:
      - "Good morning / what's the update?"
      - "Give me a daily/morning summary"
      - "What do I need to know today?"
      - "How is my store doing today?"

    OUTPUT (MorningBriefingOutput):
      revenue_7d          : float  — 7-day total revenue
      sales_count_7d      : int    — 7-day transaction count
      low_stock_count     : int    — products at/below threshold
      out_of_stock_count  : int    — products with 0 quantity
      expiring_soon_count : int    — products expiring in 7 days
      top_red_restock     : list   — [{name, current_quantity, suggested_restock_quantity, priority}]
      briefing_text       : str    — LLM-generated briefing
      generated_at        : str    — ISO timestamp
    """
    return {"__subgraph__": "morning_briefing", "store_id": store_id}
