"""
app/agent/tools/subgraphs/invoke_restock_order.py
=================================================
Stub tool for the smart_restock_subgraph.
"""

from pydantic import BaseModel, Field
from langchain_core.tools import tool


class InvokeSmartRestockInput(BaseModel):
    store_id: str = Field(..., description="The store ID to generate the restock plan for.")
    include_yellow: bool = Field(True, description="Include YELLOW priority items (not just RED). Default True.")


@tool("invoke_smart_restock_order", args_schema=InvokeSmartRestockInput)
async def invoke_smart_restock_order(store_id: str, include_yellow: bool = True) -> dict:
    """
    [SUBGRAPH TOOL] Generate a complete smart restock order plan for the store.

    This runs the smart_restock subgraph which performs:
      1. Restock priority computation (RED = urgent, YELLOW = this week)
      2. Stockout timeline estimates per product
      3. Supplier context for top RED categories (best-effort)
      4. LLM-generated ranked order memo with specific quantities

    Use when the owner asks:
      - "What should I order / restock?"
      - "Give me a restock plan / purchase order"
      - "Which products need ordering urgently?"
      - "Create an order list for me"
      - "What will run out soon?"

    Set include_yellow=False to get only critical (RED) items for a focused order.

    OUTPUT (SmartRestockOutput):
      order_items           : list   — [{name, priority, current_quantity,
                                         suggested_restock_quantity, days_to_stockout,
                                         category, estimated_cost}]
      total_estimated_cost  : float  — sum of estimated order costs
      red_count             : int    — RED priority items
      yellow_count          : int    — YELLOW priority items
      order_text            : str    — LLM-formatted order memo
      generated_at          : str    — ISO timestamp
    """
    return {
        "__subgraph__": "smart_restock",
        "store_id": store_id,
        "include_yellow": include_yellow,
    }
