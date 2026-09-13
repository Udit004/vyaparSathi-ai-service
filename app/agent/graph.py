"""
app/agent/graph.py
==================

LangGraph graph assembly for the Vyapar Copilot agent.

Graph flow
----------
                          ┌──────────────────┐
                          │    think node     │
                          │  (LLM reasoning)  │
                          └──────┬───────────┘
                                 │
                  ┌──────────────┼──────────────┐
                  │              │              │
        memory_query    pending_tool     goal_status
          needed?         calls?         == "complete"
                  │              │              │
                  ▼              ▼              ▼
          ┌──────────┐  ┌──────────┐  ┌──────────────┐
          │ memory   │  │  tool    │  │     END      │
          │  query   │  │  node    │  │ (memory write│
          └────┬─────┘  └────┬─────┘  │  happens in  │
               │              │       │  background) │
               └──────┬───────┘       └──────────────┘
                      │
                      ▼
                ┌──────────┐
                │ observe  │
                │  node    │
                └────┬─────┘
                     │
                     ▼
                ┌──────────┐
                │  think   │
                │  (loop)  │
                └──────────┘

Max loops: 3 (hard ceiling to respect LLM rate limits).

Memory writing is NOT a graph node. It runs as a background task in the
route handler after the SSE stream completes, so it never blocks the
response or causes InvalidUpdateError.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.agent.state import VyaparAgentState
from app.agent.nodes.think_node import think_node
from app.agent.nodes.tool_node import tool_node
from app.agent.nodes.observe_node import observe_node
from app.agent.nodes.memory_query import memory_query_node


# ---------------------------------------------------------------------------
# Router functions
# ---------------------------------------------------------------------------

def _route_after_think(state: VyaparAgentState) -> str:
    """
    Conditional edge function called after think_node.

    Priority:
        1. If memory_query_needed → go to memory_query node
        2. If pending_tool_calls → go to tool node
        3. If goal complete → END (memory write happens in background)
        4. Otherwise → END
    """
    # Memory query requested by the think node
    if state.get("memory_query_needed", False):
        return "memory_query"

    # Goal completed — exit graph. Memory persistence is handled
    # by the route handler as a background task after the SSE stream.
    if state.get("goal_status") == "complete":
        return END

    # Agent requested tool calls
    if state.get("pending_tool_calls"):
        return "tool"

    # No tools and goal not explicitly complete
    return END


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """
    Assemble and compile the Vyapar Copilot LangGraph.

    Args:
        checkpointer:
            MongoDB-backed LangGraph checkpointer.
            If None, graph runs without persistence.

    Returns:
        Compiled LangGraph.
    """
    workflow = StateGraph(VyaparAgentState)

    # --------------------------------------------------------------
    # Register nodes
    # --------------------------------------------------------------

    workflow.add_node("think", think_node)
    workflow.add_node("memory_query", memory_query_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("observe", observe_node)

    # --------------------------------------------------------------
    # Entry point
    # --------------------------------------------------------------

    workflow.set_entry_point("think")

    # --------------------------------------------------------------
    # Conditional edges from think
    # --------------------------------------------------------------

    workflow.add_conditional_edges(
        "think",
        _route_after_think,
        {
            "memory_query": "memory_query",
            "tool": "tool",
            "__end__": END,
        },
    )

    # --------------------------------------------------------------
    # memory_query → think (loop back to synthesize memory)
    # --------------------------------------------------------------

    workflow.add_edge("memory_query", "think")

    # --------------------------------------------------------------
    # tool → observe → think (agent loop)
    # --------------------------------------------------------------

    workflow.add_edge("tool", "observe")
    workflow.add_edge("observe", "think")

    # --------------------------------------------------------------
    # Compile graph
    # --------------------------------------------------------------

    return workflow.compile(
        checkpointer=checkpointer
    )