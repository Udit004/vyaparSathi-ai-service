"""
app/agent/graph.py
==================

LangGraph graph assembly for the Vyapar Copilot agent.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

from app.agent.state import VyaparAgentState
from app.agent.nodes.think_node import think_node
from app.agent.nodes.tool_node import tool_node
from app.agent.nodes.observe_node import observe_node


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def _route_after_think(state: VyaparAgentState) -> str:
    """
    Conditional edge function called after think_node.
    """

    # Goal completed
    if state.get("goal_status") == "complete":
        return "__end__"

    # Agent requested tool calls
    if state.get("pending_tool_calls"):
        return "tool"

    # No tools and goal not explicitly complete
    return "__end__"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(checkpointer: MongoDBSaver | AsyncMongoDBSaver | None = None):
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
    workflow.add_node("tool", tool_node)
    workflow.add_node("observe", observe_node)

    # --------------------------------------------------------------
    # Entry point
    # --------------------------------------------------------------

    workflow.set_entry_point("think")

    # --------------------------------------------------------------
    # Routing
    # --------------------------------------------------------------

    workflow.add_conditional_edges(
        "think",
        _route_after_think,
        {
            "tool": "tool",
            "__end__": END,
        },
    )

    # --------------------------------------------------------------
    # Agent loop
    # --------------------------------------------------------------

    workflow.add_edge("tool", "observe")

    workflow.add_edge(
        "observe",
        "think",
    )

    # --------------------------------------------------------------
    # Compile graph
    # --------------------------------------------------------------

    return workflow.compile(
        checkpointer=checkpointer
    )
