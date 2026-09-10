"""
app/agent/graph.py
==================
LangGraph graph assembly for the Vyapar Copilot agent.

Builds the StateGraph with four nodes:
    think   → LLM chooses tools or marks goal complete
    tool    → executes tool calls, fills context buckets
    observe → converts tool results to ToolMessages, increments loop count
    respond → synthesizes all gathered data into the final answer

Loop routing (after think node):
    ┌──────────┐     has tools?     ┌──────────┐
    │  think   │─── YES ──────────► │   tool   │
    └──────────┘                    └────┬─────┘
         ▲                               │
         │                              ▼
         │                         ┌──────────┐
         └──────── loop back ──────│  observe │
                                   └──────────┘
    ┌──────────┐
    │  think   │─── complete / max_loops ──► respond ──► END
    └──────────┘
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
try:
    # langgraph-checkpoint-mongodb >= 0.2.0 (Render, PyPI latest)
    from langgraph_checkpoint_mongodb import AsyncMongoDBSaver
except ImportError:
    # langgraph-checkpoint-mongodb <= 0.1.x (legacy local install)
    from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver  # type: ignore[no-redef]

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

    Returns:
        "tool"    → if the LLM requested tool calls
        "__end__" → if the goal is complete, failed, or loop ceiling hit
    """
    # Goal marked complete by LLM
    if state.get("goal_status") == "complete":
        return "__end__"

    # LLM requested tools
    if state.get("pending_tool_calls"):
        return "tool"

    # Fallback: no tools called but goal not explicitly complete → end
    return "__end__"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(checkpointer: AsyncMongoDBSaver | None = None) -> StateGraph:
    """
    Assemble and compile the Vyapar Copilot LangGraph.

    Args:
        checkpointer: An initialized AsyncMongoDBSaver instance for
                      cross-session persistence. If None, the graph runs
                      without persistence (useful for unit tests).

    Returns:
        A compiled CompiledGraph ready for .ainvoke() / .astream().
    """
    workflow = StateGraph(VyaparAgentState)

    # ------------------------------------------------------------------
    # Register nodes
    # ------------------------------------------------------------------
    workflow.add_node("think", think_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("observe", observe_node)

    # ------------------------------------------------------------------
    # Set entry point
    # ------------------------------------------------------------------
    workflow.set_entry_point("think")

    # ------------------------------------------------------------------
    # Conditional routing from think
    # ------------------------------------------------------------------
    workflow.add_conditional_edges(
        "think",
        _route_after_think,
        {
            "tool": "tool",
            "__end__": END,
        },
    )

    # ------------------------------------------------------------------
    # Inner loop: tool → observe → think
    # ------------------------------------------------------------------
    workflow.add_edge("tool", "observe")
    workflow.add_edge("observe", "think")

    # ------------------------------------------------------------------
    # Compile with optional checkpointer
    # ------------------------------------------------------------------
    return workflow.compile(checkpointer=checkpointer)
