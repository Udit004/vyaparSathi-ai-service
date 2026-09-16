"""
app/agent/graph.py
==================

LangGraph graph assembly for the Vyapar Copilot agent.

Graph flow
----------
              ┌──────────────────┐
              │   grader node     │  ← entry point (guardrail)
              │  (safety check)   │      harmful prompt → END (refusal)
              └──────┬───────────┘      safe prompt      → think
                     │
                     ▼
              ┌──────────────┐
              │    think node │
              │  (LLM reasoning)  │
              └──────┬───────────┘
                     │
          ┌──────────┼──────────┐
          │          │          │
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
from app.agent.nodes.grader_node import grader_node


# ---------------------------------------------------------------------------
# Router functions
# ---------------------------------------------------------------------------

def _route_after_grader(state: VyaparAgentState) -> str:
    """
    Conditional edge function called after grader_node (the entry point).

    The grader sets ``grader_denied`` when the user prompt was classified as
    harmful. In that case the graph terminates immediately — the refusal is
    already stored in ``final_answer``. Otherwise the graph proceeds to the
    normal think node.
    """
    if state.get("grader_denied", False):
        return END
    return "think"


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

    workflow.add_node("grader", grader_node)
    workflow.add_node("think", think_node)
    workflow.add_node("memory_query", memory_query_node)
    workflow.add_node("tool", tool_node)
    workflow.add_node("observe", observe_node)

    # --------------------------------------------------------------
    # Entry point — guardrail runs first to screen the user prompt
    # --------------------------------------------------------------

    workflow.set_entry_point("grader")

    # --------------------------------------------------------------
    # Conditional edges from grader
    # Denied → END (refusal already in final_answer)
    # Safe  → think
    # --------------------------------------------------------------

    workflow.add_conditional_edges(
        "grader",
        _route_after_grader,
        {
            "__end__": END,
            "think": "think",
        },
    )

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