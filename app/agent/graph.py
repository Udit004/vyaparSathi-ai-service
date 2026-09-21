"""
app/agent/graph.py
==================

LangGraph graph assembly for the Vyapar Copilot agent.

Graph flow
----------
              ┌──────────────────┐
              │   grader node     │  ← entry point (guardrail)
              │  (safety check)   │      harmful prompt → END (refusal)
              └──────┬───────────┘      safe prompt      → intent
                     │
                     ▼
              ┌──────────────┐
              │  intent node  │
              │  (routing)    │
              └──────┬───────────┘
                     │
                     ▼
              ┌──────────────┐
              │    think node │
              │  (LLM reasoning)  │
              └──────┬───────────┘
                     │
        ┌────────────┼────────────────┐
        │            │                │
    interrupt   memory_query    pending_tool  goal_status
    needed?       calls?         == "complete"
         │              │                │
         ▼              ▼                ▼
  ┌──────────┐  ┌──────────┐  ┌──────────────┐
  │ interrupt│  │ memory   │  │     END      │
  │  node    │  │  query   │  │ (memory write│
  │          │  │          │  │  in background)│
  └─────┬────┘  └────┬─────┘  └──────────────┘
        │             │
        ▼             ▼
      END          think (loop)

Interrupt path:
    think → interrupt → think (loop)
    The interrupt node calls langgraph.types.interrupt() to pause the
    graph. The clarification question is surfaced to the caller via
    __interrupt__ in the stream. When the user responds, the caller
    resumes with Command(resume=<answer>) using the same thread_id.
    The interrupt() call returns the answer, it is stored in
    clarification_history, and the graph loops back to think.

Max loops: 3 (hard ceiling to respect LLM rate limits).
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.agent.state import VyaparAgentState
from app.agent.nodes.think_node import think_node
from app.agent.nodes.tool_node import tool_node
from app.agent.nodes.observe_node import observe_node
from app.agent.nodes.memory_query import memory_query_node
from app.agent.nodes.grader_node import grader_node
from app.agent.nodes.intent_node import intent_node
from app.agent.nodes.interrupt_node import interrupt_node


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
    return "intent"


def _route_after_think(state: VyaparAgentState) -> str:
    """
    Conditional edge function called after think_node.

    Priority (highest to lowest):
        1. If needs_clarification → go to interrupt node
        2. If memory_query_needed → go to memory_query node
        3. If pending_tool_calls → go to tool node
        4. If goal complete → END (memory write happens in background)
        5. Otherwise → END
    """
    # Clarification takes highest priority — the agent is uncertain
    if state.get("needs_clarification", False):
        return "interrupt"

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
    workflow.add_node("intent", intent_node)
    workflow.add_node("interrupt", interrupt_node)
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
            "intent": "intent",
        },
    )

    workflow.add_edge("intent", "think")

    # --------------------------------------------------------------
    # Conditional edges from think
    # interrupt has highest priority (checked in router)
    # --------------------------------------------------------------

    workflow.add_conditional_edges(
        "think",
        _route_after_think,
        {
            "interrupt": "interrupt",
            "memory_query": "memory_query",
            "tool": "tool",
            "__end__": END,
        },
    )

    # --------------------------------------------------------------
    # interrupt → think (graph pauses at interrupt() inside the node;
    # when resumed via Command(resume=...), the interrupt node stores
    # the user's answer in clarification_history and the graph loops
    # back to think so the LLM can incorporate the answer and continue)
    # --------------------------------------------------------------

    workflow.add_edge("interrupt", "think")

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