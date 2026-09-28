"""
app/agent/subgraphs/web_research/graph.py
==========================================
LangGraph assembly for the Web Research Subgraph.

Workflow:
  START -> plan_research -> search_web -> select_sources -> fetch_sources -> evaluate_evidence
  evaluate_evidence -> (if needs_more_research & round < max_rounds) -> refine_query -> search_web
  evaluate_evidence -> (else) -> synthesize -> END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.agent.subgraphs.web_research.state import WebResearchState
from app.agent.subgraphs.web_research.nodes import (
    plan_research,
    search_web_node,
    select_sources_node,
    fetch_sources_node,
    evaluate_evidence_node,
    refine_query_node,
    synthesize_node,
)


def _route_after_evidence_evaluation(state: WebResearchState) -> str:
    """
    Decide whether to refine query and run another research round or proceed to synthesis.
    """
    needs_more = state.get("needs_more_research", False)
    curr_round = state.get("research_round", 1)
    max_rounds = state.get("max_research_rounds", 2)

    if needs_more and curr_round < max_rounds:
        return "refine_query"
    return "synthesize"


def build_web_research_graph():
    """Build and compile the web research StateGraph."""
    workflow = StateGraph(WebResearchState)

    # Register nodes
    workflow.add_node("plan_research", plan_research)
    workflow.add_node("search_web", search_web_node)
    workflow.add_node("select_sources", select_sources_node)
    workflow.add_node("fetch_sources", fetch_sources_node)
    workflow.add_node("evaluate_evidence", evaluate_evidence_node)
    workflow.add_node("refine_query", refine_query_node)
    workflow.add_node("synthesize", synthesize_node)

    # Set entry point
    workflow.set_entry_point("plan_research")

    # Fixed edges
    workflow.add_edge("plan_research", "search_web")
    workflow.add_edge("search_web", "select_sources")
    workflow.add_edge("select_sources", "fetch_sources")
    workflow.add_edge("fetch_sources", "evaluate_evidence")

    # Conditional routing after evidence evaluation
    workflow.add_conditional_edges(
        "evaluate_evidence",
        _route_after_evidence_evaluation,
        {
            "refine_query": "refine_query",
            "synthesize": "synthesize",
        }
    )

    workflow.add_edge("refine_query", "search_web")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


web_research_graph = build_web_research_graph()
