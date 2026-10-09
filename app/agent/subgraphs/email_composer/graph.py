"""
app/agent/subgraphs/email_composer/graph.py
===========================================
Compiled LangGraph StateGraph for Email Composer Subgraph.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.agent.subgraphs.email_composer.state import EmailComposerState
from app.agent.subgraphs.email_composer.nodes import (
    classify_and_enrich_node,
    compose_spec_node,
    render_email_node,
    validate_email_node,
    repair_email_node,
    finalize_email_node,
)


def _route_after_validation(state: EmailComposerState) -> str:
    """
    Conditional edge function after validate_email node.
    - If valid -> finalize_email
    - If invalid & retry_count < max_retries -> repair_email
    - If invalid & retry_count >= max_retries -> finalize_email
    """
    val_res = state.get("validation_result", {})
    is_valid = val_res.get("valid", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if is_valid:
        return "finalize_email"

    if retry_count < max_retries:
        return "repair_email"

    return "finalize_email"


def build_email_composer_graph():
    """
    Assemble and compile the Email Composer Subgraph.
    """
    wf = StateGraph(EmailComposerState)

    # Add Nodes
    wf.add_node("classify_and_enrich", classify_and_enrich_node)
    wf.add_node("compose_spec", compose_spec_node)
    wf.add_node("render_email", render_email_node)
    wf.add_node("validate_email", validate_email_node)
    wf.add_node("repair_email", repair_email_node)
    wf.add_node("finalize_email", finalize_email_node)

    # Entry point
    wf.set_entry_point("classify_and_enrich")

    # Linear edges
    wf.add_edge("classify_and_enrich", "compose_spec")
    wf.add_edge("compose_spec", "render_email")
    wf.add_edge("render_email", "validate_email")

    # Conditional routing after validation
    wf.add_conditional_edges(
        "validate_email",
        _route_after_validation,
        {
            "finalize_email": "finalize_email",
            "repair_email": "repair_email",
        },
    )

    # Edge from repair loops back to compose_spec
    wf.add_edge("repair_email", "compose_spec")

    # Finalize terminates the subgraph
    wf.add_edge("finalize_email", END)

    return wf.compile()


email_composer_graph = build_email_composer_graph()
