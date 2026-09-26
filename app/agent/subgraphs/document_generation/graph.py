"""
app/agent/subgraphs/document_generation/graph.py
=================================================
Compiled LangGraph StateGraph for Document Generation Subgraph.
"""

from __future__ import annotations

from typing import Any
from langgraph.graph import StateGraph, END

from app.agent.subgraphs.document_generation.state import DocumentGenerationState
from app.agent.subgraphs.document_generation.nodes import (
    prepare_document_node,
    build_document_spec_node,
    generate_document_node,
    validate_document_node,
    repair_document_node,
    upload_file_node,
    create_result_node,
)


def _route_after_validation(state: DocumentGenerationState) -> str:
    """
    Conditional edge function after validate_document node.

    - If valid -> upload_file
    - If invalid & retry_count < max_retries -> repair_document
    - If invalid & retry_count >= max_retries -> create_result (stopping failure)
    """
    val_res = state.get("validation_result", {})
    is_valid = val_res.get("valid", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if is_valid:
        return "upload_file"

    if retry_count < max_retries:
        return "repair_document"

    return "create_result"


def build_document_generation_graph():
    """
    Assemble and compile the Document Generation Subgraph.
    """
    wf = StateGraph(DocumentGenerationState)

    # Add Nodes
    wf.add_node("prepare_document", prepare_document_node)
    wf.add_node("build_document_spec", build_document_spec_node)
    wf.add_node("generate_document", generate_document_node)
    wf.add_node("validate_document", validate_document_node)
    wf.add_node("repair_document", repair_document_node)
    wf.add_node("upload_file", upload_file_node)
    wf.add_node("create_result", create_result_node)

    # Entry point
    wf.set_entry_point("prepare_document")

    # Linear edges
    wf.add_edge("prepare_document", "build_document_spec")
    wf.add_edge("build_document_spec", "generate_document")
    wf.add_edge("generate_document", "validate_document")

    # Conditional router after validation
    wf.add_conditional_edges(
        "validate_document",
        _route_after_validation,
        {
            "upload_file": "upload_file",
            "repair_document": "repair_document",
            "create_result": "create_result",
        },
    )

    # Edge from repair loops back to build_document_spec
    wf.add_edge("repair_document", "build_document_spec")

    # Upload file proceeds to create_result
    wf.add_edge("upload_file", "create_result")

    # Create result terminates the subgraph
    wf.add_edge("create_result", END)

    return wf.compile()


document_generation_graph = build_document_generation_graph()
