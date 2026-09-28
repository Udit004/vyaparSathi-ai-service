"""
app/agent/subgraphs/web_research/state.py
==========================================
WebResearchState — state definition for the Web Research Subgraph.
"""
from __future__ import annotations

from typing import Any, TypedDict


class WebResearchState(TypedDict):
    """
    Isolated state for the Web Research Subgraph workflow.
    """
    original_query: str
    research_goal: str
    search_queries: list[str]
    current_query: str
    search_results: list[dict[str, Any]]
    selected_sources: list[dict[str, Any]]
    fetched_sources: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    source_count: int
    research_round: int
    max_research_rounds: int
    needs_more_research: bool
    refined_query: str
    synthesis: dict[str, Any]
    sources: list[dict[str, Any]]
    conflicts: list[dict[str, Any]]
    errors: list[str]
    status: str
    has_temporal_context: bool
