"""
app/agent/subgraphs/web_research/schemas.py
============================================
Pydantic schemas for the Web Research Subgraph input and output contracts.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class WebResearchInput(BaseModel):
    """Input payload for the web_research subgraph."""
    query: str = Field(..., description="The user query or research prompt.")
    max_rounds: int = Field(2, ge=1, le=3, description="Maximum research refinement loops allowed (1-3).")


class WebResearchOutput(BaseModel):
    """Structured output returned by the web_research subgraph."""
    query: str = Field(..., description="Original user research query.")
    answer: str = Field(..., description="Comprehensive, grounded research answer synthesized from fetched evidence.")
    key_findings: list[str] = Field(default_factory=list, description="Bullet points of key evidence gathered.")
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of evaluated/fetched source metadata. Each: {title, url, domain, published_at, retrieved_at, reason}."
    )
    confidence: str = Field("medium", description="Confidence level of the research: 'high' | 'medium' | 'low'.")
    freshness: str = Field("", description="Freshness evaluation summary (e.g. 'Current as of September 2026').")
    conflicts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recorded factual disagreements between sources. Each: {topic, source_a, source_b, description}."
    )
    research_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Audit metadata: {rounds_taken, total_sources_evaluated, total_pages_fetched, errors, search_provider}."
    )
