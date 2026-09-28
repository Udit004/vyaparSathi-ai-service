"""
app/agent/tools/web/web_research.py
====================================
High-level web_research tool for the main LangGraph agent.

This single tool encapsulates the entire Web Research Subgraph (Tavily search,
source evaluation, Firecrawl scraping, evidence extraction, and synthesis).
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from app.agent.subgraphs.web_research.graph import web_research_graph


class WebResearchToolInput(BaseModel):
    query: str = Field(
        ...,
        description="The web research query or topic to investigate (e.g., FMCG retail trends in India, competitor prices, supplier updates).",
    )


@tool("web_research", args_schema=WebResearchToolInput)
async def web_research(query: str) -> dict[str, Any]:
    """
    [HIGH-LEVEL WEB RESEARCH TOOL] Conduct comprehensive multi-step web research on internet topics,
    market trends, competitor research, government regulations, or industry updates.

    This tool automatically plans research, searches via Tavily, selects authoritative sources,
    scrapes clean markdown content via Firecrawl, evaluates evidence, and synthesizes structured findings.

    Use when the user asks about external market data, industry trends, recent news, or competitor insights.
    Do NOT use for internal store inventory, sales, or store forecasts (use MongoDB tools for internal data).
    """
    try:
        subgraph_input = {"original_query": query, "max_research_rounds": 2}
        result_state = await web_research_graph.ainvoke(subgraph_input)
        synthesis = result_state.get("synthesis", {})
        if synthesis:
            return synthesis
        return {
            "query": query,
            "answer": "Web research completed, but no synthesis payload was generated.",
            "sources": [],
        }
    except Exception as exc:
        return {
            "query": query,
            "answer": f"Web research failed: {str(exc)}",
            "sources": [],
            "error": str(exc),
        }
