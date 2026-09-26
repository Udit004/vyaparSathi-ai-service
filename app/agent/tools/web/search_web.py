from __future__ import annotations

from typing import Any, List

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from app.config.settings import get_settings


class SearchWebInput(BaseModel):
    query: str = Field(
        ...,
        description="The web search query to run. Keep it concise and specific.",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of search results to return. Capped at 10.",
    )
    include_answer: bool = Field(
        default=True,
        description="Whether to include a direct answer synthesized by Tavily.",
    )


@tool("search_web", args_schema=SearchWebInput)
def search_web(query: str, max_results: int = 5, include_answer: bool = True) -> dict[str, Any]:
    """
    Search the web using Tavily and return relevant results and an optional answer.

    Use this tool when the question needs current information from the internet,
    competitor research, product facts, or recent updates not present in local
    app data.
    """
    settings = get_settings()
    api_key = settings.tavily_api_key or settings.__dict__.get("TAVILY_API_KEY") or None
    if not api_key:
        import os
        api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set. Add it to the environment or .env file.")

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        include_answer=include_answer,
        search_depth="basic",
    )

    results = response.get("results", []) if isinstance(response, dict) else []
    answer = response.get("answer") if isinstance(response, dict) else None

    output: dict[str, Any] = {
        "query": query,
        "results": results,
    }
    if include_answer:
        output["answer"] = answer
    return output
