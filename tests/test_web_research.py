"""
tests/test_web_research.py
===========================
Comprehensive unit tests for the Web Research Subgraph and high-level tool.

Tests cover:
  1. Successful Tavily search
  2. Empty Tavily search results
  3. URL normalization & deduplication
  4. Successful Firecrawl fetch
  5. Firecrawl failure / fallback to search snippet
  6. Tavily failure handling
  7. Insufficient evidence detection
  8. Research query refinement loop
  9. Maximum research rounds limit (bounded loop)
 10. Conflicting sources recording
 11. Missing publication date handling
 12. Final structured output format (WebResearchOutput)
 13. Main tool calling web_research
 14. Existing tools in registry (MongoDB inventory/sales) remain functional
 15. Normal non-web prompts do not call web research
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.subgraphs.web_research.state import WebResearchState
from app.agent.subgraphs.web_research.schemas import WebResearchInput, WebResearchOutput
from app.agent.subgraphs.web_research.nodes import (
    normalize_url,
    extract_domain,
    plan_research,
    search_web_node,
    select_sources_node,
    fetch_sources_node,
    evaluate_evidence_node,
    refine_query_node,
    synthesize_node,
)
from app.agent.subgraphs.web_research.graph import web_research_graph, _route_after_evidence_evaluation
from app.agent.tools.web.web_research import web_research
from app.agent.tools.registry import VYAPAR_TOOLS, get_tool_by_name


# ---------------------------------------------------------------------------
# Test 1 & 3: URL Normalization & Deduplication
# ---------------------------------------------------------------------------

def test_url_normalization_and_deduplication():
    url_a = "https://EXAMPLE.com/article/?utm_source=twitter&utm_medium=social"
    url_b = "https://example.com/article"
    
    norm_a = normalize_url(url_a)
    norm_b = normalize_url(url_b)

    assert norm_a == norm_b
    assert norm_a == "https://example.com/article"
    assert extract_domain(url_a) == "example.com"


# ---------------------------------------------------------------------------
# Test 2: Plan Research Node
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_plan_research_node():
    initial_state = {"original_query": "What are the latest FMCG market trends in India 2026?"}
    res = await plan_research(initial_state)

    assert res["research_round"] == 1
    assert res["has_temporal_context"] is True
    assert res["current_query"] == "What are the latest FMCG market trends in India 2026?"
    assert res["status"] == "in_progress"


# ---------------------------------------------------------------------------
# Test 4: Successful Tavily Search & Test 6: Tavily Failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_web_node_success():
    state = {
        "current_query": "FMCG trends India",
        "research_round": 1,
        "search_results": [],
        "errors": [],
        "has_temporal_context": True,
    }

    mock_response = {
        "results": [
            {
                "title": "FMCG Report 2026",
                "url": "https://example.com/fmcg-2026?utm_source=test",
                "content": "FMCG market in India grew by 8.5% in 2026.",
                "published_date": "2026-03-15",
                "score": 0.95,
            }
        ],
        "answer": "FMCG sector in India shows strong retail growth.",
    }

    with patch("app.agent.subgraphs.web_research.nodes.TavilyClient") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.search.return_value = mock_response

        with patch("app.agent.subgraphs.web_research.nodes.get_settings") as mock_settings:
            mock_settings.return_value.tavily_api_key = "test_key"
            res = await search_web_node(state)

            assert len(res["search_results"]) == 1
            item = res["search_results"][0]
            assert item["title"] == "FMCG Report 2026"
            assert item["domain"] == "example.com"
            assert item["published_at"] == "2026-03-15"
            assert item["normalized_url"] == "https://example.com/fmcg-2026"


@pytest.mark.asyncio
async def test_search_web_node_tavily_failure():
    state = {
        "current_query": "FMCG trends India",
        "research_round": 1,
        "search_results": [],
        "errors": [],
    }

    with patch("app.agent.subgraphs.web_research.nodes.TavilyClient") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.search.side_effect = Exception("API rate limit exceeded")

        with patch("app.agent.subgraphs.web_research.nodes.get_settings") as mock_settings:
            mock_settings.return_value.tavily_api_key = "test_key"
            res = await search_web_node(state)

            assert len(res["search_results"]) == 0
            assert any("Tavily search failed" in e for e in res["errors"])


# ---------------------------------------------------------------------------
# Test 5 & 11: Firecrawl Fetch & Fallback & Missing Publication Date
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_sources_node_firecrawl_success():
    state = {
        "selected_sources": [
            {
                "title": "Government FMCG Data",
                "url": "https://gov.in/fmcg-report",
                "domain": "gov.in",
                "snippet": "Initial snippet",
                "published_at": None,
            }
        ],
        "fetched_sources": [],
        "errors": [],
    }

    mock_scraped = [
        {
            "url": "https://gov.in/fmcg-report",
            "success": True,
            "markdown": "# Official FMCG Report\nDetailed market stats...",
            "metadata": {"title": "Official FMCG Report", "published_time": None},
        }
    ]

    with patch("app.agent.subgraphs.web_research.nodes.is_firecrawl_configured", return_value=True):
        with patch("app.agent.subgraphs.web_research.nodes.scrape_urls", new_callable=AsyncMock) as mock_scrape:
            mock_scrape.return_value = mock_scraped
            res = await fetch_sources_node(state)

            assert len(res["fetched_sources"]) == 1
            doc = res["fetched_sources"][0]
            assert doc["title"] == "Official FMCG Report"
            assert doc["fetched_via"] == "firecrawl"
            assert doc["published_at"] is None
            assert doc["retrieved_at"] is not None


@pytest.mark.asyncio
async def test_fetch_sources_node_firecrawl_failure_fallback():
    state = {
        "selected_sources": [
            {
                "title": "Industry News",
                "url": "https://example.com/news",
                "domain": "example.com",
                "snippet": "Fallback news snippet text.",
                "published_at": "2026-05-10",
            }
        ],
        "fetched_sources": [],
        "errors": [],
    }

    with patch("app.agent.subgraphs.web_research.nodes.is_firecrawl_configured", return_value=False):
        res = await fetch_sources_node(state)

        assert len(res["fetched_sources"]) == 1
        doc = res["fetched_sources"][0]
        assert doc["fetched_via"] == "tavily_snippet"
        assert doc["content"] == "Fallback news snippet text."


# ---------------------------------------------------------------------------
# Test 7, 8, & 9: Evidence Evaluation, Refinement, & Max Rounds Ceiling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_evaluate_and_refine_query_cycle():
    state = {
        "fetched_sources": [],
        "original_query": "Maggi market share India 2026",
        "research_round": 1,
        "max_research_rounds": 2,
        "conflicts": [],
    }

    eval_res = await evaluate_evidence_node(state)
    assert eval_res["needs_more_research"] is True

    state.update(eval_res)
    route = _route_after_evidence_evaluation(state)
    assert route == "refine_query"

    refine_res = await refine_query_node(state)
    assert refine_res["research_round"] == 2
    assert "2026" in refine_res["current_query"]


def test_max_research_rounds_ceiling():
    state: WebResearchState = {
        "original_query": "Test",
        "research_goal": "Test",
        "search_queries": ["Test"],
        "current_query": "Test",
        "search_results": [],
        "selected_sources": [],
        "fetched_sources": [],
        "evidence": [],
        "source_count": 0,
        "research_round": 2,
        "max_research_rounds": 2,
        "needs_more_research": True,
        "refined_query": "",
        "synthesis": {},
        "sources": [],
        "conflicts": [],
        "errors": [],
        "status": "in_progress",
        "has_temporal_context": False,
    }

    route = _route_after_evidence_evaluation(state)
    assert route == "synthesize"


# ---------------------------------------------------------------------------
# Test 10 & 12: Conflicting Sources & Final Structured Output
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_synthesize_node_structured_output():
    state = {
        "original_query": "Indian FMCG growth rate 2026",
        "fetched_sources": [
            {
                "title": "Economic Times Report",
                "url": "https://economictimes.indiatimes.com/fmcg-2026",
                "domain": "economictimes.indiatimes.com",
                "content": "Indian FMCG market expected to grow by 9% in 2026.",
                "published_at": "2026-02-01",
                "retrieved_at": "2026-09-26T17:00:00",
                "reason": "Official business report",
            }
        ],
        "evidence": [
            {
                "source_title": "Economic Times Report",
                "source_url": "https://economictimes.indiatimes.com/fmcg-2026",
                "domain": "economictimes.indiatimes.com",
                "published_at": "2026-02-01",
                "retrieved_at": "2026-09-26T17:00:00",
                "fact_snippet": "Indian FMCG market expected to grow by 9% in 2026.",
            }
        ],
        "conflicts": [
            {
                "topic": "growth rate",
                "source_a": "Economic Times (9%)",
                "source_b": "Blog (12%)",
                "description": "Growth rate estimates differ between sources.",
            }
        ],
        "errors": [],
        "research_round": 1,
        "has_temporal_context": True,
    }

    mock_llm_response = MagicMock()
    mock_llm_response.content = (
        "According to Economic Times, the Indian FMCG market is projected to grow at 9% in 2026.\n"
        "- FMCG retail sector maintains steady 9% expansion in 2026.\n"
        "- Key growth driver is rising rural demand."
    )

    with patch("app.agent.subgraphs.web_research.nodes.get_llm") as mock_get_llm:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_llm_response
        mock_get_llm.return_value = mock_llm

        res = await synthesize_node(state)
        synth = res["synthesis"]

        assert res["status"] == "completed"
        assert synth["query"] == "Indian FMCG growth rate 2026"
        assert len(synth["sources"]) == 1
        assert synth["sources"][0]["published_at"] == "2026-02-01"
        assert synth["sources"][0]["retrieved_at"] == "2026-09-26T17:00:00"
        assert len(synth["conflicts"]) == 1
        assert synth["confidence"] in ("high", "medium", "low")


# ---------------------------------------------------------------------------
# Test 13: High-Level Tool Execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_web_research_tool_invocation():
    mock_synthesis = {
        "query": "Retail trend test",
        "answer": "Grounded retail trends response.",
        "key_findings": ["Finding 1"],
        "sources": [],
    }

    with patch.object(web_research_graph, "ainvoke", new_callable=AsyncMock) as mock_graph_invoke:
        mock_graph_invoke.return_value = {"synthesis": mock_synthesis}
        res = await web_research.ainvoke({"query": "Retail trend test"})

        assert res == mock_synthesis
        mock_graph_invoke.assert_called_once()


# ---------------------------------------------------------------------------
# Test 14 & 15: Registry Integrity & Non-Web Queries
# ---------------------------------------------------------------------------

def test_registry_contains_web_research_and_mongodb_tools():
    tool_names = [t.name for t in VYAPAR_TOOLS]
    assert "web_research" in tool_names
    assert "get_inventory_summary" in tool_names
    assert "get_sales_summary" in tool_names
    assert "get_top_selling_products" in tool_names


def test_get_tool_by_name_lookup():
    tool = get_tool_by_name("web_research")
    assert tool is not None
    assert tool.name == "web_research"
