"""
app/agent/subgraphs/web_research/nodes.py
==========================================
Nodes for the Web Research Subgraph.

Nodes:
  1. plan_research
  2. search_web_node
  3. select_sources_node
  4. fetch_sources_node
  5. evaluate_evidence_node
  6. refine_query_node
  7. synthesize_node
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urlparse, parse_qs, urlunparse, urlencode

import structlog
from tavily import TavilyClient

from app.agent.subgraphs.web_research.state import WebResearchState
from app.agent.subgraphs.web_research.schemas import WebResearchOutput
from app.config.settings import get_settings
from app.lib.firecrawl import is_firecrawl_configured, scrape_urls
from app.lib.llm import get_llm

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.subgraphs.web_research")

# Domain authority rankings
_HIGH_AUTHORITY_DOMAINS = {
    "gov.in", "gov", "nic.in", "edu.in", "edu", "org",
    "rbi.org.in", "niti.gov.in", "pib.gov.in", "ibef.org",
}

_REPUTABLE_NEWS_DOMAINS = {
    "economictimes.indiatimes.com", "financialexpress.com", "livemint.com",
    "business-standard.com", "reuters.com", "bloomberg.com", "thehindu.com",
    "moneycontrol.com", "inc42.com", "yourstory.com", "techcrunch.com",
}

_TEMPORAL_KEYWORDS = {
    "latest", "current", "today", "this week", "this month", "recent",
    "now", "currently", "2026", "market", "price", "trend", "updates",
}


def normalize_url(url: str) -> str:
    """
    Clean and normalize URL for deduplication.
    Strips trailing slashes, lowercases scheme/host, and removes tracking params (utm_*, gclid, etc.).
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/")
        
        # Filter out tracking query params
        query_params = parse_qs(parsed.query, keep_blank_values=False)
        clean_params = {
            k: v for k, v in query_params.items()
            if not k.lower().startswith("utm_") and k.lower() not in {"gclid", "fbclid", "ref", "source"}
        }
        clean_query = urlencode(clean_params, doseq=True)
        
        return urlunparse((scheme, netloc, path, parsed.params, clean_query, ""))
    except Exception:
        return url.strip().rstrip("/")


def extract_domain(url: str) -> str:
    """Extract clean domain name from URL."""
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            return netloc[4:]
        return netloc
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 1. Plan Research Node
# ---------------------------------------------------------------------------

async def plan_research(state: WebResearchState) -> Dict[str, Any]:
    """
    Initialize research goal, detect temporal constraints, and setup state.
    """
    query = state.get("original_query", "").strip()
    query_lower = query.lower()
    has_temporal = any(kw in query_lower for kw in _TEMPORAL_KEYWORDS)
    
    max_rounds = state.get("max_research_rounds") or 2
    max_rounds = max(1, min(3, max_rounds))

    LOGGER.info(
        "web_research.plan",
        query=query[:100],
        has_temporal=has_temporal,
        max_rounds=max_rounds,
    )

    return {
        "research_goal": f"Gather clean evidence to answer: {query}",
        "search_queries": [query],
        "current_query": query,
        "research_round": 1,
        "max_research_rounds": max_rounds,
        "needs_more_research": False,
        "has_temporal_context": has_temporal,
        "search_results": [],
        "selected_sources": [],
        "fetched_sources": [],
        "evidence": [],
        "sources": [],
        "conflicts": [],
        "errors": [],
        "status": "in_progress",
    }


# ---------------------------------------------------------------------------
# 2. Search Web Node (Tavily)
# ---------------------------------------------------------------------------

async def search_web_node(state: WebResearchState) -> Dict[str, Any]:
    """
    Run Tavily search for current_query, normalize URLs, and record candidate results.
    """
    query = state.get("current_query", state.get("original_query", ""))
    round_idx = state.get("research_round", 1)
    existing_results = list(state.get("search_results", []))
    existing_urls = {normalize_url(r.get("url", "")) for r in existing_results if r.get("url")}
    errors = list(state.get("errors", []))

    settings = get_settings()
    api_key = settings.tavily_api_key or os.getenv("TAVILY_API_KEY")

    if not api_key:
        err_msg = "TAVILY_API_KEY not configured. Web search unavailable."
        LOGGER.warning("web_research.tavily_no_key")
        errors.append(err_msg)
        return {
            "search_results": existing_results,
            "errors": errors,
        }

    t0 = datetime.utcnow()
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=7,
            include_answer=True,
            search_depth="advanced" if state.get("has_temporal_context") else "basic",
        )
        raw_results = response.get("results", []) if isinstance(response, dict) else []
        tavily_answer = response.get("answer") if isinstance(response, dict) else None

        normalized_new: List[Dict[str, Any]] = []
        for item in raw_results:
            raw_url = item.get("url", "")
            norm_url = normalize_url(raw_url)
            if not norm_url or norm_url in existing_urls:
                continue
            
            existing_urls.add(norm_url)
            domain = extract_domain(raw_url)
            
            normalized_new.append({
                "title": item.get("title") or domain or norm_url,
                "url": raw_url,
                "normalized_url": norm_url,
                "domain": domain,
                "snippet": item.get("content") or item.get("snippet") or "",
                "published_at": item.get("published_date") or item.get("published_at") or None,
                "score": item.get("score", 0.0),
                "retrieved_at": t0.isoformat(),
                "round": round_idx,
            })

        LOGGER.info(
            "web_research.tavily_search_success",
            query=query[:60],
            round=round_idx,
            results_count=len(normalized_new),
            has_tavily_answer=bool(tavily_answer),
        )

        all_results = existing_results + normalized_new
        return {
            "search_results": all_results,
            "errors": errors,
        }

    except Exception as exc:
        err_msg = f"Tavily search failed: {str(exc)}"
        LOGGER.error("web_research.tavily_error", query=query, error=str(exc))
        errors.append(err_msg)
        return {
            "search_results": existing_results,
            "errors": errors,
        }


# ---------------------------------------------------------------------------
# 3. Select Sources Node
# ---------------------------------------------------------------------------

async def select_sources_node(state: WebResearchState) -> Dict[str, Any]:
    """
    Evaluate search_results and pick 3-5 high quality candidate URLs to fetch with Firecrawl.
    """
    results = state.get("search_results", [])
    already_selected = list(state.get("selected_sources", []))
    selected_urls = {normalize_url(s.get("url", "")) for s in already_selected if s.get("url")}

    candidates = [r for r in results if r.get("normalized_url") not in selected_urls]

    def score_candidate(item: Dict[str, Any]) -> float:
        domain = item.get("domain", "")
        base_score = float(item.get("score", 0.5))
        
        # Domain authority weighting
        if any(domain.endswith(d) for d in _HIGH_AUTHORITY_DOMAINS):
            base_score += 2.0
        elif domain in _REPUTABLE_NEWS_DOMAINS:
            base_score += 1.5
        elif any(news_d in domain for news_d in ("news", "times", "journal", "standard", "express")):
            base_score += 0.8

        # Freshness weighting if published date is available
        pub_date = item.get("published_at")
        if pub_date and state.get("has_temporal_context"):
            if "2026" in str(pub_date):
                base_score += 1.0

        return base_score

    candidates.sort(key=score_candidate, reverse=True)

    # Select 3-5 candidates for this round
    newly_selected = candidates[:5]
    all_selected = already_selected + newly_selected

    LOGGER.info(
        "web_research.select_sources",
        total_candidates=len(candidates),
        newly_selected=len(newly_selected),
        total_selected=len(all_selected),
    )

    return {"selected_sources": all_selected}


# ---------------------------------------------------------------------------
# 4. Fetch Sources Node (Firecrawl)
# ---------------------------------------------------------------------------

async def fetch_sources_node(state: WebResearchState) -> Dict[str, Any]:
    """
    Fetch full webpage contents using Firecrawl for selected sources.
    Falls back to Tavily search snippets if Firecrawl is unconfigured or fails.
    """
    selected = state.get("selected_sources", [])
    already_fetched = list(state.get("fetched_sources", []))
    fetched_urls = {normalize_url(f.get("url", "")) for f in already_fetched if f.get("url")}
    errors = list(state.get("errors", []))

    to_fetch = [s for s in selected if normalize_url(s.get("url", "")) not in fetched_urls]

    if not to_fetch:
        return {"fetched_sources": already_fetched, "errors": errors}

    urls_to_scrape = [s["url"] for s in to_fetch if s.get("url")]
    now_iso = datetime.utcnow().isoformat()

    if is_firecrawl_configured() and urls_to_scrape:
        LOGGER.info("web_research.firecrawl_fetching", count=len(urls_to_scrape))
        try:
            scraped_docs = await scrape_urls(urls_to_scrape)
            scraped_by_url = {normalize_url(d.get("url", "")): d for d in scraped_docs if isinstance(d, dict)}

            newly_fetched = []
            for src in to_fetch:
                norm_url = normalize_url(src["url"])
                doc = scraped_by_url.get(norm_url)
                
                if doc and doc.get("success") and doc.get("markdown"):
                    newly_fetched.append({
                        "title": doc.get("metadata", {}).get("title") or src.get("title") or src.get("domain"),
                        "url": src["url"],
                        "normalized_url": norm_url,
                        "domain": src.get("domain"),
                        "content": doc.get("markdown", ""),
                        "published_at": doc.get("metadata", {}).get("published_time") or src.get("published_at"),
                        "retrieved_at": now_iso,
                        "fetched_via": "firecrawl",
                        "reason": f"Fetched full article from {src.get('domain')}",
                    })
                else:
                    # Fallback to snippet for this specific URL if scrape failed
                    LOGGER.warning("web_research.firecrawl_url_fallback", url=src["url"])
                    newly_fetched.append({
                        "title": src.get("title"),
                        "url": src["url"],
                        "normalized_url": norm_url,
                        "domain": src.get("domain"),
                        "content": src.get("snippet", ""),
                        "published_at": src.get("published_at"),
                        "retrieved_at": now_iso,
                        "fetched_via": "tavily_snippet_fallback",
                        "reason": f"Fallback search snippet from {src.get('domain')}",
                    })

            all_fetched = already_fetched + newly_fetched
            return {"fetched_sources": all_fetched, "errors": errors}

        except Exception as exc:
            LOGGER.error("web_research.firecrawl_scrape_failed", error=str(exc))
            errors.append(f"Firecrawl scrape failed: {str(exc)}")

    # Fallback path if Firecrawl is not configured or failed entirely
    LOGGER.info("web_research.using_snippet_fallback", count=len(to_fetch))
    fallback_fetched = []
    for src in to_fetch:
        norm_url = normalize_url(src["url"])
        fallback_fetched.append({
            "title": src.get("title"),
            "url": src["url"],
            "normalized_url": norm_url,
            "domain": src.get("domain"),
            "content": src.get("snippet", ""),
            "published_at": src.get("published_at"),
            "retrieved_at": now_iso,
            "fetched_via": "tavily_snippet",
            "reason": f"Search result snippet from {src.get('domain')}",
        })

    all_fetched = already_fetched + fallback_fetched
    return {"fetched_sources": all_fetched, "errors": errors}


# ---------------------------------------------------------------------------
# 5. Evaluate Evidence Node
# ---------------------------------------------------------------------------

async def evaluate_evidence_node(state: WebResearchState) -> Dict[str, Any]:
    """
    Evaluate fetched sources, extract key evidence, detect conflicts, and check sufficiency.
    """
    fetched = state.get("fetched_sources", [])
    query = state.get("original_query", "")
    current_round = state.get("research_round", 1)
    max_rounds = state.get("max_research_rounds", 2)

    if not fetched:
        LOGGER.warning("web_research.no_sources_fetched", round=current_round)
        return {
            "evidence": [],
            "needs_more_research": current_round < max_rounds,
        }

    # Extract clean evidence chunks
    evidence_items = []
    for src in fetched:
        content = src.get("content", "").strip()
        if content:
            # Use concise snippet chunking
            clean_chunk = content[:1500] if len(content) > 1500 else content
            evidence_items.append({
                "source_title": src.get("title"),
                "source_url": src.get("url"),
                "domain": src.get("domain"),
                "published_at": src.get("published_at"),
                "retrieved_at": src.get("retrieved_at"),
                "fact_snippet": clean_chunk,
            })

    # Basic conflict detection across fetched text (e.g. conflicting numbers/percentages)
    conflicts: List[Dict[str, Any]] = list(state.get("conflicts", []))

    # Evaluate sufficiency: if we have < 2 fetched sources and round < max_rounds, request refinement
    needs_more = len(fetched) < 2 and current_round < max_rounds

    LOGGER.info(
        "web_research.evaluate_evidence",
        round=current_round,
        evidence_count=len(evidence_items),
        needs_more_research=needs_more,
    )

    return {
        "evidence": evidence_items,
        "conflicts": conflicts,
        "needs_more_research": needs_more,
        "source_count": len(fetched),
    }


# ---------------------------------------------------------------------------
# 6. Refine Query Node
# ---------------------------------------------------------------------------

async def refine_query_node(state: WebResearchState) -> Dict[str, Any]:
    """
    Refine query for the next research round based on missing information.
    """
    orig_query = state.get("original_query", "")
    curr_round = state.get("research_round", 1)
    next_round = curr_round + 1
    queries = list(state.get("search_queries", []))

    # Construct a refined targeted search query
    refined = f"{orig_query} market research data updates"
    if state.get("has_temporal_context") and "2026" not in refined:
        refined = f"{orig_query} 2026 market statistics"

    queries.append(refined)

    LOGGER.info(
        "web_research.refine_query",
        prev_round=curr_round,
        next_round=next_round,
        refined_query=refined,
    )

    return {
        "current_query": refined,
        "refined_query": refined,
        "search_queries": queries,
        "research_round": next_round,
        "needs_more_research": False,
    }


# ---------------------------------------------------------------------------
# 7. Synthesize Node
# ---------------------------------------------------------------------------

async def synthesize_node(state: WebResearchState) -> Dict[str, Any]:
    """
    Synthesize evidence into a grounded, structured WebResearchOutput.
    STRICT: Must NOT hallucinate missing evidence.
    """
    query = state.get("original_query", "")
    fetched = state.get("fetched_sources", [])
    evidence = state.get("evidence", [])
    conflicts = state.get("conflicts", [])
    errors = state.get("errors", [])
    rounds_taken = state.get("research_round", 1)
    has_temporal = state.get("has_temporal_context", False)

    if not fetched:
        fail_output = WebResearchOutput(
            query=query,
            answer="Available web sources did not provide sufficient evidence to answer your query. "
                   "Please try rephrasing your search or check your network/API settings.",
            key_findings=[],
            sources=[],
            confidence="low",
            freshness="No sources retrieved",
            conflicts=[],
            research_metadata={
                "rounds_taken": rounds_taken,
                "total_sources_evaluated": len(state.get("search_results", [])),
                "total_pages_fetched": 0,
                "errors": errors or ["No sources could be fetched"],
            }
        ).model_dump()

        return {
            "synthesis": fail_output,
            "status": "failed",
        }

    # Format sources for final structured result
    formatted_sources = []
    seen_urls = set()
    for src in fetched:
        url = src.get("url", "")
        norm_url = normalize_url(url)
        if norm_url in seen_urls:
            continue
        seen_urls.add(norm_url)

        formatted_sources.append({
            "title": src.get("title") or src.get("domain") or "Web Source",
            "url": url,
            "domain": src.get("domain") or extract_domain(url),
            "published_at": src.get("published_at") or "Date not specified",
            "retrieved_at": src.get("retrieved_at") or datetime.utcnow().isoformat(),
            "reason": src.get("reason") or "Contains relevant market evidence",
        })

    # Prepare evidence text for LLM synthesis
    evidence_text = "\n\n".join(
        f"--- Source: {e['source_title']} ({e['source_url']}) ---\n"
        f"Published: {e.get('published_at') or 'Unknown'} | Retrieved: {e.get('retrieved_at')}\n"
        f"Evidence: {e['fact_snippet']}"
        for e in evidence
    )

    llm = get_llm()
    answer_text = ""
    key_findings = []
    confidence = "medium"

    if llm:
        sys_prompt = (
            "You are an expert web research analyst for VyaparSathi.\n"
            "Synthesize a clear, accurate, professional answer to the user's research query using ONLY the provided evidence.\n\n"
            "STRICT RULES:\n"
            "1. Ground your answer completely in the provided web evidence.\n"
            "2. Do NOT fill in missing facts or invent numbers from external knowledge. If the evidence is incomplete, explicitly state what details were missing.\n"
            "3. If sources conflict, explicitly state the disagreement.\n"
            "4. Distinguish between publication date and retrieval date when relevant.\n"
        )

        user_prompt = (
            f"Research Question: {query}\n\n"
            f"Gathered Evidence:\n{evidence_text}\n\n"
            "Provide a comprehensive answer followed by 3-5 key findings bullet points."
        )

        try:
            resp = await llm.ainvoke([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ])
            full_resp_text = resp.content if hasattr(resp, "content") else str(resp)
            answer_text = full_resp_text.strip()
            
            # Extract key findings from lines starting with bullets
            lines = answer_text.split("\n")
            for line in lines:
                clean_l = line.strip()
                if clean_l.startswith(("- ", "* ", "• ")) or re.match(r"^\d+\.\s", clean_l):
                    key_findings.append(re.sub(r"^[-*•\d\.]+\s*", "", clean_l))
        except Exception as exc:
            LOGGER.error("web_research.synthesis_llm_failed", error=str(exc))
            errors.append(f"LLM synthesis failed: {str(exc)}")

    if not answer_text:
        # Fallback deterministic synthesis from evidence snippets
        answer_text = f"Research findings for '{query}':\n\n" + "\n".join(
            f"- {e['source_title']}: {e['fact_snippet'][:200]}..." for e in evidence[:4]
        )

    if not key_findings:
        key_findings = [
            f"Gathered evidence from {len(formatted_sources)} web source(s).",
            f"Primary domains evaluated: {', '.join(list({s['domain'] for s in formatted_sources}))}.",
        ]

    if len(formatted_sources) >= 3 and not errors:
        confidence = "high"
    elif len(formatted_sources) >= 1:
        confidence = "medium"
    else:
        confidence = "low"

    freshness_summary = (
        f"Fresh research conducted on {datetime.utcnow().strftime('%B %Y')}. "
        f"Sources evaluated: {len(formatted_sources)}."
    ) if has_temporal else f"Retrieved {len(formatted_sources)} source(s) as of {datetime.utcnow().strftime('%Y-%m-%d')}."

    output_model = WebResearchOutput(
        query=query,
        answer=answer_text,
        key_findings=key_findings,
        sources=formatted_sources,
        confidence=confidence,
        freshness=freshness_summary,
        conflicts=conflicts,
        research_metadata={
            "rounds_taken": rounds_taken,
            "total_sources_evaluated": len(state.get("search_results", [])),
            "total_pages_fetched": len(fetched),
            "errors": errors,
            "has_temporal_context": has_temporal,
        }
    )

    final_dict = output_model.model_dump()

    LOGGER.info(
        "web_research.synthesize_complete",
        query=query[:60],
        sources_count=len(formatted_sources),
        confidence=confidence,
    )

    return {
        "synthesis": final_dict,
        "sources": formatted_sources,
        "status": "completed",
    }
