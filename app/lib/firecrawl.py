"""
app/lib/firecrawl.py
====================
Firecrawl web search / scraping integration for VyaparSathi AI Service.

Configured through FIRECRAWL_API_KEY in `.env`.

Provides:
- get_firecrawl_client(): Shared AsyncFirecrawl client (None when unconfigured).
- is_firecrawl_configured(): True when an API key is available.
- scrape_url(): Clean markdown for one page.
- scrape_urls(): Clean markdown for several pages (sequential, order preserved).
- search_web(): Web search with optional result scraping.
- map_site(): URL discovery for a site.
"""
from __future__ import annotations

import os
from typing import Any

import structlog
from firecrawl import AsyncFirecrawl

from app.config.settings import get_settings

LOGGER = structlog.get_logger("vyaparsathi.ai.lib.firecrawl")

_client: AsyncFirecrawl | None = None
_client_initialized: bool = False


def _resolve_api_key() -> str | None:
    settings = get_settings()
    return (
        settings.firecrawl_api_key
        or os.getenv("FIRECRAWL_API_KEY")
    )


def is_firecrawl_configured() -> bool:
    """Return True when FIRECRAWL_API_KEY is available."""
    return bool(_resolve_api_key())


def get_firecrawl_client() -> AsyncFirecrawl | None:
    """
    Return the shared AsyncFirecrawl client, or None when no API key is set.
    """
    global _client, _client_initialized

    if _client_initialized:
        return _client

    _client_initialized = True
    settings = get_settings()
    api_key = _resolve_api_key()

    if not api_key:
        LOGGER.warning("firecrawl_api_key_missing", message="FIRECRAWL_API_KEY not configured.")
        return None

    try:
        _client = AsyncFirecrawl(
            api_key=api_key,
            api_url=settings.firecrawl_api_url,
            timeout=settings.firecrawl_timeout_seconds,
        )
        LOGGER.info("firecrawl_client_initialized")
    except Exception as e:
        LOGGER.exception("firecrawl_init_failed", error=str(e))
        _client = None

    return _client


def _require_client() -> AsyncFirecrawl:
    client = get_firecrawl_client()
    if client is None:
        raise RuntimeError("Firecrawl is not configured. Set FIRECRAWL_API_KEY in .env.")
    return client


async def scrape_url(
    url: str,
    formats: list[str] | None = None,
    only_main_content: bool = True,
) -> dict[str, Any]:
    """
    Scrape a single URL and return a normalized payload:
    {"url": str, "success": bool, "markdown": str, "metadata": dict}.
    """
    client = _require_client()
    payload = _to_dict(
        await client.scrape_url(
            url,
            formats=formats or ["markdown"],
            only_main_content=only_main_content,
        )
    )

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    markdown = payload.get("markdown") or payload.get("html") or ""

    return {
        "url": metadata.get("url") or url,
        "success": bool(markdown),
        "markdown": markdown,
        "metadata": metadata,
    }


async def scrape_urls(
    urls: list[str],
    formats: list[str] | None = None,
    only_main_content: bool = True,
) -> list[dict[str, Any]]:
    """
    Scrape several URLs. Failures are reported per URL instead of raising.
    """
    documents: list[dict[str, Any]] = []
    for url in urls:
        try:
            documents.append(
                await scrape_url(url, formats=formats, only_main_content=only_main_content)
            )
        except Exception as e:
            LOGGER.warning("firecrawl_scrape_failed", url=url, error=str(e))
            documents.append({"url": url, "success": False, "error": str(e)})
    return documents


async def search_web(
    query: str,
    limit: int = 5,
    scrape_results: bool = False,
) -> dict[str, Any]:
    """
    Run a web search and return a normalized payload:
    {"query": str, "success": bool, "results": [...]}.

    When `scrape_results` is True, each result also carries clean markdown.
    """
    client = _require_client()
    payload = _to_dict(
        await client.search(
            query,
            limit=limit,
            scrape_options={"formats": ["markdown"], "onlyMainContent": True} if scrape_results else None,
        )
    )

    results: list[dict[str, Any]] = []
    for key in ("web", "news", "images", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            results.extend(value)

    return {"query": query, "success": bool(results), "results": results}


async def map_site(url: str, limit: int = 100) -> dict[str, Any]:
    """
    Discover URLs on a site.
    """
    client = _require_client()
    result = await client.map_url(url, limit=limit)
    return _to_dict(result)


def _to_dict(result: Any) -> dict[str, Any]:
    """Normalize SDK responses (pydantic models or dicts) into plain dicts."""
    if isinstance(result, dict):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump(exclude_none=True)
    if hasattr(result, "dict"):
        return result.dict()
    return {"data": result}
