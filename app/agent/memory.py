"""
app/agent/memory.py
===================
mem0 integration for long-term user preferences and store knowledge.

Uses the Mem0 Python SDK (MemoryClient) to store and retrieve
cross-session memory for:

    - User preferences  (namespace = user_id)
    - Store knowledge   (namespace = store_id, isolated per store)

The client is initialized once as an application-level singleton.
If MEM0_API_KEY is not configured, all functions degrade gracefully
(no-op) so the agent still works without long-term memory.

NOTE: The mem0 SDK is synchronous. We wrap all calls in
``asyncio.to_thread()`` so they don't block the async event loop.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

import structlog

from app.config.settings import get_settings
from app.lib.summarizer import summarize
from app.agent.prompts.summarizer_prompts import (
    MEMORY_EXTRACTION_INSTRUCTION,
    MEMORY_QUERY_INSTRUCTION,
    SUMMARIZER_MEMORY_INSTRUCTION,
)

LOGGER = structlog.get_logger("vyaparsathi.ai.memory")


# ---------------------------------------------------------------------------
# Application-level singleton
# ---------------------------------------------------------------------------

_client: Optional[Any] = None  # mem0.MemoryClient
_enabled: bool = False
_project_instructions_configured: bool = False

_MEM0_CUSTOM_INSTRUCTIONS = (
    "This project is Vyapar Copilot for one Indian retail store. Store only "
    "durable information useful across future conversations.\n\n"
    "Extract:\n"
    "- Explicit user communication preferences such as language, tone, and detail.\n"
    "- Store-specific facts, recurring inventory or sales patterns, and explicit "
    "restocking or business decisions.\n\n"
    "Exclude:\n"
    "- Greetings, small talk, one-off questions, temporary tool outputs, and "
    "assistant refusal text.\n"
    "- Passwords, API keys, tokens, personal identifiers, and sensitive financial "
    "details.\n"
    "- Guesses or facts that are not explicitly supported by the conversation."
)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def get_memory_client():
    """
    Return the shared Mem0 MemoryClient, creating it on first access.

    Returns None if mem0 is not configured or initialization fails.
    """
    global _client, _enabled

    if _client is not None:
        return _client

    settings = get_settings()

    if not settings.mem0_enabled or not settings.mem0_api_key:
        LOGGER.info("mem0_disabled", reason="MEM0_API_KEY not set or disabled")
        return None

    try:
        from mem0 import MemoryClient

        _client = MemoryClient(api_key=settings.mem0_api_key)
        _enabled = True
        _configure_project_instructions(_client)
        LOGGER.info(
            "mem0_initialized",
            api_key_prefix=settings.mem0_api_key[:8],
            mem0_enabled=settings.mem0_enabled,
        )
    except Exception as exc:
        LOGGER.error("mem0_init_failed", error=str(exc), exc_info=True)
        _client = None
        _enabled = False

    return _client


def _configure_project_instructions(client: Any) -> None:
    """Configure Mem0 extraction rules once, without making startup fatal."""
    global _project_instructions_configured
    if _project_instructions_configured:
        return

    try:
        project = getattr(client, "project", None)
        update = getattr(project, "update", None)
        if callable(update):
            update(custom_instructions=_MEM0_CUSTOM_INSTRUCTIONS)
            _project_instructions_configured = True
            LOGGER.info("mem0_custom_instructions_configured")
        else:
            LOGGER.warning("mem0_custom_instructions_unsupported")
    except Exception as exc:
        LOGGER.warning("mem0_custom_instructions_failed", error=str(exc))


def is_enabled() -> bool:
    """Return True if mem0 is available and configured."""
    return _enabled and _client is not None


def get_memory_status() -> dict:
    """Return diagnostic info about mem0 configuration."""
    settings = get_settings()
    return {
        "enabled": is_enabled(),
        "mem0_enabled_setting": settings.mem0_enabled,
        "has_api_key": bool(settings.mem0_api_key),
        "api_key_prefix": settings.mem0_api_key[:8] if settings.mem0_api_key else None,
        "client_initialized": _client is not None,
    }


# ---------------------------------------------------------------------------
# User memory (namespace = user_id)
# ---------------------------------------------------------------------------

async def add_user_memory(
    user_id: str,
    messages: list[dict],
    *,
    curated_messages: list[dict] | None = None,
) -> bool:
    """
    Store conversation messages into the user's long-term memory.

    Args:
        user_id: Authenticated user ID.
        messages: List of {role, content} dicts from the conversation.

    Returns:
        True if the data was stored, False otherwise.
    """
    client = get_memory_client()
    if not client or not messages:
        if not client:
            LOGGER.debug("add_user_memory_skip", reason="mem0 not configured")
        return False

    curated = curated_messages if curated_messages is not None else await curate_memory_messages(messages)
    if not curated:
        LOGGER.debug("add_user_memory_skip", reason="no durable memory extracted")
        return False

    try:
        # mem0 SDK is synchronous — run in thread to avoid blocking event loop
        result = await asyncio.to_thread(client.add, curated, user_id=user_id)
        LOGGER.info("user_memory_added", user_id=user_id, count=len(curated), result=str(result)[:200])
        return True
    except Exception as exc:
        LOGGER.error("user_memory_add_failed", user_id=user_id, error=str(exc), exc_info=True)
        return False


async def search_user_memory(user_id: str, query: str) -> list[dict]:
    """
    Search the user's long-term memory for relevant facts/preferences.

    Args:
        user_id: Authenticated user ID.
        query: Natural-language query string.

    Returns:
        List of memory result dicts (may be empty).
    """
    client = get_memory_client()
    if not client:
        return []

    try:
        results = await asyncio.to_thread(client.search, query, user_id=user_id)
        count = len(results) if results else 0
        LOGGER.info("user_memory_searched", user_id=user_id, query=query[:80], results=count)
        return results or []
    except Exception as exc:
        LOGGER.error("user_memory_search_failed", user_id=user_id, error=str(exc), exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Store memory (namespace = store_id, isolated per store)
# ---------------------------------------------------------------------------

async def add_store_memory(
    store_id: str,
    messages: list[dict],
    *,
    curated_messages: list[dict] | None = None,
) -> bool:
    """
    Store conversation messages into the store's long-term memory.

    Each store is fully isolated — mem0 namespaces by store_id.

    Args:
        store_id: Store being queried.
        messages: List of {role, content} dicts from the conversation.

    Returns:
        True if stored, False otherwise.
    """
    client = get_memory_client()
    if not client or not messages:
        return False

    curated = curated_messages if curated_messages is not None else await curate_memory_messages(messages)
    if not curated:
        LOGGER.debug("add_store_memory_skip", reason="no durable memory extracted")
        return False

    try:
        result = await asyncio.to_thread(client.add, curated, user_id=store_id)
        LOGGER.info("store_memory_added", store_id=store_id, count=len(curated), result=str(result)[:200])
        return True
    except Exception as exc:
        LOGGER.error("store_memory_add_failed", store_id=store_id, error=str(exc), exc_info=True)
        return False


async def search_store_memory(store_id: str, query: str) -> list[dict]:
    """
    Search the store's long-term memory for patterns, notes, past decisions.

    Args:
        store_id: Store being queried.
        query: Natural-language query string.

    Returns:
        List of memory result dicts (may be empty).
    """
    client = get_memory_client()
    if not client:
        return []

    try:
        results = await asyncio.to_thread(client.search, query, user_id=store_id)
        count = len(results) if results else 0
        LOGGER.info("store_memory_searched", store_id=store_id, query=query[:80], results=count)
        return results or []
    except Exception as exc:
        LOGGER.error("store_memory_search_failed", store_id=store_id, error=str(exc), exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Combined retrieval for the think node
# ---------------------------------------------------------------------------

async def load_memory_context(
    *,
    user_id: str,
    store_id: str,
    user_prompt: str,
    current_goal: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Load both user-preference and store-knowledge memory for the agent.

    Returns:
        Tuple of (user_preferences, store_knowledge) dicts.
        Each dict has a "raw" key with the list of mem0 results and
        a "summary" key with a flattened text summary for the LLM.
    """
    memory_query = await build_memory_query(user_prompt, current_goal=current_goal)
    user_results = await search_user_memory(user_id, memory_query)
    store_results = await search_store_memory(store_id, memory_query)

    # Cap raw results to keep the persisted state bounded — the summary
    # is what the LLM actually reads.
    user_results = _cap_results(_filter_relevant_results(user_results))
    store_results = _cap_results(_filter_relevant_results(store_results))

    user_prefs = {
        "raw": user_results,
        "summary": await summarize_memory_results(user_results, memory_query),
    }
    store_knowledge = {
        "raw": store_results,
        "summary": await summarize_memory_results(store_results, memory_query),
    }

    return user_prefs, store_knowledge


def should_retrieve_memory(user_prompt: str) -> bool:
    """Return whether the request needs durable memory rather than live tools."""
    prompt = (user_prompt or "").lower()
    memory_signals = (
        "remember", "last time", "previous", "earlier", "we decided",
        "our decision", "usually", "prefer", "preference", "my language",
        "my style", "what did we discuss",
    )
    return any(signal in prompt for signal in memory_signals)


async def build_memory_query(user_prompt: str, *, current_goal: str = "") -> str:
    """Distill the current request into a focused Mem0 semantic-search query."""
    prompt = (user_prompt or "").strip()
    if not prompt:
        return "retail store inventory sales forecasting restocking preferences"

    query_input = f"Current request: {prompt}"
    if current_goal:
        query_input += f"\nCurrent goal: {current_goal}"

    query = await summarize(
        query_input,
        instruction=MEMORY_QUERY_INSTRUCTION,
        max_tokens=48,
    )
    query = " ".join((query or "").split())
    if not query or query.upper() == "NONE":
        return prompt[:300]
    return query[:300]


async def curate_memory_messages(messages: list[dict]) -> list[dict]:
    """Convert a completed exchange into one durable memory record or skip it."""
    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("NVIDIA_API_KEY"):
        LOGGER.debug("memory_curation_skip", reason="small model unavailable")
        return []

    text = _flatten_results(messages)
    if not text.strip():
        return []

    extracted = await summarize(
        text,
        instruction=MEMORY_EXTRACTION_INSTRUCTION,
        max_tokens=160,
    )
    extracted = (extracted or "").strip()
    if not extracted or extracted.upper() == "NONE":
        return []
    return [{"role": "user", "content": extracted}]


def _filter_relevant_results(results: list[dict]) -> list[dict]:
    """Drop low-confidence scored results while preserving SDK results without scores."""
    if not results:
        return []

    filtered = []
    for result in results:
        score = result.get("score") if isinstance(result, dict) else None
        if isinstance(score, (int, float)) and score < 0.55:
            continue
        filtered.append(result)
    return filtered


# ---------------------------------------------------------------------------
# Result capping — prevents mem0 from flooding the context window
# ---------------------------------------------------------------------------

# Hard caps on how much mem0 data we surface to the LLM per session.
# mem0 can return a large, unbounded list of entries; without these caps
# the summary string alone can balloon to ~50K tokens and blow the
# context window.
_MAX_MEMORY_ENTRIES = 5
_MAX_MEMORY_CHARS = 2_000
_MAX_MEMORY_ENTRY_CHARS = 400


def _cap_results(results: list[dict]) -> list[dict]:
    """
    Trim a mem0 result list to a bounded size.

    Keeps the first ``_MAX_MEMORY_ENTRIES`` results (mem0 ranks by
    relevance, so the head of the list is the most relevant) and truncates
    each entry's text to a sane length.
    """
    if not results:
        return []

    capped = list(results[:_MAX_MEMORY_ENTRIES])
    for i, r in enumerate(capped):
        if isinstance(r, dict):
            text = r.get("memory") or r.get("text")
            if isinstance(text, str) and len(text) > _MAX_MEMORY_ENTRY_CHARS:
                r = dict(r)
                r["memory"] = text[:_MAX_MEMORY_ENTRY_CHARS] + "...[truncated]"
                capped[i] = r
    return capped


async def summarize_memory_results(results: list[dict], query: str) -> str:
    """
    Produce a compact summary of mem0 search results for the LLM.

    Applies hard caps first (so we never send a huge payload to the
    summarizer), then optionally re-summarizes with a small/fast model
    (GROQ/NVIDIA) for extra compression. Falls back to the capped
    flatten if no summarizer is available.
    """
    results = _cap_results(results)
    if not results:
        return ""

    flat = _flatten_results(results)
    if len(flat) <= _MAX_MEMORY_CHARS:
        return flat

    # Use the small summarizer to compress further
    summary = await summarize(
        flat,
        instruction=SUMMARIZER_MEMORY_INSTRUCTION,
        max_tokens=200,
    )
    return summary or flat


def _flatten_results(results: list[dict]) -> str:
    """Flatten capped mem0 results into plain text (no LLM call)."""
    parts = []
    for r in results:
        text = r.get("memory") or r.get("text") or str(r)
        # Defensive: coerce to str so str.join never receives a list/dict
        # (would raise "sequence item 0: expected str instance, list found").
        parts.append(str(text) if not isinstance(text, str) else text)
    return "\n".join(parts)