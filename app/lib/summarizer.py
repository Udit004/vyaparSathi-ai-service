"""
app/lib/summarizer.py
=====================
Small/fast LLM summarizer for compressing large payloads before they
enter the agent's context window.

Uses GROQ (primary) or NVIDIA (fallback) — both expose an
OpenAI-compatible chat completions endpoint. These providers offer
small, cheap models (e.g. llama-3.1-8b-instant on Groq, nvidia/llama-3.1-8b-instruct
on NVIDIA) that are ideal for summarization: fast, low-cost, and
accurate enough for compression tasks.

The summarizer is best-effort: if no provider key is configured or a
call fails, it falls back to a deterministic truncation so the agent
never breaks.

Used by:
    - app/agent/memory.py   → compress mem0 search results
    - app/agent/nodes/think_node.py → compress sliding-window history
    - app/agent/nodes/observe_node.py → compress tool payloads before
      they become ToolMessages in the LLM context
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.summarizer")

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

# Order matters: GROQ is primary, NVIDIA is fallback.
_PROVIDERS = (
    ("groq", "https://api.groq.com/openai/v1", "llama-3.1-8b-instant"),
    ("nvidia", "https://integrate.api.nvidia.com/v1", "llama-3.1-8b-instruct"),
)

# Per-provider env var name for the API key
_PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
}

# Cache of instantiated OpenAI async clients keyed by provider name.
_clients: dict[str, Any] = {}


def _get_client(provider: str):
    """Return (or create) an async OpenAI client for the given provider."""
    if provider in _clients:
        return _clients[provider]

    env_name = _PROVIDER_KEY_ENV[provider]
    api_key = os.environ.get(env_name)
    if not api_key:
        return None

    try:
        from openai import AsyncOpenAI
    except Exception as exc:  # pragma: no cover - dependency guard
        LOGGER.error("summarizer_openai_import_failed", provider=provider, error=str(exc))
        return None

    _, base_url, _ = _PROVIDERS[provider]
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=10.0, max_retries=1)
    _clients[provider] = client
    LOGGER.info("summarizer_client_initialized", provider=provider, env=env_name)
    return client


def _available_provider():
    """Return the first provider name with a configured API key, or None."""
    for provider, _, _ in _PROVIDERS:
        if os.environ.get(_PROVIDER_KEY_ENV[provider]):
            return provider
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def summarize(
    text,
    *,
    instruction: str,
    max_tokens: int = 256,
    provider: str | None = None,
) -> str:
    """
    Summarize ``text`` using a small/fast model.

    Args:
        text: The raw text to compress. Non-string values (lists, dicts,
            None) are flattened to a string before use.
        instruction: Task instruction (e.g. "Summarize this inventory report").
        max_tokens: Maximum tokens for the summary.
        provider: Force a specific provider ("groq"|"nvidia"); auto-detect if None.

    Returns:
        A compressed summary string. Falls back to truncation if no
        provider is available or the call fails.
    """
    text = _flatten_text(text)
    if not text or not text.strip():
        return ""

    provider = provider or _available_provider()
    client = _get_client(provider) if provider else None

    if client is None:
        LOGGER.debug("summarizer_fallback_truncation", reason="no provider key", chars=len(text))
        return _truncate(text, max_chars=max_tokens * 4)

    _, _, model = _PROVIDERS[provider]
    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        summary = completion.choices[0].message.content or ""
        LOGGER.info(
            "summarizer_success",
            provider=provider,
            model=model,
            in_chars=len(text),
            out_chars=len(summary),
        )
        return summary.strip()
    except Exception as exc:
        LOGGER.warning(
            "summarizer_failed",
            provider=provider,
            error=str(exc),
            fallback="truncation",
        )
        return _truncate(text, max_chars=max_tokens * 4)


def summarize_sync(
    text,
    *,
    instruction: str,
    max_tokens: int = 256,
    provider: str | None = None,
) -> str:
    """Synchronous wrapper around :func:`summarize`.

    Safe to call from sync code. If called from within a running event
    loop, it schedules the coroutine on that loop and blocks until it
    completes (the caller is responsible for not deadlocking).
    """
    text = _flatten_text(text)
    if not text or not text.strip():
        return ""
    coro = summarize(text, instruction=instruction, max_tokens=max_tokens, provider=provider)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside a running loop — block on the coroutine.
            # This works because the async OpenAI client uses its own
            # transport and won't deadlock on the same loop.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _flatten_text(value) -> str:
    """
    Recursively coerce a value into a plain string.

    Some callers pass non-string inputs (lists, dicts, None) — e.g. a
    tool payload that is a list of dicts. Passing such a value directly to
    ``str.join`` raises::

        TypeError: sequence item 0: expected str instance, list found

    This helper walks nested lists/dicts and flattens them into a single
    string so every downstream ``.join`` / ``len`` call is safe.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return " ".join(_flatten_text(v) for v in value)
    if isinstance(value, dict):
        if "text" in value:
            return _flatten_text(value["text"])
        return str(value)
    if value is None:
        return ""
    return str(value)


def _truncate(text, *, max_chars: int) -> str:
    """Deterministic truncation fallback."""
    text = _flatten_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...[truncated]"