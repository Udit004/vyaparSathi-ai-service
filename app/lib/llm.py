"""
app/lib/llm.py
==============
LLM factory with automatic provider fallback.

Gemini's free tier is capped at 20 requests/day per model, so a single
rate-limit error can take the whole agent down. This module builds a
fallback chain that transparently tries the next provider when the
current one is exhausted.

Fallback order:
    1. gemini-2.5-flash          (primary reasoning — best quality)
    2. openai/gpt-oss-120b       (GROQ — large, high-capability reasoning model)
    3. openai/gpt-oss-20b        (GROQ — lightweight fallback, requires openai/ prefix)
    4. gemini-2.5-flash-lite     (Gemini cheap fallback, rate-limited)
    5. nvidia/llama-3.1-8b-instruct (NVIDIA — fast, cheap fallback)

The returned object behaves like a single chat model: ``ainvoke`` and
``bind_tools`` both work, and tool calls are propagated to every member
of the chain.
"""

from __future__ import annotations

import os
from functools import lru_cache, partial
from typing import Any

import structlog

from app.config.settings import get_settings

LOGGER = structlog.get_logger("vyaparsathi.ai.llm")


def _build_gemini(model: str, **kwargs) -> Any:
    """Build a ChatGoogleGenerativeAI instance for the given model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
        max_retries=0,
    )


def _build_openai(model: str, *, base_url: str, api_key: str) -> Any:
    """
    Build a ChatOpenAI instance pointed at an OpenAI-compatible endpoint.

    ``base_url`` is required for non-OpenAI providers (GROQ, NVIDIA) because
    without it ChatOpenAI defaults to ``https://api.openai.com/v1`` and the
    provider-specific model names (e.g. ``nvidia/llama-3.1-8b-instruct``)
    cannot be resolved. Each fallback entry in ``_PROVIDERS`` binds the
    correct ``base_url`` via ``functools.partial``.

    ``api_key`` is the provider-specific API key (GROQ_API_KEY or NVIDIA_API_KEY)
    passed explicitly so ChatOpenAI does not rely on the generic
    ``OPENAI_API_KEY`` environment variable.
    """
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.3,
        max_retries=0,
    )


def _provider_env(provider_name: str, env_var: str) -> str | None:
    """Return the API key for a provider, preferring settings then env."""
    settings = get_settings()
    if provider_name == "gemini":
        return settings.gemini_api_key or os.environ.get(env_var)
    if provider_name == "nvidia":
        return settings.nvidia_api_key or os.environ.get(env_var)
    if provider_name == "groq":
        return settings.groq_api_key or os.environ.get(env_var)
    return os.environ.get(env_var)


# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

# OpenAI-compatible base URLs per provider. Bound into ``_build_openai`` via
# ``functools.partial`` so each fallback entry targets the right endpoint.
_OPENAI_BASE_URL = {
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "groq": "https://api.groq.com/openai/v1",
}

# Each entry: (provider_name, model_id, build_fn, api_key_env)
# The order here IS the fallback order.
# Strategy:
#   1. gemini-2.5-flash          — primary reasoning model (best quality)
#   2. openai/gpt-oss-120b       — Groq large/powerful reasoning fallback (free tier, high rate limit)
#   3. openai/gpt-oss-20b        — Groq lightweight fallback (MUST include openai/ prefix for Groq routing)
#   4. gemini-2.5-flash-lite      — last resort (rate-limited, 20/day free tier)
#   5. nvidia/llama-3.1-8b-instruct — NVIDIA fallback (fast, cheap)
# NOTE: All Groq model IDs must include the "openai/" prefix when using the
# Groq OpenAI-compatible endpoint — without it, ChatOpenAI routes to OpenAI's
# servers and returns 404 (model not found).
_PROVIDERS = [
    ("gemini", "gemini-2.5-flash", _build_gemini, "GEMINI_API_KEY"),
    ("groq120b", "openai/gpt-oss-120b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ_API_KEY"),
    ("groq20b", "openai/gpt-oss-20b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ_API_KEY"),
    ("gemini", "gemini-2.5-flash-lite", _build_gemini, "GEMINI_API_KEY"),
    ("nvidia", "nvidia/llama-3.1-8b-instruct", partial(_build_openai, base_url=_OPENAI_BASE_URL["nvidia"]), "NVIDIA_API_KEY"),
]


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

class _FallbackLLM:
    """
    A Runnable wrapper that tries each provider in order until one succeeds.

    Rate-limit / quota errors (429) and connection errors trigger a fallthrough
    to the next provider. The first successful response is returned.
    """

    def __init__(self, members: list[Any], names: list[str]):
        self._members = members
        self._names = names

    # -- LangChain Runnable interface ---------------------------------------

    def invoke(self, input, config=None, **kwargs):
        last_error = None
        for member, name in zip(self._members, self._names):
            try:
                LOGGER.debug("llm_fallback_try", provider=name)
                return member.invoke(input, config=config, **kwargs)
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "llm_fallback_member_failed",
                    provider=name,
                    error=str(exc)[:200],
                )
        raise last_error  # type: ignore[misc]

    async def ainvoke(self, input, config=None, **kwargs):
        last_error = None
        for member, name in zip(self._members, self._names):
            try:
                LOGGER.debug("llm_fallback_try", provider=name)
                return await member.ainvoke(input, config=config, **kwargs)
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "llm_fallback_member_failed",
                    provider=name,
                    error=str(exc)[:200],
                )
        raise last_error  # type: ignore[misc]

    def bind_tools(self, tools, **kwargs):
        """Bind tools to every member of the fallback chain."""
        new_members = []
        for member in self._members:
            try:
                new_members.append(member.bind_tools(tools, **kwargs))
            except Exception:
                new_members.append(member)
        return _FallbackLLM(new_members, list(self._names))

    # Pass through any other attribute to the first member (for compatibility)
    def __getattr__(self, item):
        return getattr(self._members[0], item)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

@lru_cache()
def get_llm():
    """
    Return the fallback LLM chain, or None if no provider is configured.

    The chain is built once and cached. Members that fail to construct
    (e.g. missing API key) are silently skipped.
    """
    members: list[Any] = []
    names: list[str] = []
    for provider_name, model, build_fn, env_var in _PROVIDERS:
        api_key = _provider_env(provider_name, env_var)
        if not api_key:
            LOGGER.debug("llm_provider_skip_no_key", provider=provider_name, model=model)
            continue
        try:
            llm = build_fn(model, api_key=api_key)
            members.append(llm)
            names.append(f"{provider_name}/{model}")
            LOGGER.info(
                "llm_provider_ready",
                provider=provider_name,
                model=model,
                position=len(members),
            )
        except Exception as exc:
            LOGGER.warning(
                "llm_provider_init_failed",
                provider=provider_name,
                model=model,
                error=str(exc)[:200],
            )

    if not members:
        LOGGER.error("llm_no_provider_configured")
        return None

    LOGGER.info("llm_fallback_chain_built", members=names)
    return _FallbackLLM(members, names)


def get_llm_status() -> dict:
    """Return diagnostic info about which providers are available."""
    available = []
    for provider_name, model, _build_fn, env_var in _PROVIDERS:
        api_key = _provider_env(provider_name, env_var)
        available.append(
            {
                "provider": provider_name,
                "model": model,
                "configured": bool(api_key),
            }
        )
    return {
        "providers": available,
        "active_count": sum(1 for p in available if p["configured"]),
    }