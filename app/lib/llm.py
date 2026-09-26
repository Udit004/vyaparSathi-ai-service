"""
app/lib/llm.py
==============
LLM factory with automatic provider and multi-key fallback.

Provides a unified LLM interface with multi-provider, multi-key fallback ordering:
    1. Gemini models (gemini-2.5-flash)
    2. Groq models (llama-3.3-70b-versatile, llama-3.1-8b-instant)
    3. OpenRouter free models (meta-llama/llama-3.3-70b-instruct:free, google/gemma-2-9b-it:free)
    4. NVIDIA NIM models (nvidia/llama-3.1-8b-instruct)
    5. Ollama local models (if OLLAMA_BASE_URL is configured or running locally)

The returned ``_FallbackLLM`` object behaves like a single chat model:
``ainvoke``, ``invoke``, ``bind_tools``, ``with_structured_output``, and ``with_config``
all work transparently across every member in the fallback chain.
"""

from __future__ import annotations

import os
from functools import partial
from typing import Any, Callable

import structlog

from app.config.settings import get_settings
from app.lib.gemini_keys import get_provider_keys

LOGGER = structlog.get_logger("vyaparsathi.ai.llm")


def _build_gemini(model: str, *, api_key: str | None = None, **kwargs) -> Any:
    """Build a ChatGoogleGenerativeAI instance for the given model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    key = api_key or settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=key,
        temperature=0.3,
        max_retries=0,
    )


def _build_openai(model: str, *, base_url: str, api_key: str, default_headers: dict | None = None) -> Any:
    """
    Build a ChatOpenAI instance pointed at an OpenAI-compatible endpoint.
    """
    from langchain_openai import ChatOpenAI

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    if default_headers:
        headers.update(default_headers)

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.3,
        max_retries=0,
        default_headers=headers,
    )


def _build_ollama(model: str, *, base_url: str = "http://localhost:11434/v1", api_key: str = "ollama") -> Any:
    """Build a ChatOpenAI instance targeting a local Ollama server."""
    return _build_openai(model, base_url=base_url, api_key=api_key or "ollama")


# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

_OPENAI_BASE_URL = {
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

# Providers prioritized for Complex Reasoning, Planning, Synthesis, and Main Thinking Nodes
_LARGE_PROVIDERS = [
    ("gemini", "gemini-2.5-flash", _build_gemini, "GEMINI"),
    ("groq", "openai/gpt-oss-120b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct:free", partial(_build_openai, base_url=_OPENAI_BASE_URL["openrouter"]), "OPENROUTER"),
    ("openrouter", "openai/gpt-oss-120b:free", partial(_build_openai, base_url=_OPENAI_BASE_URL["openrouter"]), "OPENROUTER"),
    ("nvidia", "meta/llama-3.2-11b-vision-instruct", partial(_build_openai, base_url=_OPENAI_BASE_URL["nvidia"]), "NVIDIA"),
    ("ollama", "llama3", _build_ollama, "OLLAMA"),
]

# Providers prioritized for Fast/Lightweight tasks: Memory extraction, retrieval, summarization, intent routing
_SMALL_PROVIDERS = [
    ("groq", "openai/gpt-oss-20b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("groq", "qwen/qwen3.8-27b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("openrouter", "openai/gpt-oss-20b:free", partial(_build_openai, base_url=_OPENAI_BASE_URL["openrouter"]), "OPENROUTER"),
    ("nvidia", "meta/llama-3.2-11b-vision-instruct", partial(_build_openai, base_url=_OPENAI_BASE_URL["nvidia"]), "NVIDIA"),
    ("gemini", "gemini-2.5-flash", _build_gemini, "GEMINI"),
    ("ollama", "llama3", _build_ollama, "OLLAMA"),
]

_PROVIDERS = _LARGE_PROVIDERS


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

class _FallbackLLM:
    """
    A Runnable wrapper that tries each provider in order until one succeeds.

    Rate-limit / quota errors (429), API failures (400/404), and connection
    errors trigger a fallthrough to the next provider. The first successful
    response is returned.
    """

    def __init__(self, members: list[Any], names: list[str], builders: list[Callable[[], Any]] | None = None):
        self._members = members
        self._names = names
        self._builders = builders or [lambda m=m: m for m in members]

    # -- LangChain Runnable interface ---------------------------------------

    def invoke(self, input, config=None, **kwargs):
        last_error = None
        for idx, (member, name) in enumerate(zip(self._members, self._names)):
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
        for idx, (member, name) in enumerate(zip(self._members, self._names)):
            try:
                LOGGER.debug("llm_fallback_try", provider=name)
                return await member.ainvoke(input, config=config, **kwargs)
            except Exception as exc:
                last_error = exc
                err_msg = str(exc).lower()
                if "event loop" in err_msg or "closed" in err_msg:
                    try:
                        fresh_member = self._builders[idx]()
                        return await fresh_member.ainvoke(input, config=config, **kwargs)
                    except Exception as exc2:
                        last_error = exc2
                LOGGER.warning(
                    "llm_fallback_member_failed",
                    provider=name,
                    error=str(exc)[:200],
                )
        raise last_error  # type: ignore[misc]

    async def astream(self, input, config=None, **kwargs):
        last_error = None
        for idx, (member, name) in enumerate(zip(self._members, self._names)):
            try:
                LOGGER.debug("llm_fallback_try_astream", provider=name)
                async for chunk in member.astream(input, config=config, **kwargs):
                    yield chunk
                return
            except Exception as exc:
                last_error = exc
                err_msg = str(exc).lower()
                if "event loop" in err_msg or "closed" in err_msg:
                    try:
                        fresh_member = self._builders[idx]()
                        async for chunk in fresh_member.astream(input, config=config, **kwargs):
                            yield chunk
                        return
                    except Exception as exc2:
                        last_error = exc2
                LOGGER.warning(
                    "llm_fallback_member_failed",
                    provider=name,
                    error=str(exc)[:200],
                )
        raise last_error  # type: ignore[misc]

    def bind_tools(self, tools, **kwargs):
        """Bind tools to every member of the fallback chain."""
        new_members = []
        new_names = []
        new_builders = []
        for member, name, builder in zip(self._members, self._names, self._builders):
            try:
                bound = member.bind_tools(tools, **kwargs)
                new_members.append(bound)
                new_names.append(name)
                new_builders.append(lambda b=builder, t=tools, kw=kwargs: b().bind_tools(t, **kw))
            except Exception as exc:
                LOGGER.warning("llm_bind_tools_member_failed", provider=name, error=str(exc)[:200])
                new_members.append(member)
                new_names.append(name)
                new_builders.append(builder)
        return _FallbackLLM(new_members, new_names, new_builders)

    def with_structured_output(self, schema: Any, **kwargs) -> _FallbackLLM:
        """Bind structured output schema to every member of the fallback chain."""
        new_members = []
        new_names = []
        new_builders = []
        for member, name, builder in zip(self._members, self._names, self._builders):
            try:
                structured_member = member.with_structured_output(schema, **kwargs)
                new_members.append(structured_member)
                new_names.append(name)
                new_builders.append(lambda b=builder, s=schema, kw=kwargs: b().with_structured_output(s, **kw))
            except Exception as exc:
                LOGGER.warning(
                    "llm_with_structured_output_member_failed",
                    provider=name,
                    error=str(exc)[:200],
                )
        if not new_members:
            LOGGER.error("llm_with_structured_output_no_supported_members")
            raise RuntimeError("No configured LLM member supports structured output for this schema.")
        return _FallbackLLM(new_members, new_names, new_builders)

    def with_config(self, config: dict | None = None, **kwargs) -> _FallbackLLM:
        """Apply Runnable config (e.g. callbacks=[]) to every member of the fallback chain."""
        new_members = []
        new_builders = []
        for member, builder in zip(self._members, self._builders):
            if hasattr(member, "with_config"):
                try:
                    conf = member.with_config(config, **kwargs)
                    new_members.append(conf)
                    new_builders.append(lambda b=builder, c=config, kw=kwargs: b().with_config(c, **kw))
                except Exception:
                    new_members.append(member)
                    new_builders.append(builder)
            else:
                new_members.append(member)
                new_builders.append(builder)
        return _FallbackLLM(new_members, list(self._names), new_builders)

    def __getattr__(self, item):
        return getattr(self._members[0], item)


# ---------------------------------------------------------------------------
# Internal builder helper
# ---------------------------------------------------------------------------

def _build_chain(provider_list: list) -> _FallbackLLM | None:
    members: list[Any] = []
    names: list[str] = []
    builders: list[Callable[[], Any]] = []

    for provider_name, model, build_fn, provider_prefix in provider_list:
        if provider_name == "ollama":
            settings = get_settings()
            base_url = settings.ollama_base_url or os.getenv("OLLAMA_BASE_URL")
            if not base_url:
                continue
            try:
                b_fn = partial(build_fn, model, base_url=base_url)
                llm = b_fn()
                members.append(llm)
                builders.append(b_fn)
                names.append(f"ollama/{model}")
            except Exception as exc:
                LOGGER.warning("llm_provider_init_failed", provider="ollama", model=model, error=str(exc)[:200])
            continue

        keys = get_provider_keys(provider_prefix)
        if not keys:
            continue

        for idx, key in enumerate(keys, 1):
            try:
                if provider_name == "gemini":
                    b_fn = partial(_build_gemini, model, api_key=key)
                else:
                    b_fn = partial(build_fn, model, api_key=key)

                llm = b_fn()
                members.append(llm)
                builders.append(b_fn)
                key_tag = f"key_{idx}"
                names.append(f"{provider_name}/{model} ({key_tag})")
            except Exception as exc:
                LOGGER.warning(
                    "llm_provider_init_failed",
                    provider=provider_name,
                    model=model,
                    key_index=idx,
                    error=str(exc)[:200],
                )

    if not members:
        return None

    return _FallbackLLM(members, names, builders)


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------

def get_large_llm() -> _FallbackLLM | None:
    """
    Return the high-capacity LLM fallback chain (Gemini 2.5, Llama 70B, etc.).
    Used for main thinking nodes, planning, complex reasoning, and research synthesis.
    """
    return _build_chain(_LARGE_PROVIDERS)


def get_small_llm() -> _FallbackLLM | None:
    """
    Return the lightweight, fast LLM fallback chain (GPT-20B, Qwen 27B, Llama 11B).
    Used for memory extraction, memory retrieval formatting, summarizations, and intent routing.
    """
    return _build_chain(_SMALL_PROVIDERS) or _build_chain(_LARGE_PROVIDERS)


def get_fast_llm() -> _FallbackLLM | None:
    """Alias for get_small_llm()."""
    return get_small_llm()


def get_llm() -> _FallbackLLM | None:
    """
    Default LLM getter — returns the large/high-capacity LLM chain.
    """
    return get_large_llm()


def clear_llm_cache():
    """No-op kept for backwards compatibility."""
    pass


def get_llm_status() -> dict:
    """Return diagnostic info about which providers and keys are available."""
    available = []
    for provider_name, model, _build_fn, provider_prefix in _PROVIDERS:
        keys = get_provider_keys(provider_prefix) if provider_name != "ollama" else [os.getenv("OLLAMA_BASE_URL", "")]
        keys = [k for k in keys if k]
        available.append(
            {
                "provider": provider_name,
                "model": model,
                "configured": bool(keys),
                "key_count": len(keys),
            }
        )
    return {
        "providers": available,
        "active_count": sum(1 for p in available if p["configured"]),
    }