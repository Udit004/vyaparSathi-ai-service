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
import time
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
    ("gemini", "gemini-2.5-flash-lite", _build_gemini, "GEMINI"),
    ("nvidia", "meta/llama-3.2-11b-vision-instruct", partial(_build_openai, base_url=_OPENAI_BASE_URL["nvidia"]), "NVIDIA"),
    ("groq", "openai/gpt-oss-120b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct:free", partial(_build_openai, base_url=_OPENAI_BASE_URL["openrouter"]), "OPENROUTER"),
    ("ollama", "llama3", _build_ollama, "OLLAMA"),
]

# Providers prioritized for Fast/Lightweight tasks: Memory extraction, retrieval, summarization, intent routing, planner
_SMALL_PROVIDERS = [
    ("groq", "openai/gpt-oss-20b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("groq", "qwen/qwen3.8-27b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("gemini", "gemini-2.5-flash-lite", _build_gemini, "GEMINI"),
    ("nvidia", "meta/llama-3.2-11b-vision-instruct", partial(_build_openai, base_url=_OPENAI_BASE_URL["nvidia"]), "NVIDIA"),
    ("gemini", "gemini-2.5-flash", _build_gemini, "GEMINI"),
]

_SAFEGUARD_PROVIDERS = [
    ("gemini", "gemini-2.5-flash-lite", _build_gemini, "GEMINI"),
    ("groq", "openai/gpt-oss-safeguard-20b", partial(_build_openai, base_url=_OPENAI_BASE_URL["groq"]), "GROQ"),
    ("nvidia", "meta/llama-3.2-11b-vision-instruct", partial(_build_openai, base_url=_OPENAI_BASE_URL["nvidia"]), "NVIDIA"),
]


# ---------------------------------------------------------------------------
# Fallback chain
# ---------------------------------------------------------------------------

# Cooldown registry for rate-limited (429 / RESOURCE_EXHAUSTED) providers
_PROVIDER_COOLDOWNS: dict[str, float] = {}
_COOLDOWN_SECONDS = 30.0


class _FallbackLLM:
    """
    A Runnable wrapper that tries each provider in order until one succeeds.

    Initializes members lazily: only the primary provider is constructed upfront.
    Fallback members are initialized on-demand if the primary fails.
    Rate-limited providers (429 / RESOURCE_EXHAUSTED / TPM limits) are put on a cooldown
    to avoid burning latency on known-exhausted keys.
    """

    def __init__(self, members: list[Any], names: list[str], builders: list[Callable[[], Any]] | None = None):
        self._members = list(members)
        self._names = list(names)
        self._builders = builders or [lambda m=m: m for m in members]

    def _get_member(self, idx: int) -> Any:
        """Lazily initialize the member at index if not already instantiated."""
        if self._members[idx] is None:
            t0 = time.perf_counter()
            self._members[idx] = self._builders[idx]()
            LOGGER.debug(
                "llm_member_lazy_initialized",
                provider=self._names[idx],
                ms=round((time.perf_counter() - t0) * 1000, 1),
            )
        return self._members[idx]

    # -- LangChain Runnable interface ---------------------------------------

    def invoke(self, input, config=None, **kwargs):
        now = time.time()
        valid_indices = [idx for idx, name in enumerate(self._names) if _PROVIDER_COOLDOWNS.get(name, 0) <= now]
        if not valid_indices:
            LOGGER.warning("llm_all_providers_on_cooldown_resetting", providers=self._names)
            for name in self._names:
                _PROVIDER_COOLDOWNS.pop(name, None)
            valid_indices = list(range(len(self._names)))

        last_error = None
        for idx in valid_indices:
            name = self._names[idx]
            try:
                member = self._get_member(idx)
                LOGGER.debug("llm_fallback_try", provider=name)
                return member.invoke(input, config=config, **kwargs)
            except Exception as exc:
                last_error = exc
                err_msg = str(exc).lower()
                if any(k in err_msg for k in ("429", "resource_exhausted", "quota", "rate_limit", "tpm", "tokens per minute")):
                    _PROVIDER_COOLDOWNS[name] = time.time() + _COOLDOWN_SECONDS
                    LOGGER.warning("llm_provider_cooldown_set", provider=name, cooldown_s=_COOLDOWN_SECONDS)
                LOGGER.warning(
                    "llm_fallback_member_failed",
                    provider=name,
                    error=str(exc)[:200],
                )

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"All LLM providers ({self._names}) failed to invoke.")

    async def ainvoke(self, input, config=None, **kwargs):
        now = time.time()
        valid_indices = [idx for idx, name in enumerate(self._names) if _PROVIDER_COOLDOWNS.get(name, 0) <= now]
        if not valid_indices:
            LOGGER.warning("llm_all_providers_on_cooldown_resetting", providers=self._names)
            for name in self._names:
                _PROVIDER_COOLDOWNS.pop(name, None)
            valid_indices = list(range(len(self._names)))

        last_error = None
        for idx in valid_indices:
            name = self._names[idx]
            try:
                member = self._get_member(idx)
                LOGGER.debug("llm_fallback_try", provider=name)
                return await member.ainvoke(input, config=config, **kwargs)
            except Exception as exc:
                last_error = exc
                err_msg = str(exc).lower()
                if any(k in err_msg for k in ("429", "resource_exhausted", "quota", "rate_limit", "tpm", "tokens per minute")):
                    _PROVIDER_COOLDOWNS[name] = time.time() + _COOLDOWN_SECONDS
                    LOGGER.warning("llm_provider_cooldown_set", provider=name, cooldown_s=_COOLDOWN_SECONDS)
                if "event loop" in err_msg or "closed" in err_msg:
                    try:
                        fresh_member = self._builders[idx]()
                        self._members[idx] = fresh_member
                        return await fresh_member.ainvoke(input, config=config, **kwargs)
                    except Exception as exc2:
                        last_error = exc2
                LOGGER.warning(
                    "llm_fallback_member_failed",
                    provider=name,
                    error=str(exc)[:200],
                )

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"All LLM providers ({self._names}) failed to invoke.")

    async def astream(self, input, config=None, **kwargs):
        now = time.time()
        valid_indices = [idx for idx, name in enumerate(self._names) if _PROVIDER_COOLDOWNS.get(name, 0) <= now]
        if not valid_indices:
            LOGGER.warning("llm_all_providers_on_cooldown_resetting", providers=self._names)
            for name in self._names:
                _PROVIDER_COOLDOWNS.pop(name, None)
            valid_indices = list(range(len(self._names)))

        last_error = None
        for idx in valid_indices:
            name = self._names[idx]
            try:
                member = self._get_member(idx)
                LOGGER.debug("llm_fallback_try_astream", provider=name)
                async for chunk in member.astream(input, config=config, **kwargs):
                    yield chunk
                return
            except Exception as exc:
                last_error = exc
                err_msg = str(exc).lower()
                if any(k in err_msg for k in ("429", "resource_exhausted", "quota", "rate_limit", "tpm", "tokens per minute")):
                    _PROVIDER_COOLDOWNS[name] = time.time() + _COOLDOWN_SECONDS
                    LOGGER.warning("llm_provider_cooldown_set", provider=name, cooldown_s=_COOLDOWN_SECONDS)
                if "event loop" in err_msg or "closed" in err_msg:
                    try:
                        fresh_member = self._builders[idx]()
                        self._members[idx] = fresh_member
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

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"All LLM providers ({self._names}) failed to stream.")

    def bind_tools(self, tools, **kwargs):
        """Bind tools lazily to fallback members."""
        new_names = list(self._names)
        new_builders = []
        for b in self._builders:
            def _make_tool_bound_builder(builder=b):
                instance = builder()
                if hasattr(instance, "bind_tools"):
                    try:
                        return instance.bind_tools(tools, **kwargs)
                    except Exception as e:
                        LOGGER.warning("bind_tools_failed_on_member", error=str(e)[:150])
                        return instance
                return instance
            new_builders.append(_make_tool_bound_builder)

        new_members: list[Any] = [None] * len(new_builders)
        if self._members and self._members[0] is not None:
            try:
                new_members[0] = self._members[0].bind_tools(tools, **kwargs)
            except Exception:
                new_members[0] = new_builders[0]()
        elif new_builders:
            new_members[0] = new_builders[0]()
        return _FallbackLLM(new_members, new_names, new_builders)

    def with_structured_output(self, schema: Any, **kwargs) -> _FallbackLLM:
        """Bind structured output schema lazily to fallback members."""
        new_names = list(self._names)
        new_builders = []
        for b in self._builders:
            def _make_structured_builder(builder=b):
                instance = builder()
                if hasattr(instance, "with_structured_output"):
                    try:
                        return instance.with_structured_output(schema, **kwargs)
                    except Exception as e:
                        LOGGER.warning("with_structured_output_failed_on_member", error=str(e)[:150])
                        return instance
                return instance
            new_builders.append(_make_structured_builder)

        new_members: list[Any] = [None] * len(new_builders)
        if self._members and self._members[0] is not None:
            try:
                new_members[0] = self._members[0].with_structured_output(schema, **kwargs)
            except Exception:
                new_members[0] = new_builders[0]()
        elif new_builders:
            new_members[0] = new_builders[0]()
        return _FallbackLLM(new_members, new_names, new_builders)

    def with_config(self, config: dict | None = None, **kwargs) -> _FallbackLLM:
        """Apply Runnable config (e.g. callbacks=[]) lazily to fallback members."""
        new_names = list(self._names)
        new_builders = [
            (lambda b=b, c=config, kw=kwargs: b().with_config(c, **kw) if hasattr(b(), "with_config") else b())
            for b in self._builders
        ]
        new_members: list[Any] = [None] * len(new_builders)
        if self._members and self._members[0] is not None:
            m = self._members[0]
            new_members[0] = m.with_config(config, **kwargs) if hasattr(m, "with_config") else m
        elif new_builders:
            new_members[0] = new_builders[0]()
        return _FallbackLLM(new_members, new_names, new_builders)

    def __getattr__(self, item):
        member0 = self._get_member(0)
        return getattr(member0, item)


# ---------------------------------------------------------------------------
# Internal builder helper
# ---------------------------------------------------------------------------

# Module-level cache so the chain is built once per process, not on every call.
_llm_cache: _FallbackLLM | None = None
_small_llm_cache: _FallbackLLM | None = None
_safeguard_llm_cache: _FallbackLLM | None = None


def _build_chain(provider_list: list) -> _FallbackLLM | None:
    names: list[str] = []
    builders: list[Callable[[], Any]] = []

    for provider_name, model, build_fn, provider_prefix in provider_list:
        if provider_name == "ollama":
            settings = get_settings()
            base_url = settings.ollama_base_url or os.getenv("OLLAMA_BASE_URL")
            if not base_url:
                continue
            b_fn = partial(build_fn, model, base_url=base_url)
            builders.append(b_fn)
            names.append(f"ollama/{model}")
            continue

        keys = get_provider_keys(provider_prefix)
        if not keys:
            continue

        for idx, key in enumerate(keys, 1):
            if provider_name == "gemini":
                b_fn = partial(_build_gemini, model, api_key=key)
            else:
                b_fn = partial(build_fn, model, api_key=key)

            builders.append(b_fn)
            key_tag = f"key_{idx}"
            names.append(f"{provider_name}/{model} ({key_tag})")

    if not builders:
        return None

    # Lazy members: initialize only the primary member upfront to eliminate startup lag
    members: list[Any] = [None] * len(builders)
    try:
        t0 = time.perf_counter()
        members[0] = builders[0]()
        LOGGER.info("llm_primary_member_ready", provider=names[0], ms=round((time.perf_counter() - t0) * 1000, 1))
    except Exception as exc:
        LOGGER.warning("llm_primary_member_init_failed", provider=names[0], error=str(exc)[:200])

    return _FallbackLLM(members, names, builders)


def get_llm() -> _FallbackLLM | None:
    """
    Return the primary fallback LLM chain for complex reasoning, planning,
    and synthesis tasks.

    Builds the ``_FallbackLLM`` once and caches it at module level.
    Returns ``None`` when no provider key is configured.
    """
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache
    chain = _build_chain(_LARGE_PROVIDERS)
    if chain is None:
        LOGGER.error("get_llm_no_provider_configured")
        return None
    LOGGER.info("get_llm_fallback_chain_built", members=chain._names)
    _llm_cache = chain
    return _llm_cache


def get_safeguard_llm() -> _FallbackLLM | None:
    """
    Return the fallback LLM chain specifically for safety grading (Guard models).
    """
    members: list[Any] = []
    names: list[str] = []
    builders: list[Callable[[], Any]] = []

    for provider_name, model, build_fn, provider_prefix in _SAFEGUARD_PROVIDERS:
        keys = get_provider_keys(provider_prefix)
        if not keys:
            LOGGER.debug("safeguard_llm_provider_skip_no_key", provider=provider_name, model=model)
            continue

        for idx, key in enumerate(keys, 1):
            try:
                b_fn = partial(build_fn, model, api_key=key)
                llm = b_fn()
                members.append(llm)
                builders.append(b_fn)
                key_tag = f"key_{idx}"
                names.append(f"{provider_name}/{model} ({key_tag})")
                LOGGER.info(
                    "safeguard_llm_provider_ready",
                    provider=provider_name,
                    model=model,
                    key_index=idx,
                    position=len(members),
                )
            except Exception as exc:
                LOGGER.warning(
                    "safeguard_llm_provider_init_failed",
                    provider=provider_name,
                    model=model,
                    key_index=idx,
                    error=str(exc)[:200],
                )

    if not members:
        LOGGER.error("safeguard_llm_no_provider_configured")
        return None

    LOGGER.info("safeguard_llm_fallback_chain_built", members=names)
    return _FallbackLLM(members, names, builders)


def get_small_llm() -> _FallbackLLM | None:
    """
    Return the fallback LLM chain prioritizing small/fast models.

    Builds the chain once and caches it at module level.
    Iterates through Groq, OpenRouter, NVIDIA, and Ollama first,
    falling back to Gemini only if no fast models are available.
    """
    global _small_llm_cache
    if _small_llm_cache is not None:
        return _small_llm_cache

    small_providers = [p for p in _SMALL_PROVIDERS if p[0] != "gemini"]
    gemini_providers = [p for p in _SMALL_PROVIDERS if p[0] == "gemini"]
    reordered_providers = small_providers + gemini_providers

    chain = _build_chain(reordered_providers)
    if chain is None:
        LOGGER.error("small_llm_no_provider_configured")
        return None

    LOGGER.info("small_llm_fallback_chain_built", members=chain._names)
    _small_llm_cache = chain
    return _small_llm_cache


def clear_llm_cache():
    """Clear all cached LLM chain instances (useful for testing or key rotation)."""
    global _llm_cache, _small_llm_cache, _safeguard_llm_cache
    _llm_cache = None
    _small_llm_cache = None
    _safeguard_llm_cache = None
    LOGGER.info("llm_cache_cleared")


def get_llm_status() -> dict:
    """Return diagnostic info about which providers and keys are available."""
    available = []
    for provider_name, model, _build_fn, provider_prefix in _LARGE_PROVIDERS:
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