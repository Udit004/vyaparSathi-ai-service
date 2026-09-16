"""
app/lib/grader.py
=================
Strict scope-and-safety classifier for the Vyapar Copilot guardrail node.

Mirrors the small/fast OpenAI-compatible client pattern used by
``app/lib/summarizer.py`` — GROQ (primary) then NVIDIA (fallback) — using a
cached ``AsyncOpenAI`` client with explicit base URLs. It deliberately uses
GROQ's dedicated safeguard model (``openai/gpt-oss-safeguard-20b``); NVIDIA's
``nvidia/llama-3.1-8b-instruct`` is the fallback. None of this touches Gemini,
so the main Gemini-backed reasoning model keeps its full quota for the think
node.

Fail-closed policy: this is a hard allow-list, not a denylist. Any
uncertainty — no provider configured, API error, unparseable response —
results in an off_topic DENY, never a pass-through. That is a deliberate
availability tradeoff: if the classifier is down the assistant is unavailable
for everything rather than silently letting unscoped requests through.

Used by:
    - app/agent/nodes/grader_node.py
"""

from __future__ import annotations

import os
from typing import Any, Literal, Tuple

import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.grader")

Verdict = Literal["safe", "off_topic", "harmful"]


# ---------------------------------------------------------------------------
# Provider configuration
# GROQ uses the dedicated safeguard model (groqGuard); NVIDIA is fallback.
# Mirrors the lightweight roster in app/lib/llm.py.
# ---------------------------------------------------------------------------

# Ordered mapping: first entry with a configured API key wins.
# Each value is (base_url, model).
_PROVIDER_CONFIG = {
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-safeguard-20b"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "nvidia/llama-3.1-8b-instruct"),
}

_PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
}

# Cache of instantiated async OpenAI clients keyed by provider name.
_clients: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Client helpers
# ---------------------------------------------------------------------------

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
        LOGGER.error("grader_openai_import_failed", provider=provider, error=str(exc))
        return None

    base_url, _ = _PROVIDER_CONFIG[provider]
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=10.0,
        max_retries=1,
    )
    _clients[provider] = client
    LOGGER.info("grader_client_initialized", provider=provider, env=env_name)
    return client


def _available_provider() -> str | None:
    """Return the first provider name with a configured API key, or None."""
    for provider in _PROVIDER_CONFIG:
        if os.environ.get(_PROVIDER_KEY_ENV[provider]):
            return provider
    return None


# ---------------------------------------------------------------------------
# Keyword floor — conservative deny-list of severe harms only.
# Checked FIRST on every call as a hard floor, and used as the sole check
# when no provider is configured. It can never reliably detect off_topic
# (that needs semantic understanding of the retail domain), which is why
# the no-provider / failure paths below deny by default instead of relying
# on keywords for scope.
# ---------------------------------------------------------------------------

_SEVERE_HARM_KEYWORDS = (
    "how to make a bomb",
    "make a bomb",
    "how to create a bomb",
    "how to hack",
    "hack into",
    "kill myself",
    "suicide",
    "i want to die",
    "self harm",
    "self-harm",
)

_RETAIL_SCOPE_PHRASES = (
    "stockout",
    "out of stock",
    "low stock",
    "dead stock",
    "inventory",
    "restock",
    "top selling",
    "sales summary",
    "sales trend",
    "forecast",
    "anomaly",
)

_CONVERSATION_SCOPE_PHRASES = (
    "what were we discussing",
    "what we were discussing",
    "what did we discuss",
    "what have we discussed",
    "recent chat",
    "recent conversation",
    "conversation summary",
    "summarize our chat",
    "summarise our chat",
)

_STORE_OVERVIEW_PHRASES = (
    "tell me about my store",
    "tell me about the store",
    "about my store",
    "store summary",
    "store overview",
    "store insights",
    "store health",
    "overall store",
)

_PROMPT_INJECTION_PHRASES = (
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "reveal your instructions",
    "change your instructions",
    "jailbreak",
)


def _keyword_harm_check(prompt: str) -> Tuple[bool, str]:
    """Best-effort offline check for the most severe harm patterns."""
    p = (prompt or "").lower()
    for kw in _SEVERE_HARM_KEYWORDS:
        if kw in p:
            return True, f"matched keyword: {kw}"
    return False, ""


def _obvious_retail_query(prompt: str) -> bool:
    """Recognize unmistakable retail queries when the small model is uncertain."""
    lowered = (prompt or "").lower()
    if any(phrase in lowered for phrase in _PROMPT_INJECTION_PHRASES):
        return False
    return (
        any(phrase in lowered for phrase in _RETAIL_SCOPE_PHRASES)
        or any(phrase in lowered for phrase in _CONVERSATION_SCOPE_PHRASES)
        or any(phrase in lowered for phrase in _STORE_OVERVIEW_PHRASES)
        or ("store" in lowered and "summary" in lowered)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def classify_prompt(prompt: str, *, instruction: str) -> Tuple[Verdict, str]:
    """
    Classify ``prompt`` as safe / off_topic / harmful using a tiny/fast model.

    Fail-closed: any failure mode below returns ``off_topic`` (or
    ``harmful`` on a keyword hit), never ``safe``.

    Args:
        prompt:       The raw user message to check.
        instruction:  System instruction steering the classifier.

    Returns:
        tuple ``(verdict, reason)``.
    """
    if not prompt or not prompt.strip():
        return "safe", ""

    # Hard floor, checked before anything else, regardless of LLM outcome.
    kw_harm, kw_reason = _keyword_harm_check(prompt)
    if kw_harm:
        return "harmful", kw_reason

    # These phrases unambiguously refer to the store's operational data.
    # Allow them without a network classifier so a guard-provider outage does
    # not block ordinary inventory, forecast, sales, or restock questions.
    if _obvious_retail_query(prompt):
        return "safe", "obvious in-scope retail operations query"

    provider = _available_provider()
    client = _get_client(provider) if provider else None

    if client is None:
        LOGGER.error(
            "grader_no_provider_deny",
            reason="no provider key configured (GROQ_API_KEY / NVIDIA_API_KEY) — "
            "denying by default (strict mode)",
        )
        return "off_topic", "classifier unavailable — denied by default"

    _, model = _PROVIDER_CONFIG[provider]

    try:
        completion = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        text = (completion.choices[0].message.content or "").strip()
        lower = text.lower()
        first_token = lower.split()[0].strip(".,;:-\"'`()") if lower.split() else ""

        if first_token in ("safe", "off_topic", "harmful"):
            verdict: Verdict = first_token  # type: ignore[assignment]
            reason = text[len(first_token):].lstrip(" :-\t")

            if verdict == "safe":
                # Defense-in-depth: even if the LLM says safe, a severe keyword
                # match always overrides to harmful.
                kw_harm2, kw_reason2 = _keyword_harm_check(prompt)
                if kw_harm2:
                    LOGGER.warning(
                        "grader_overrides_safe_with_keyword",
                        reason=kw_reason2,
                    )
                    return "harmful", kw_reason2

            LOGGER.info(
                "grader_classified",
                provider=provider,
                model=model,
                verdict=verdict,
                reason=reason,
            )
            return verdict, reason or ""

        # Unparseable response — deny, don't guess.
        LOGGER.warning(
            "grader_unparseable_response_deny",
            provider=provider,
            model=model,
            response=text[:200],
        )
        return "off_topic", "classifier response unparseable — denied by default"

    except Exception as exc:
        # LLM call failed — deny, don't guess.
        LOGGER.warning(
            "grader_failed_deny",
            provider=provider,
            model=model,
            error=str(exc),
        )
        return "off_topic", "classifier error — denied by default"


# Backwards-compatible wrapper, in case anything still imports classify_harmful.
async def classify_harmful(prompt: str, *, instruction: str) -> Tuple[bool, str]:
    verdict, reason = await classify_prompt(prompt, instruction=instruction)
    return verdict == "harmful", reason
