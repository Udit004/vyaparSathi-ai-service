"""
app/services/chat_service.py
============================
Simple LLM chat endpoint — no agent graph, no tool calls.

All Copilot / agentic logic has moved to ``app/agent/``:
    app/agent/state.py        — VyaparAgentState
    app/agent/checkpointer.py — MongoDB checkpoint (short-term memory)
    app/agent/tools/          — read-only tool implementations
    app/agent/graph.py        — looping agent graph
    app/agent/utils.py        — SSE + text helpers
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from app.lib.llm import get_llm
from app.schemas.chat import ChatRequest, ChatResponse

LOGGER = structlog.get_logger("vyaparsathi.ai.chat")


def _as_text(value: Any) -> str:
    """Coerce any LLM content value to a plain string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_as_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


def generate_chat_response(request: ChatRequest) -> ChatResponse:
    """Direct single-turn LLM chat — no agent loop, no tools."""
    llm = get_llm()

    if not llm:
        LOGGER.warning("chat_llm_unavailable", reason="gemini_api_key_not_configured")
        return ChatResponse(
            response="AI chat is unavailable because the Gemini API key is not configured."
        )

    response = llm.invoke(request.message)
    return ChatResponse(response=_as_text(getattr(response, "content", "")))
