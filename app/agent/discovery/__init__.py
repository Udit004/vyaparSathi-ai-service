"""
app/agent/discovery/__init__.py
================================
Discovery layer for the Vyapar Copilot agent.

This package contains:
    filters.py — deterministic filtering and ranking logic

Purpose:
    Sits between Discovery Tools and the LLM context.
    Applies business rules without LLM reasoning.
"""

from app.agent.discovery.filters import (
    filter_candidates,
    rank_candidates,
    build_compact_context,
)

__all__ = [
    "filter_candidates",
    "rank_candidates",
    "build_compact_context",
]
