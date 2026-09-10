"""
app/agent/utils.py
==================
Utility functions for the Vyapar Copilot agent.
Includes SSE streaming helpers and text formatting.
"""

from typing import Any
import json

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
