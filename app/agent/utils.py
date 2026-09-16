"""Small defensive helpers shared by agent routing nodes."""

from __future__ import annotations

from typing import Any


def message_text(value: Any) -> str:
    """Convert message content/state values into safe plain text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts)
    if isinstance(value, dict):
        text = value.get("text") or value.get("content")
        return text if isinstance(text, str) else ""
    return "" if value is None else str(value)


def latest_human_prompt(messages: Any, fallback: Any = "") -> str:
    """Return the newest human message as plain text, ignoring stale scalar state."""
    for message in reversed(messages or []):
        if getattr(message, "type", None) == "human":
            content = message_text(getattr(message, "content", ""))
            if content:
                return content
    return message_text(fallback)

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
