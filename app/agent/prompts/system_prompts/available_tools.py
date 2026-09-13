"""
app/agent/prompts/system_prompts/available_tools.py
====================================================
Prompt fragment listing the tools still available this loop.
"""


def build_available_tools(available: list[str]) -> str:
    """
    Build the "available tools this loop" prompt fragment.

    Args:
        available: List of tool names not yet called this loop.

    Returns:
        A single-line string, or empty string if no tools are available
        or if we're on the final loop (handled by the caller).
    """
    if not available:
        return ""
    return f"Available tools this loop: {', '.join(available)}\n"