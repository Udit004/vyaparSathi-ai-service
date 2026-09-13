"""
app/agent/prompts/summarizer_prompts/tool.py
==============================================
Instruction for compressing tool payloads.

Used by ``app/agent/nodes/observe_node.py`` to summarize large tool
responses (inventory lists, sales histories, forecasts, etc.) before
they become ToolMessages in the main LLM context window. The full raw
data remains in the *_context buckets and persisted state — only the
message the LLM actually reads is compressed.
"""


def build_summarizer_tool_instruction(tool_name: str) -> str:
    """
    Build the summarizer instruction for a specific tool's output.

    Args:
        tool_name: Name of the tool whose output is being summarized.

    Returns:
        A prompt string tailored to that tool.
    """
    return (
        f"You are compressing the output of the '{tool_name}' "
        "tool for an AI agent's context window. "
        "Produce a concise, factual summary that preserves: "
        "the key numbers, totals, counts, and any list items "
        "the agent would need to reason about. "
        "Drop formatting noise and redundant detail. "
        "Return plain text, no markdown headers."
    )