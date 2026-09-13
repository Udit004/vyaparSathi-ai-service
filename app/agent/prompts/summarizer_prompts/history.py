"""
app/agent/prompts/summarizer_prompts/history.py
=================================================
Instruction for compressing the sliding-window history.

Used by ``app/agent/nodes/think_node.py`` to summarize the older
messages in the conversation into a compact summary before they enter
the main LLM context window. Preserves the user's original question,
every tool that was called and its key findings, and any decisions
made.
"""

SUMMARIZER_HISTORY_INSTRUCTION = (
    "You are compressing an agent's earlier conversation into a "
    "compact summary for the LLM's context window. Preserve: the "
    "user's original question, every tool that was called and its "
    "key findings, and any decisions made. Drop filler, repeated "
    "data, and formatting noise. Return plain text, no headers."
)