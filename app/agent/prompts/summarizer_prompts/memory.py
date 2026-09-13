"""
app/agent/prompts/summarizer_prompts/memory.py
================================================
Instruction for compressing mem0 long-term memory results.

Used by ``app/agent/memory.py`` to summarize the (potentially huge)
list of mem0 search results into a compact, factual summary of user
preferences and store knowledge before it enters the main LLM context.
"""

SUMMARIZER_MEMORY_INSTRUCTION = (
    "You are compressing long-term memory for an AI agent. "
    "Produce a compact, factual summary of the user preferences "
    "and store knowledge below. Keep only concrete facts, "
    "preferences, and patterns. Drop filler and duplicates. "
    "Return plain text, no markdown headers."
)