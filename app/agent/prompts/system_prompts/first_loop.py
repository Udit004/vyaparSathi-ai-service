"""
app/agent/prompts/system_prompts/first_loop.py
================================================
Prompt fragment injected on the first loop when no tool data has been
gathered yet. Instructs the LLM to use the available tools to gather
the data it needs, without over-fetching.
"""

FIRST_LOOP = (
    "Use the available tools to gather the data you need. "
    "Call only the tools that are relevant — do not over-fetch.\n\n"
)