"""
app/agent/prompts/system_prompts/memory_warning.py
===================================================
Warning injected when long-term memory context is present.

The LLM must understand that mem0 memory contains user preferences and
store knowledge ONLY — it is NOT a substitute for live tool data. Any
question about inventory, sales, forecasts, restock, or insights MUST
call the relevant tools, because memory is stale by design.
"""

MEMORY_WARNING = (
    "WARNING: The long-term memory above contains user preferences and store "
    "knowledge only. It does NOT contain current inventory, sales, forecast, "
    "or insight data. If the user's question requires any of that data, you "
    "MUST call the relevant tool(s) to fetch live data. Do NOT answer from "
    "memory alone — memory is stale by design and will give incorrect results.\n\n"
)