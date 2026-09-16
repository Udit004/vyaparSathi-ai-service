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

MEMORY_QUERY_INSTRUCTION = (
    "Create one concise semantic search query for retrieving relevant long-term "
    "memory for Vyapar Copilot. Use the current user request and the assistant "
    "scope below. Keep only stable topics, preferences, store facts, prior "
    "decisions, and recurring patterns that could help answer the request. "
    "Do not answer the request. Do not include IDs, secrets, refusal text, "
    "tool instructions, or temporary numeric results. Return one plain-text "
    "query of at most 30 words.\n\n"
    "Assistant scope: Indian retail inventory, sales, forecasting, restocking, "
    "operational insights, and user communication preferences."
)

MEMORY_EXTRACTION_INSTRUCTION = (
    "Extract only durable long-term memory from this retail assistant exchange. "
    "Keep store-specific facts, recurring inventory or sales patterns, explicit "
    "business decisions, and user preferences about language, tone, or detail. "
    "Exclude greetings, one-off questions, temporary tool results, model/refusal "
    "text, personal identifiers, credentials, and sensitive financial data. "
    "If there is nothing durable, return exactly NONE. Return concise plain text."
)