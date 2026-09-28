"""
app/agent/prompts/system_prompts/first_loop.py
================================================
Prompt fragment injected on the first loop when no tool data has been
gathered yet. Guides the LLM to select the correct tool(s) efficiently.
"""

FIRST_LOOP = (
    "INITIAL TURN GUIDANCE:\n"
    "- If the user request is a simple greeting ('hi', 'hello', 'hey', 'good morning') or casual conversation, respond directly, warmly, and concisely without calling tools or using report tables.\n"
    "- For data/business queries, review available tools and call the most relevant tool(s).\n"
    "- Call high-level subgraphs (e.g. `invoke_morning_briefing`, `web_research`) if the request is broad or requires deep research.\n"
    "- Call complementary atomic tools in PARALLEL if multiple data aspects are requested (e.g. low stock + restock forecast).\n"
    "- Do not call unnecessary tools — gather precisely what is needed.\n\n"
)