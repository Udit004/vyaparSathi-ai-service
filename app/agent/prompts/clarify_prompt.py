"""
app/agent/prompts/clarify_prompt.py
=====================================
System instruction injected into the think node when the graph
resumes after a user clarification answer.

This tells the LLM that a previous clarification was answered and
it should incorporate that answer into its reasoning.
"""

CLARIFICATION_CONTEXT_TEMPLATE = (
    "\n\n--- Clarification Context ---\n"
    "A previous clarification question was asked and the user has "
    "now answered it. Incorporate this answer into your analysis:\n\n"
    "Previous question: \"{question}\"\n"
    "User's answer: \"{answer}\"\n"
    "Why it was needed: {reason}\n"
    "---\n\n"
)
