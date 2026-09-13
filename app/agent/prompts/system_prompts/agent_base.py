"""
app/agent/prompts/system_prompts/agent_base.py
==============================================
Identity, store context, goal, and loop counter — the core of the
system message that never changes across loops.
"""


def build_agent_base(*, store_id: str, user_prompt: str, current_goal: str, loop: int, max_loops: int) -> str:
    """
    Build the base system prompt fragment.

    Args:
        store_id:      The store this conversation is scoped to.
        user_prompt:   The raw user question/instruction.
        current_goal:  The normalized goal extracted by the think node.
        loop:          Current loop index (0-based).
        max_loops:     Hard ceiling on agent loops.

    Returns:
        A multi-line string ready to be concatenated into the system message.
    """
    return (
        "You are Vyapar Copilot, an AI assistant for Indian retail stores.\n"
        f"Store ID: {store_id}\n"
        f"User's question: \"{user_prompt}\"\n"
        f"Goal: {current_goal}\n"
        f"Current loop: {loop + 1} / {max_loops}\n\n"
    )