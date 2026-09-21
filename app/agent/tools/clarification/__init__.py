from pydantic import BaseModel, Field
from langchain_core.tools import tool


class AskForClarificationInput(BaseModel):
    reason: str = Field(
        ...,
        description="A brief explanation of why clarification is needed and what information is missing.",
    )


@tool("ask_for_clarification", args_schema=AskForClarificationInput)
async def ask_for_clarification(reason: str) -> str:
    """
    Ask the user a clarification question when the agent is uncertain
    about what the user wants, needs critical information not provided,
    or cannot answer confidently with available tools.

    Use this tool when:
    - The user's question is ambiguous or too vague to answer with available tools
    - You need specific parameters (e.g., date range, product name, store location)
    - You're unsure which tool to call or what the user is looking for
    - You've exhausted your available tools and still can't answer confidently

    Do NOT use this tool when you can answer the question directly or
    when you can call a relevant tool to get the needed information.
    """
    return f"Clarification needed: {reason}"
