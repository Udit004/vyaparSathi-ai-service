from langchain_core.tools import tool
from pydantic import BaseModel, Field

class NavigateInput(BaseModel):
    page: str = Field(description="The name of the page to navigate to. Allowed values: 'analytics', 'sellers', 'buyers', 'purchases', 'billing', 'billing-history', 'overview', 'ai-dashboard', 'automations', 'staff', 'settings'")

@tool("tool_navigate_page", args_schema=NavigateInput)
def tool_navigate_page(page: str) -> str:
    """Navigates the user to a specific page in the Vyapar Sathi store dashboard. Use this tool when the user asks to open or go to a specific page or section of the store."""
    return f"Successfully navigated to {page}."
