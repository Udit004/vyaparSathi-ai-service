from langchain_core.tools import tool
from pydantic import BaseModel, Field

class NavigateInput(BaseModel):
    route: str = Field(description="The route of the page to navigate to. Allowed routes: '/storeDashboard/:storeId/analytics', '/storeDashboard/:storeId/sellers', '/storeDashboard/:storeId/buyers', '/storeDashboard/:storeId/purchases', '/storeDashboard/:storeId/billing', '/storeDashboard/:storeId/billing-history', '/storeDashboard/:storeId/overview', '/storeDashboard/:storeId/ai-dashboard', '/storeDashboard/:storeId/automations', '/storeDashboard/:storeId/staff', '/storeDashboard/:storeId/settings'")

@tool("tool_navigate_page", args_schema=NavigateInput)
def tool_navigate_page(route: str) -> str:
    """Navigates the user to a specific page in the Vyapar Sathi store dashboard. Use this tool when the user asks to open or go to a specific page or section of the store."""
    return f"Successfully navigated to {route}."
