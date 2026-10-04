"""
Tool for fetching and managing proactive business insights and suggestions.
Supported for both Text Copilot and Voice Agent.
"""
from __future__ import annotations

from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import structlog

from app.agent.service.insights.proactive_insights import (
    get_unread_insights,
    generate_proactive_insights_for_merchant,
    dismiss_proactive_insight,
)

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.proactive_insights")


class ProactiveInsightsInput(BaseModel):
    action: str = Field(
        ...,
        description="Action to perform: 'get' to fetch active proactive recommendations, 'generate' to refresh insights, 'dismiss' to clear an insight.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="ID of the merchant user. Injected automatically if available.",
    )
    store_id: Optional[str] = Field(
        default=None,
        description="ID of the store. Injected automatically if available.",
    )
    insight_id: Optional[str] = Field(
        default=None,
        description="ID of the specific proactive insight to dismiss (required if action='dismiss').",
    )


@tool("manage_proactive_insights", args_schema=ProactiveInsightsInput)
async def manage_proactive_insights(
    action: str,
    user_id: Optional[str] = None,
    store_id: Optional[str] = None,
    insight_id: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Fetch or manage proactive insights, automated reorder alerts, and risk warnings for the merchant.
    Use this tool when greeting the user or when asking for proactive business suggestions.
    """
    effective_user_id = user_id or kwargs.get("configurable", {}).get("user_id") or kwargs.get("user_id")
    effective_store_id = store_id or kwargs.get("configurable", {}).get("store_id") or kwargs.get("store_id")

    if not effective_user_id:
        return {"error": "Merchant user_id is required to fetch proactive insights."}

    if action == "get":
        insights = await get_unread_insights(user_id=effective_user_id, store_id=effective_store_id)
        if not insights:
            # Try generating fresh insights
            insights = await generate_proactive_insights_for_merchant(user_id=effective_user_id, store_id=effective_store_id)

        return {
            "status": "success",
            "insights_count": len(insights),
            "proactive_insights": insights
        }
    elif action == "generate":
        insights = await generate_proactive_insights_for_merchant(user_id=effective_user_id, store_id=effective_store_id)
        return {
            "status": "success",
            "message": "Fresh proactive insights generated.",
            "insights_count": len(insights),
            "proactive_insights": insights
        }
    elif action == "dismiss":
        if not insight_id:
            return {"error": "insight_id is required to dismiss a proactive insight."}
        success = await dismiss_proactive_insight(user_id=effective_user_id, insight_id=insight_id)
        return {
            "status": "success" if success else "failed",
            "message": "Insight dismissed successfully." if success else "Insight not found or already dismissed."
        }
    else:
        return {"error": f"Invalid action '{action}'. Supported actions: 'get', 'generate', 'dismiss'."}
