"""
Tool for retrieving store 30-day benchmarks, peak sales hours, and dead stock analytics.
Supported for both Text Copilot and Voice Agent.
"""
from __future__ import annotations

from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import structlog

from app.agent.service.analytics.store_baselines import get_store_baselines as fetch_store_baselines

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.store_baselines")


class StoreBaselinesInput(BaseModel):
    user_id: Optional[str] = Field(
        default=None,
        description="ID of the merchant user. Injected automatically if available.",
    )
    store_id: Optional[str] = Field(
        default=None,
        description="ID of the store. Injected automatically if available.",
    )


@tool("get_store_baselines", args_schema=StoreBaselinesInput)
async def get_store_baselines(
    user_id: Optional[str] = None,
    store_id: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Retrieve 30-day historical business baselines, average daily revenue, peak sales hours, and dead stock counts.
    Use this tool to compare current performance against store baselines or answer questions about peak business hours.
    """
    effective_user_id = user_id or kwargs.get("configurable", {}).get("user_id") or kwargs.get("user_id")
    effective_store_id = store_id or kwargs.get("configurable", {}).get("store_id") or kwargs.get("store_id")

    baselines = await fetch_store_baselines(user_id=effective_user_id, store_id=effective_store_id)
    if "error" in baselines:
        return {"status": "failed", "error": baselines["error"]}

    return {
        "status": "success",
        "store_baselines": baselines
    }
