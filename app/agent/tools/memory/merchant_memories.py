"""
Tool for managing long-term merchant preferences and business memory.
Supports both Text Copilot and Voice Agent.
"""
from __future__ import annotations

from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import structlog

from app.agent.service.memory.merchant_memories import (
    get_merchant_memory,
    update_merchant_memory,
)

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.merchant_memories")


class MerchantMemoryInput(BaseModel):
    action: str = Field(
        ...,
        description="Action to perform: 'get' to fetch merchant memories, 'update' to modify preferences/facts.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="ID of the user/merchant. Injected automatically if available.",
    )
    preferred_language: Optional[str] = Field(
        default=None,
        description="Preferred language for conversation (e.g. English, Hindi, Hinglish).",
    )
    communication_channel: Optional[str] = Field(
        default=None,
        description="Preferred contact channel for reports (e.g. email, whatsapp, voice).",
    )
    new_facts: Optional[list[str]] = Field(
        default=None,
        description="List of key facts or habits to remember about this merchant (e.g. ['Closed on Sundays', 'Prefers Amul distributor for dairy']).",
    )


@tool("manage_merchant_memories", args_schema=MerchantMemoryInput)
async def manage_merchant_memories(
    action: str,
    user_id: Optional[str] = None,
    preferred_language: Optional[str] = None,
    communication_channel: Optional[str] = None,
    new_facts: Optional[list[str]] = None,
    **kwargs,
) -> dict:
    """
    Fetch or update long-term merchant preferences, habits, business targets, and key facts.
    Use this tool to personalize responses or record user preferences expressed during chat or voice calls.
    """
    effective_user_id = user_id or kwargs.get("configurable", {}).get("user_id") or kwargs.get("user_id")

    if not effective_user_id:
        return {"error": "Merchant user_id is required to access or save memories."}

    if action == "get":
        memory = await get_merchant_memory(effective_user_id)
        return {
            "status": "success",
            "merchant_memory": memory
        }
    elif action == "update":
        updated = await update_merchant_memory(
            user_id=effective_user_id,
            preferred_language=preferred_language,
            communication_channel=communication_channel,
            new_facts=new_facts,
        )
        return {
            "status": "success",
            "message": "Merchant memory updated successfully.",
            "merchant_memory": updated
        }
    else:
        return {"error": f"Invalid action '{action}'. Supported actions: 'get', 'update'."}
