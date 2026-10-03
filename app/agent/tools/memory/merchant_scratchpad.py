"""
app/agent/tools/memory/merchant_scratchpad.py
=============================================
Redis-backed Merchant Diary & Temporary Scratchpad for multi-turn task planning.
Allows both Text Copilot and Voice Assistant to create, inspect, modify, and finalize
working drafts (Purchase Orders, Todo Notes, Restock Plans, Email Drafts).
"""

from __future__ import annotations

import datetime
import json
import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import structlog

from app.agent.memory.redis_cache import get_redis

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.scratchpad")

_SCRATCHPAD_KEY_PREFIX = "vyapar:scratchpad:"
_SCRATCHPAD_TTL = 7 * 86400  # 7 days


def _scratchpad_key(target_id: str) -> str:
    return f"{_SCRATCHPAD_KEY_PREFIX}{target_id}"


class WriteScratchpadInput(BaseModel):
    title: str = Field(..., description="Short descriptive title of the note or draft (e.g. 'Draft Purchase Order for Ramesh Traders').")
    content: str = Field(..., description="The main text body or markdown content of the note.")
    category: str = Field(
        default="general_note",
        description="Category: 'purchase_order_draft', 'restock_plan', 'customer_udhaar_note', 'general_note', or 'todo'.",
    )
    draft_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured JSON data for the draft (e.g. items list, quantities, supplier info).",
    )
    store_id: Optional[str] = Field(default=None, description="The store ID.")
    user_id: Optional[str] = Field(default=None, description="The user/owner ID.")


class ReadScratchpadInput(BaseModel):
    category: Optional[str] = Field(
        default=None,
        description="Optional filter by category (e.g. 'purchase_order_draft', 'restock_plan', 'general_note'). Leave empty to fetch all.",
    )
    query: Optional[str] = Field(
        default=None,
        description="Optional keyword search to filter notes by title or content.",
    )
    store_id: Optional[str] = Field(default=None, description="The store ID.")
    user_id: Optional[str] = Field(default=None, description="The user/owner ID.")


class UpdateScratchpadInput(BaseModel):
    note_id: str = Field(..., description="The unique ID of the draft note to update.")
    updated_content: Optional[str] = Field(None, description="New content text if updating.")
    updated_draft_data: Optional[Dict[str, Any]] = Field(None, description="Updated structured JSON data.")
    status: Optional[str] = Field(None, description="Status update: 'draft', 'in_progress', 'completed', or 'cancelled'.")
    store_id: Optional[str] = Field(default=None, description="The store ID.")
    user_id: Optional[str] = Field(default=None, description="The user/owner ID.")


class DeleteScratchpadInput(BaseModel):
    note_id: str = Field(..., description="The unique ID of the draft note to delete.")
    store_id: Optional[str] = Field(default=None, description="The store ID.")
    user_id: Optional[str] = Field(default=None, description="The user/owner ID.")


@tool("write_scratchpad_note", args_schema=WriteScratchpadInput)
async def write_scratchpad_note(
    title: str,
    content: str,
    category: str = "general_note",
    draft_data: Optional[Dict[str, Any]] = None,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Save a working draft, temporary note, purchase order plan, or execution diary entry into Redis storage.
    Both Text and Voice assistants can read and modify this draft in future turns.

    Use this when:
    - You prepare a draft purchase order, supplier quote, or restock plan that needs review before sending.
    - The merchant asks to "Note this down in my diary", "Keep a draft of...", or "Save this restock list".
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")
    target_id = store_id or user_id

    if not target_id:
        return {"success": False, "message": "Identity missing from context."}

    note_id = f"note_{uuid.uuid4().hex[:8]}"
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    note_record = {
        "note_id": note_id,
        "title": title,
        "content": content,
        "category": category,
        "draft_data": draft_data or {},
        "status": "draft",
        "created_at": now_iso,
        "updated_at": now_iso,
    }

    try:
        redis = await get_redis()
        if redis:
            key = _scratchpad_key(target_id)
            raw = await redis.get(key)
            notes = json.loads(raw) if raw else []
            notes.insert(0, note_record)
            await redis.set(key, json.dumps(notes[:40]), ex=_SCRATCHPAD_TTL)
            LOGGER.info("scratchpad_note_saved", note_id=note_id, title=title[:40])
    except Exception as exc:
        LOGGER.warning("scratchpad_save_error", error=str(exc))
        return {"success": False, "message": f"Failed to save note: {exc}"}

    return {
        "success": True,
        "message": f"Successfully saved to Merchant Diary: '{title}' (ID: {note_id})",
        "note": note_record,
    }


@tool("read_scratchpad_notes", args_schema=ReadScratchpadInput)
async def read_scratchpad_notes(
    category: Optional[str] = None,
    query: Optional[str] = None,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Read active drafts, purchase orders, diary entries, or notes from the Merchant Diary.

    Use this when:
    - The merchant asks "Mera draft PO dikhao", "What notes do I have in my diary?", "Check active restock drafts".
    - You need to retrieve a previously drafted purchase order before sending or updating it.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")
    target_id = store_id or user_id

    if not target_id:
        return {"found": False, "notes": [], "message": "Identity missing from context."}

    all_notes = []
    try:
        redis = await get_redis()
        if redis:
            key = _scratchpad_key(target_id)
            raw = await redis.get(key)
            if raw:
                all_notes = json.loads(raw)
    except Exception as exc:
        LOGGER.warning("scratchpad_read_error", error=str(exc))

    filtered = []
    for n in all_notes:
        if category and n.get("category") != category:
            continue
        if query:
            q_lower = query.lower().strip()
            draft_str = json.dumps(n.get("draft_data", {})) if isinstance(n.get("draft_data"), (dict, list)) else ""
            t_lower = (n.get("title", "") + " " + n.get("content", "") + " " + draft_str).lower()
            if q_lower not in t_lower:
                continue
        filtered.append(n)

    return {
        "found": len(filtered) > 0,
        "total_notes": len(filtered),
        "notes": filtered[:15],
    }


@tool("update_scratchpad_note", args_schema=UpdateScratchpadInput)
async def update_scratchpad_note(
    note_id: str,
    updated_content: Optional[str] = None,
    updated_draft_data: Optional[Dict[str, Any]] = None,
    status: Optional[str] = None,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Update or modify an existing draft note or purchase order in the Merchant Diary.

    Use this when:
    - The merchant asks to change order quantities, add more items, or mark a draft as completed/cancelled.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")
    target_id = store_id or user_id

    if not target_id:
        return {"success": False, "message": "Identity missing from context."}

    found = False
    updated_note = None
    try:
        redis = await get_redis()
        if redis:
            key = _scratchpad_key(target_id)
            raw = await redis.get(key)
            notes = json.loads(raw) if raw else []
            for n in notes:
                if n.get("note_id") == note_id:
                    if updated_content is not None:
                        n["content"] = updated_content
                    if updated_draft_data is not None:
                        n["draft_data"] = {**n.get("draft_data", {}), **updated_draft_data}
                    if status is not None:
                        n["status"] = status
                    n["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    updated_note = n
                    found = True
                    break

            if found:
                await redis.set(key, json.dumps(notes), ex=_SCRATCHPAD_TTL)
                LOGGER.info("scratchpad_note_updated", note_id=note_id)
    except Exception as exc:
        LOGGER.warning("scratchpad_update_error", error=str(exc))
        return {"success": False, "message": f"Error updating note: {exc}"}

    if not found:
        return {"success": False, "message": f"Note with ID '{note_id}' not found."}

    return {
        "success": True,
        "message": f"Successfully updated note '{note_id}'.",
        "note": updated_note,
    }


@tool("delete_scratchpad_note", args_schema=DeleteScratchpadInput)
async def delete_scratchpad_note(
    note_id: str,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Remove or discard a draft note from the Merchant Diary.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")
    target_id = store_id or user_id

    try:
        redis = await get_redis()
        if redis:
            key = _scratchpad_key(target_id)
            raw = await redis.get(key)
            notes = json.loads(raw) if raw else []
            new_notes = [n for n in notes if n.get("note_id") != note_id]
            await redis.set(key, json.dumps(new_notes), ex=_SCRATCHPAD_TTL)
            LOGGER.info("scratchpad_note_deleted", note_id=note_id)
            return {"success": True, "message": f"Note '{note_id}' removed from diary."}
    except Exception as exc:
        LOGGER.warning("scratchpad_delete_error", error=str(exc))
        return {"success": False, "message": f"Failed to delete note: {exc}"}

    return {"success": False, "message": "Failed to delete note."}
