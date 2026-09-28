"""
app/agent/nodes/context_node.py
================================
Context node — fetches a lightweight user and store snapshot from
MongoDB once per agent run, BEFORE the grader/think nodes execute.

Why here?
---------
The think node builds the system prompt on every loop. Having the
owner's name, store name, currency, and key thresholds available from
loop 0 lets the LLM:

  • Greet the owner by name
  • Use the correct currency symbol throughout
  • Know the store's low-stock threshold when interpreting inventory data
  • Adapt tone to the business type (retail vs wholesale)

Design constraints
------------------
  • ONE MongoDB round-trip per field (two total: users + stores).
    Both are fired concurrently with asyncio.gather so latency is
    bounded by the slower of the two (~1-5 ms on a local/Atlas cluster).
  • No write operations — this is a pure read node.
  • Graceful degradation — if either query fails the node logs a warning
    and writes an empty dict; the rest of the graph continues unchanged.
  • Collection names match the Mongoose models used by the Express
    backend: ``users`` and ``stores``.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Any

import structlog
from bson import ObjectId
from bson.errors import InvalidId

from app.agent.state import VyaparAgentState
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.context")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _fetch_user(db, user_id: str) -> dict[str, Any]:
    """
    Fetch a compact user snapshot from the ``users`` collection.

    Returns {} on any error (invalid ObjectId, document not found,
    network issue) so the caller never needs to handle exceptions.
    """
    try:
        oid = ObjectId(user_id)
    except (InvalidId, TypeError):
        LOGGER.warning("context_node_invalid_user_id", user_id=user_id)
        return {}

    try:
        doc = await db["users"].find_one(
            {"_id": oid},
            {
                "name": 1,
                "email": 1,
                "preferences": 1,
                "user_preferences": 1,
                "communication_style": 1,
                "preferred_language": 1,
                "_id": 0,
            },
        )
        if not doc:
            LOGGER.warning("context_node_user_not_found", user_id=user_id)
            return {}

        prefs = doc.get("preferences") or doc.get("user_preferences") or {}
        if not isinstance(prefs, dict):
            prefs = {}

        if doc.get("communication_style"):
            prefs["communication_style"] = doc.get("communication_style")
        if doc.get("preferred_language"):
            prefs["preferred_language"] = doc.get("preferred_language")

        return {
            "name": doc.get("name", ""),
            "email": doc.get("email", ""),
            "preferences": prefs,
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "context_node_user_fetch_error",
            user_id=user_id,
            error=str(exc),
        )
        return {}


async def _fetch_store(db, store_id: str) -> dict[str, Any]:
    """
    Fetch a compact store snapshot from the ``stores`` collection.

    Accepts either a valid 24-char hex ObjectId string or a store name.
    Returns {} on any error so the caller never needs to handle
    exceptions.
    """
    try:
        oid = ObjectId(store_id)
    except (InvalidId, TypeError):
        # Try looking up by store name
        doc = await db["stores"].find_one(
            {"name": store_id},
            {"_id": 1},
        )
        if not doc:
            LOGGER.warning("context_node_store_not_found", store_id=store_id)
            return {}
        oid = doc["_id"]

    try:
        doc = await db["stores"].find_one(
            {"_id": oid},
            {
                "name": 1,
                "businessType": 1,
                "address.city": 1,
                "settings.currency": 1,
                "settings.lowStockThreshold": 1,
                "settings.leadTimeDays": 1,
                "_id": 0,
            },
        )
        if not doc:
            LOGGER.warning("context_node_store_not_found", store_id=store_id)
            return {}

        settings = doc.get("settings", {}) or {}
        address = doc.get("address", {}) or {}

        return {
            "name": doc.get("name", ""),
            "business_type": doc.get("businessType", "retail"),
            "city": address.get("city", ""),
            "currency": settings.get("currency", "INR"),
            "low_stock_threshold": settings.get("lowStockThreshold", 10),
            "lead_time_days": settings.get("leadTimeDays", 3),
        }
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "context_node_store_fetch_error",
            store_id=store_id,
            error=str(exc),
        )
        return {}


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


async def context_node(state: VyaparAgentState) -> Dict[str, Any]:
    """
    Load a lightweight user + store snapshot into the graph state.

    Runs ONCE at the start of every agent run (before the grader).
    Uses asyncio.gather to fire both MongoDB queries concurrently.

    On success, writes:
        user_context  → {name, email}
        store_context → {name, business_type, city, currency,
                         low_stock_threshold, lead_time_days}

    On failure (any query), writes empty dicts so the graph continues
    unaffected and the think node degrades gracefully to generic
    prompts without owner/store personalisation.
    """
    user_id = state.get("user_id", "")
    store_id = state.get("store_id", "")

    LOGGER.info(
        "context_node_start",
        user_id=user_id,
        store_id=store_id,
    )

    try:
        db = get_database()
        user_ctx, store_ctx = await asyncio.gather(
            _fetch_user(db, user_id),
            _fetch_store(db, store_id),
        )
    except Exception as exc:  # noqa: BLE001
        # DB client itself failed — degrade gracefully
        LOGGER.warning(
            "context_node_db_error",
            error=str(exc),
        )
        user_ctx, store_ctx = {}, {}

    LOGGER.info(
        "context_node_complete",
        user_name=user_ctx.get("name", "<unknown>"),
        store_name=store_ctx.get("name", "<unknown>"),
        store_city=store_ctx.get("city", ""),
        currency=store_ctx.get("currency", "INR"),
    )

    return {
        "user_context": user_ctx,
        "store_context": store_ctx,
    }
