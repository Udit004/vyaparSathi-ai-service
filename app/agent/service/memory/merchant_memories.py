"""
Merchant Long-Term Personalization Memory Service.
Manages long-term merchant preferences, language choices, business targets, and key facts.
Collection: merchant_memories
"""
from __future__ import annotations

from datetime import datetime, timezone
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.memory.merchant_memories")

async def get_merchant_memory(user_id: str) -> dict:
    """
    Fetch merchant personalized memory document by user_id.
    """
    if not user_id:
        return {}

    db = get_database()
    doc = await db["merchant_memories"].find_one({"user_id": str(user_id)})
    if not doc:
        # Create default empty profile memory
        doc = {
            "user_id": str(user_id),
            "preferred_language": "hinglish",
            "communication_channel": "email",
            "business_profile": {
                "primary_category": "General Retail",
                "min_desired_margin_pct": 15.0,
                "default_reorder_buffer_days": 7
            },
            "key_facts": [],
            "frequently_asked_topics": [],
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        await db["merchant_memories"].insert_one(doc)

    doc["_id"] = str(doc.get("_id"))
    return doc

async def update_merchant_memory(
    user_id: str,
    preferred_language: str | None = None,
    communication_channel: str | None = None,
    business_profile: dict | None = None,
    new_facts: list[str] | None = None,
) -> dict:
    """
    Update or append merchant memories, preferences, and key business facts.
    """
    if not user_id:
        return {"error": "user_id is required"}

    db = get_database()
    current = await get_merchant_memory(user_id)

    update_fields: dict = {
        "updated_at": datetime.now(timezone.utc)
    }

    if preferred_language:
        update_fields["preferred_language"] = preferred_language
    if communication_channel:
        update_fields["communication_channel"] = communication_channel
    if business_profile:
        current_profile = current.get("business_profile", {})
        current_profile.update(business_profile)
        update_fields["business_profile"] = current_profile

    if update_fields:
        await db["merchant_memories"].update_one(
            {"user_id": str(user_id)},
            {"$set": update_fields},
            upsert=True
        )

    if new_facts:
        clean_facts = [f.strip() for f in new_facts if f and isinstance(f, str)]
        if clean_facts:
            await db["merchant_memories"].update_one(
                {"user_id": str(user_id)},
                {"$addToSet": {"key_facts": {"$each": clean_facts}}},
                upsert=True
            )

    LOGGER.info("merchant_memory_updated", user_id=user_id)
    return await get_merchant_memory(user_id)
