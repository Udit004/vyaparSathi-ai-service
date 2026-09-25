from __future__ import annotations

import datetime
from typing import Any

import structlog
from pydantic import ValidationError

from app.lib.llm import get_llm
from app.agent.memory.models import ExtractionResult, ExtractedMemory

LOGGER = structlog.get_logger("vyaparsathi.ai.memory.extractor")

EXTRACTION_SYSTEM_PROMPT = """You are a highly analytical memory extraction system for a business assistant.
Your job is to read the recent conversation and extract atomic, structured memories that are worth retaining long-term.

CRITICAL RULES:
1. DO NOT extract trivial conversation (greetings, simple questions).
2. DO NOT store live business values (e.g., current stock is 42) as semantic memory, as they change rapidly.
3. PRESERVE PROVENANCE: 
   - If the user stated a fact, source_role = "user".
   - If the assistant inferred or recommended something, source_role = "assistant".
   - DO NOT convert an assistant recommendation into a user fact.
4. ATOMICITY: Prefer several small atomic memories over one large summary.
5. CONFIDENCE & IMPORTANCE: Rate confidence (0.0-1.0) based on direct evidence. Rate importance based on long-term utility.
6. DO NOT invent dates if not provided.

MEMORY TYPES TO EXTRACT:
- user_preference: Stable user preferences (e.g., language, style).
- user_fact: Stable information about the user.
- store_fact: Stable business/store information.
- store_pattern: Repeated historical behavior (only if explicitly identified as a pattern).
- episodic_event: Something that happened at a specific point in time.
- decision: A decision made by the user/business.
- assistant_recommendation: A recommendation by the assistant worth keeping.
"""

async def extract_memories(
    user_id: str,
    store_id: str,
    messages: list[dict],
    tool_results: list[dict] | None = None,
) -> list[ExtractedMemory]:
    """
    Extracts structured memories from the conversation using LLM structured output.
    """
    if not messages:
        return []

    llm = get_llm()
    if not llm:
        LOGGER.warning("memory_extraction_skipped", reason="no LLM available")
        return []

    structured_llm = llm.with_structured_output(ExtractionResult)

    now = datetime.datetime.now(datetime.timezone.utc)
    
    # Format the input
    convo_text = "\n".join(
        f"{str(m.get('role', 'unknown')).upper()}: {m.get('content', '')}"
        for m in messages
    )
    
    user_msg = (
        f"CURRENT UTC TIME: {now.isoformat()}\n\n"
        f"CONVERSATION:\n{convo_text}\n\n"
        "Extract valid memories."
    )

    try:
        LOGGER.info("memory_extraction_started", user_id=user_id, store_id=store_id)
        result: ExtractionResult = await structured_llm.ainvoke([
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])
        
        memories = result.memories if result and result.memories else []
        LOGGER.info("memory_extraction_completed", extracted_count=len(memories))
        return memories

    except Exception as exc:
        LOGGER.error("memory_extraction_failed", error=str(exc))
        return []
