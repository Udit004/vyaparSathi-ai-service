from __future__ import annotations

import json
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from app.lib.llm import get_llm
from app.lib.summarizer import summarize
from app.agent.memory.models import ExtractedMemory
from app.agent.memory.pinecone_client import query_vectors

LOGGER = structlog.get_logger("vyaparsathi.ai.memory.conflict")

class ResolutionAction(BaseModel):
    action: Literal["ADD", "UPDATE", "SUPERSEDE", "NO_CHANGE"]
    target_id: str | None = None
    reason: str = Field(description="Explanation of why this action was chosen.")

class ResolutionResult(BaseModel):
    actions: list[ResolutionAction] = Field(default_factory=list)

CONFLICT_SYSTEM_PROMPT = """You are a Memory Reconciliation Engine.
You must compare a NEW candidate memory against EXISTING memories and determine the correct action.

ACTIONS:
- ADD: The candidate is genuinely new information.
- NO_CHANGE: The candidate is a duplicate or provides no new value. (Update last_confirmed_at).
- SUPERSEDE: The candidate CONTRADICTS an existing active memory (e.g., preference changed).
- UPDATE: The candidate adds minor details to an existing memory, so we replace it.

RULES:
1. Do not supersede unless there is a clear contradiction.
2. If it's the exact same fact (e.g., "User prefers Hindi" and "User wants Hindi"), output NO_CHANGE and specify the target_id to confirm.
3. If it contradicts (e.g., "User prefers Hindi" vs "User wants English"), output SUPERSEDE with the target_id of the old memory.
"""

async def reconcile_memory(
    candidate: ExtractedMemory,
    filter_dict: dict,
) -> ResolutionAction:
    """
    Check if the candidate memory conflicts with or duplicates existing Pinecone memories.
    """
    existing = await query_vectors(filter_dict, candidate.content, top_k=5)
    
    # Filter for high relevance only
    relevant = [m for m in existing if m.get("score", 0.0) > 0.6]
    
    if not relevant:
        return ResolutionAction(action="ADD", reason="No relevant existing memories found.")

    existing_text = "\n".join(
        f"ID: {m['id']} | Status: {m['metadata'].get('status', 'active')} | Content: {m['text']}"
        for m in relevant
    )

    user_msg = (
        f"NEW CANDIDATE MEMORY:\nType: {candidate.memory_type}\nContent: {candidate.content}\n\n"
        f"EXISTING MEMORIES:\n{existing_text}\n\n"
        "What is the correct action?"
    )

    llm = get_llm()
    if not llm:
        # Fallback heuristic
        highest_score = max((m.get("score", 0.0) for m in relevant), default=0.0)
        if highest_score > 0.85:
            target = next(m for m in relevant if m.get("score") == highest_score)
            return ResolutionAction(action="NO_CHANGE", target_id=target["id"], reason="High similarity heuristic fallback.")
        return ResolutionAction(action="ADD", reason="Fallback heuristic.")

    structured_llm = llm.with_structured_output(ResolutionAction)
    
    try:
        result: ResolutionAction = await structured_llm.ainvoke([
            {"role": "system", "content": CONFLICT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])
        return result
    except Exception as exc:
        LOGGER.error("reconciliation_failed", error=str(exc))
        return ResolutionAction(action="ADD", reason="Error during reconciliation.")
