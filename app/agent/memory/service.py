from __future__ import annotations

import structlog

from app.agent.memory.extractor import extract_memories
from app.agent.memory.conflict import reconcile_memory
from app.agent.memory.writer import write_new_memory, supersede_memory, confirm_memory

LOGGER = structlog.get_logger("vyaparsathi.ai.memory.service")

async def process_and_persist_memory(
    *,
    user_id: str,
    store_id: str,
    messages: list[dict],
    store_ids: list[str] | None = None,
) -> dict[str, bool]:
    """
    Main Multi-Level Memory Pipeline.
    
    1. Extract structured memories using LLM.
    2. Loop over extracted memories.
    3. Determine scope (user vs store vs multi-store) based on memory_type.
    4. Reconcile against existing Pinecone vectors (Conflict & Deduplication).
    5. Write / Supersede / Confirm.
    """
    if not messages:
        return {"user_ok": True, "store_ok": True, "multi_store_ok": True}

    memories = await extract_memories(user_id=user_id, store_id=store_id, messages=messages)
    
    if not memories:
        LOGGER.info("pinecone_memory_pipeline_complete_no_memories")
        return {"user_ok": True, "store_ok": True, "multi_store_ok": True}

    results = {"inserted": 0, "updated": 0, "superseded": 0, "skipped": 0}

    for memory in memories:
        # Determine the scope and filter dict
        filter_dict = {"status": "active"}
        extra_metadata = {}
        
        if memory.memory_type in ("user_preference", "user_fact"):
            filter_dict["user_id"] = user_id
            filter_dict["scope"] = "user"
            extra_metadata["user_id"] = user_id
            extra_metadata["scope"] = "user"
            memory.user_id = user_id
            
        elif memory.memory_type in ("store_fact", "store_pattern"):
            filter_dict["store_id"] = store_id
            filter_dict["scope"] = "store"
            extra_metadata["store_id"] = store_id
            extra_metadata["scope"] = "store"
            memory.store_id = store_id
            
        else:
            # Default to store context for episodic events, decisions, and recommendations, 
            # unless it explicitly mentions multi-store strategy, but we keep it simple for now.
            filter_dict["store_id"] = store_id
            filter_dict["scope"] = "store"
            extra_metadata["store_id"] = store_id
            extra_metadata["scope"] = "store"
            memory.store_id = store_id

        # Reconcile memory (Deduplication & Conflict)
        resolution = await reconcile_memory(memory, filter_dict)
        
        if resolution.action == "ADD":
            await write_new_memory(memory, extra_metadata)
            results["inserted"] += 1
            
        elif resolution.action == "NO_CHANGE" and resolution.target_id:
            await confirm_memory(resolution.target_id)
            results["skipped"] += 1
            
        elif resolution.action == "SUPERSEDE" and resolution.target_id:
            await supersede_memory(resolution.target_id, memory, extra_metadata)
            results["superseded"] += 1
            
        elif resolution.action == "UPDATE" and resolution.target_id:
            # We treat update as a supersede for auditability
            await supersede_memory(resolution.target_id, memory, extra_metadata)
            results["updated"] += 1

    LOGGER.info("pinecone_memory_pipeline_complete", stats=results)
    return {"user_ok": True, "store_ok": True, "multi_store_ok": True}
