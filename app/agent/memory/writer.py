from __future__ import annotations

import datetime
from typing import Any

import structlog

from app.agent.memory.models import ExtractedMemory
from app.agent.memory.pinecone_client import upsert_vector, _get_pinecone_index

LOGGER = structlog.get_logger("vyaparsathi.ai.memory.writer")

async def write_new_memory(memory: ExtractedMemory, extra_metadata: dict[str, Any]) -> str | None:
    """Writes a brand new memory to Pinecone."""
    now = datetime.datetime.now(datetime.timezone.utc)
    memory.created_at = now
    memory.last_confirmed_at = now
    memory.status = "active"
    
    metadata = memory.model_dump(exclude_none=True)
    metadata.update(extra_metadata)
    
    # Don't embed the massive object, just the content for vector search
    text_to_embed = memory.content
    
    memory_id = await upsert_vector(text_to_embed, metadata)
    if memory_id:
        LOGGER.info("memory_inserted", memory_id=memory_id, type=memory.memory_type)
    return memory_id


async def supersede_memory(old_id: str, new_memory: ExtractedMemory, extra_metadata: dict[str, Any]) -> str | None:
    """Supersedes an old memory with a new one."""
    import asyncio
    
    now = datetime.datetime.now(datetime.timezone.utc)
    
    # 1. Update old memory
    index = _get_pinecone_index()
    if index:
        try:
            # We must fetch the old memory to keep its values, but Pinecone updates just merge metadata
            await asyncio.to_thread(
                index.update,
                id=old_id,
                set_metadata={"status": "superseded", "valid_to": now.isoformat()}
            )
            LOGGER.info("memory_superseded", old_id=old_id)
        except Exception as e:
            LOGGER.error("failed_to_supersede_old_memory", old_id=old_id, error=str(e))
            
    # 2. Write new memory
    new_memory.supersedes = old_id
    return await write_new_memory(new_memory, extra_metadata)


async def confirm_memory(memory_id: str) -> bool:
    """Updates the last_confirmed_at timestamp of an existing memory."""
    import asyncio
    
    index = _get_pinecone_index()
    if not index:
        return False
        
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        await asyncio.to_thread(
            index.update,
            id=memory_id,
            set_metadata={"last_confirmed_at": now}
        )
        LOGGER.info("memory_confirmed", memory_id=memory_id)
        return True
    except Exception as e:
        LOGGER.error("failed_to_confirm_memory", memory_id=memory_id, error=str(e))
        return False
