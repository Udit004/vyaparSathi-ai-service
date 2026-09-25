from __future__ import annotations

import asyncio
import datetime
import os
import uuid
from typing import Any

import structlog

from app.config.settings import get_settings
from app.lib.pinecone import get_pinecone_client

LOGGER = structlog.get_logger("vyaparsathi.ai.memory.pinecone")

def _get_embeddings():
    """Get Gemini embeddings instance."""
    from app.lib.gemini_keys import RotatingGoogleGenerativeAIEmbeddings, get_gemini_api_keys
    keys = get_gemini_api_keys()
    if not keys:
        return None
    return RotatingGoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def _get_pinecone_index():
    """Get Pinecone index object."""
    pc = get_pinecone_client()
    if not pc:
        return None

    settings = get_settings()
    index_name = settings.pinecone_index_name or "vyapar-sathi"

    try:
        return pc.Index(index_name)
    except Exception as exc:
        LOGGER.error("pinecone_index_access_failed", error=str(exc))
        return None


async def upsert_vector(
    text: str,
    metadata: dict[str, Any],
    existing_id: str | None = None,
) -> str | None:
    """
    Generate embedding for text and upsert to Pinecone index with metadata.
    Returns the memory ID if successful.
    """
    if not text or not text.strip():
        return None

    embeddings_service = _get_embeddings()
    index = _get_pinecone_index()

    if not embeddings_service or not index:
        return None

    try:
        vector = await asyncio.to_thread(embeddings_service.embed_query, text)
        memory_id = existing_id or f"mem_{uuid.uuid4().hex[:16]}"
        
        # Ensure we have a valid ISO format string for any datetime objects
        safe_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, datetime.datetime):
                safe_metadata[k] = v.isoformat()
            elif v is not None:
                safe_metadata[k] = v
                
        safe_metadata["text"] = text

        await asyncio.to_thread(
            index.upsert,
            vectors=[{"id": memory_id, "values": vector, "metadata": safe_metadata}],
        )
        return memory_id
    except Exception as exc:
        LOGGER.error("pinecone_upsert_failed", error=str(exc))
        return None


async def query_vectors(
    filter_dict: dict[str, Any],
    query_text: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Embed query text and search Pinecone index using metadata filter.
    """
    embeddings_service = _get_embeddings()
    index = _get_pinecone_index()

    if not embeddings_service or not index:
        return []

    try:
        query_vector = await asyncio.to_thread(embeddings_service.embed_query, query_text)

        response = await asyncio.to_thread(
            index.query,
            vector=query_vector,
            filter=filter_dict,
            top_k=top_k,
            include_metadata=True,
        )

        matches = getattr(response, "matches", []) or []
        results = []
        for match in matches:
            meta = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {}) or {}
            score = match.get("score", 0.0) if isinstance(match, dict) else getattr(match, "score", 0.0)
            mem_text = meta.get("text") or ""
            
            # Construct a record that mimics ExtractedMemory + search score
            result = {
                "id": getattr(match, "id", "") if not isinstance(match, dict) else match.get("id", ""),
                "score": float(score),
                "metadata": meta,
                "text": mem_text,
            }
            results.append(result)

        return results
    except Exception as exc:
        LOGGER.error("pinecone_search_failed", error=str(exc))
        return []
