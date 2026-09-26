"""
app/agent/memory.py
===================
Custom Pinecone Vector Database Long-Term Memory (LTM) Manager.

Replaces mem0 with a multi-level vector memory system built directly on Pinecone
and Gemini Embeddings (models/gemini-embedding-001).

Memory Levels & Architecture:
-----------------------------
1. USER PREFERENCES (`user_preference`)
   - Scoped by `user_id`.
   - Stores user-specific preferences (language, response tone, report detail, owner business goals).

2. STORE MEMORY (`store_memory`)
   - Scoped by `store_id` (fully isolated per store).
   - Stores store-specific domain facts, local restock schedules, peak sales patterns, supplier notes.

3. MULTI-STORE MEMORY (`multi_store_memory`)
   - Scoped by `user_id` and list of `store_ids`.
   - Stores cross-store chain strategy, inter-store inventory transfer patterns, enterprise insights.

Memory Reconciliation & De-duplication:
--------------------------------------
- Before adding new facts, existing matching memories are fetched from Pinecone.
- A fast LLM reconciliation call evaluates whether to ADD, UPDATE (replace old memory ID),
  DELETE (remove obsolete fact), or NO_CHANGE (prevent duplicates).
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import uuid
from typing import Any, Optional

import structlog

from app.config.settings import get_settings
from app.lib.pinecone import get_pinecone_client, ensure_index_exists
from app.lib.summarizer import summarize
from app.agent.prompts.summarizer_prompts import (
    MEMORY_EXTRACTION_INSTRUCTION,
    MEMORY_QUERY_INSTRUCTION,
    SUMMARIZER_MEMORY_INSTRUCTION,
    MULTI_LEVEL_MEMORY_EXTRACTION_INSTRUCTION,
    MEMORY_RECONCILIATION_INSTRUCTION,
)

LOGGER = structlog.get_logger("vyaparsathi.ai.memory.pinecone")

# Memory Types
TYPE_USER_PREFERENCE = "user_preference"
TYPE_STORE_MEMORY = "store_memory"
TYPE_MULTI_STORE_MEMORY = "multi_store_memory"


# ---------------------------------------------------------------------------
# Embedding & Index Helpers
# ---------------------------------------------------------------------------

def _get_embeddings():
    """Get Gemini embeddings instance with automatic key rotation and rate-limit handling."""
    from app.lib.gemini_keys import RotatingGoogleGenerativeAIEmbeddings, get_gemini_api_keys

    keys = get_gemini_api_keys()
    if not keys:
        LOGGER.warning("gemini_api_key_missing_for_embeddings")
        return None

    return RotatingGoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def _get_pinecone_index():
    """Get Pinecone index object for vyapar-sathi."""
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


def get_memory_client():
    """Alias for backward compatibility."""
    return get_pinecone_client()


def is_enabled() -> bool:
    """Return True if Pinecone memory system is configured and available."""
    settings = get_settings()
    api_key = settings.pinecone_api_key or os.getenv("PINCONE_API_KEY")
    gemini_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
    return bool(api_key and gemini_key)


def get_memory_status() -> dict:
    """Return diagnostic info about Pinecone long-term memory system."""
    settings = get_settings()
    pc = get_pinecone_client()
    index_stats = None

    if pc:
        try:
            index_name = settings.pinecone_index_name or "vyapar-sathi"
            index = pc.Index(index_name)
            index_stats = index.describe_index_stats()
        except Exception:
            index_stats = None

    return {
        "enabled": is_enabled(),
        "provider": "pinecone",
        "has_pinecone_api_key": bool(settings.pinecone_api_key or os.getenv("PINCONE_API_KEY")),
        "has_gemini_api_key": bool(settings.gemini_api_key or os.getenv("GEMINI_API_KEY")),
        "index_name": settings.pinecone_index_name or "vyapar-sathi",
        "index_stats": index_stats,
    }


# ---------------------------------------------------------------------------
# Memory Persistence Core (Upsert / Delete)
# ---------------------------------------------------------------------------

async def _upsert_memory_vector(
    memory_type: str,
    text: str,
    metadata_fields: dict[str, Any],
    existing_id: str | None = None,
) -> bool:
    """
    Generate embedding for text and upsert to Pinecone index with metadata.
    """
    if not text or not text.strip():
        return False

    embeddings_service = _get_embeddings()
    index = _get_pinecone_index()

    if not embeddings_service or not index:
        LOGGER.warning("pinecone_upsert_skipped", reason="embeddings or pinecone index unavailable")
        return False

    try:
        vector = await asyncio.to_thread(embeddings_service.embed_query, text)
        memory_id = existing_id or f"mem_{memory_type}_{uuid.uuid4().hex[:12]}"
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        metadata = {
            "memory_type": memory_type,
            "text": text,
            "updated_at": now_iso,
            **metadata_fields,
        }
        if "created_at" not in metadata:
            metadata["created_at"] = now_iso

        await asyncio.to_thread(
            index.upsert,
            vectors=[{"id": memory_id, "values": vector, "metadata": metadata}],
        )

        LOGGER.info(
            "pinecone_memory_upserted",
            memory_type=memory_type,
            memory_id=memory_id,
            text_preview=text[:80],
        )
        return True
    except Exception as exc:
        LOGGER.error("pinecone_upsert_failed", memory_type=memory_type, error=str(exc), exc_info=True)
        return False


async def _delete_memory_vector(memory_id: str) -> bool:
    """Delete a memory vector entry by ID from Pinecone index."""
    index = _get_pinecone_index()
    if not index or not memory_id:
        return False
    try:
        await asyncio.to_thread(index.delete, ids=[memory_id])
        LOGGER.info("pinecone_memory_deleted", memory_id=memory_id)
        return True
    except Exception as exc:
        LOGGER.error("pinecone_delete_failed", memory_id=memory_id, error=str(exc))
        return False


# ---------------------------------------------------------------------------
# Memory Search Core (Query)
# ---------------------------------------------------------------------------

async def _query_memory_vectors(
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
            mem_text = meta.get("text") or meta.get("memory") or ""

            if mem_text:
                results.append({
                    "id": getattr(match, "id", "") if not isinstance(match, dict) else match.get("id", ""),
                    "memory": mem_text,
                    "text": mem_text,
                    "score": float(score),
                    "created_at": meta.get("created_at"),
                    "memory_type": meta.get("memory_type"),
                    "metadata": meta,
                })

        LOGGER.info("pinecone_memory_searched", filter=filter_dict, query=query_text[:60], matches=len(results))
        return results
    except Exception as exc:
        LOGGER.error("pinecone_search_failed", filter=filter_dict, error=str(exc), exc_info=True)
        return []


# ---------------------------------------------------------------------------
# LLM Memory Reconciliation & Deduplication Engine
# ---------------------------------------------------------------------------

async def _reconcile_and_update_memory(
    memory_type: str,
    new_candidate_text: str,
    filter_dict: dict[str, Any],
    metadata_fields: dict[str, Any],
) -> bool:
    """
    1. Fetch existing matching memories from Pinecone under filter_dict.
    2. Pass existing memories + candidate fact to small LLM reconciler.
    3. Execute returned vector actions (ADD, UPDATE, DELETE, NO_CHANGE).
    """
    if not new_candidate_text or not new_candidate_text.strip():
        return False

    existing_memories = await _query_memory_vectors(filter_dict, new_candidate_text, top_k=5)

    if not existing_memories:
        return await _upsert_memory_vector(memory_type, new_candidate_text, metadata_fields)

    reconciliation_prompt = (
        f"{MEMORY_RECONCILIATION_INSTRUCTION}\n\n"
        "EXISTING MEMORIES:\n"
        + "\n".join([f"- ID: {m['id']} | Content: \"{m['text']}\"" for m in existing_memories])
        + "\n\nNEW CANDIDATE FACT:\n"
        + f'"{new_candidate_text}"\n'
    )

    try:
        from app.lib.llm import get_small_llm
        llm = get_small_llm()
        if llm:
            res = await llm.ainvoke(reconciliation_prompt)
            raw_json_str = res.content if hasattr(res, "content") else str(res)
        else:
            raw_json_str = await summarize(
                text=reconciliation_prompt,
                instruction="You are a JSON memory reconciler. Output valid JSON only.",
                max_tokens=256,
            )

        actions_data = None
        if raw_json_str and "{" in raw_json_str:
            json_start = raw_json_str.find("{")
            json_end = raw_json_str.rfind("}") + 1
            json_snippet = raw_json_str[json_start:json_end]
            actions_data = json.loads(json_snippet)

        actions = actions_data.get("actions", []) if isinstance(actions_data, dict) else []

        if not actions:
            highest_score = max([m.get("score", 0.0) for m in existing_memories], default=0.0)
            if highest_score > 0.85:
                LOGGER.info("reconciliation_fallback_no_change", highest_score=highest_score)
                return True
            return await _upsert_memory_vector(memory_type, new_candidate_text, metadata_fields)

        for act in actions:
            action_type = str(act.get("action", "")).upper()
            target_id = act.get("id")
            text_val = act.get("text") or new_candidate_text

            if action_type == "ADD":
                await _upsert_memory_vector(memory_type, text_val, metadata_fields)
            elif action_type == "UPDATE" and target_id:
                await _upsert_memory_vector(memory_type, text_val, metadata_fields, existing_id=target_id)
                LOGGER.info("pinecone_memory_reconciled_update", memory_id=target_id, text=text_val[:80])
            elif action_type == "DELETE" and target_id:
                await _delete_memory_vector(target_id)
                LOGGER.info("pinecone_memory_reconciled_delete", memory_id=target_id)
            elif action_type == "NO_CHANGE":
                LOGGER.info("pinecone_memory_reconciled_no_change", memory_id=target_id)

        return True
    except Exception as exc:
        LOGGER.error("reconciliation_failed_fallback_upsert", error=str(exc))
        return await _upsert_memory_vector(memory_type, new_candidate_text, metadata_fields)


# ---------------------------------------------------------------------------
# Multi-Level Memory Extraction & Pipeline Core
# ---------------------------------------------------------------------------

async def extract_multi_level_memory(messages: list[dict]) -> dict[str, list[str]]:
    """
    Analyze conversation messages using LLM to extract durable, concise facts
    categorized strictly by memory level (user_preferences, store_knowledge, multi_store_knowledge).

    Filters out ephemeral Q&A logs, raw turns, greetings, out-of-scope queries, and temporary numbers.
    Returns dict: {"user_preferences": [...], "store_knowledge": [...], "multi_store_knowledge": [...]}
    """
    if not messages:
        return {"user_preferences": [], "store_knowledge": [], "multi_store_knowledge": []}

    formatted_turns = []
    for msg in messages:
        if isinstance(msg, dict):
            role = str(msg.get("role", "user")).upper()
            content = msg.get("content", "")
            if content and isinstance(content, str):
                formatted_turns.append(f"{role}: {content}")

    chat_text = "\n\n".join(formatted_turns)
    if not chat_text.strip():
        return {"user_preferences": [], "store_knowledge": [], "multi_store_knowledge": []}

    prompt = (
        f"{MULTI_LEVEL_MEMORY_EXTRACTION_INSTRUCTION}\n\n"
        f"CONVERSATION EXCHANGE TO ANALYZE:\n{chat_text}"
    )

    try:
        from app.lib.llm import get_small_llm
        llm = get_small_llm()
        if llm:
            res = await llm.ainvoke(prompt)
            raw_json_str = res.content if hasattr(res, "content") else str(res)
        else:
            raw_json_str = await summarize(
                text=chat_text,
                instruction=MULTI_LEVEL_MEMORY_EXTRACTION_INSTRUCTION,
                max_tokens=512,
            )

        extracted_data = {}
        if raw_json_str and "{" in raw_json_str:
            json_start = raw_json_str.find("{")
            json_end = raw_json_str.rfind("}") + 1
            json_snippet = raw_json_str[json_start:json_end]
            extracted_data = json.loads(json_snippet)

        user_prefs = [str(x).strip() for x in extracted_data.get("user_preferences", []) if str(x).strip()]
        store_know = [str(x).strip() for x in extracted_data.get("store_knowledge", []) if str(x).strip()]
        multi_know = [str(x).strip() for x in extracted_data.get("multi_store_knowledge", []) if str(x).strip()]

        LOGGER.info(
            "multi_level_memory_extracted",
            user_prefs_count=len(user_prefs),
            store_know_count=len(store_know),
            multi_know_count=len(multi_know),
        )

        return {
            "user_preferences": user_prefs,
            "store_knowledge": store_know,
            "multi_store_knowledge": multi_know,
        }
    except Exception as exc:
        LOGGER.error("multi_level_memory_extraction_failed", error=str(exc), exc_info=True)
        return {"user_preferences": [], "store_knowledge": [], "multi_store_knowledge": []}


async def process_and_persist_memory(
    *,
    user_id: str,
    store_id: str,
    messages: list[dict],
    store_ids: list[str] | None = None,
) -> dict[str, bool]:
    """
    Main Multi-Level Memory Pipeline.

    1. Extract level-specific durable facts using LLM (user_preferences, store_knowledge, multi_store_knowledge).
    2. Skip memory levels that have no extracted facts (prevents noise and redundant DB queries).
    3. Reconcile extracted facts against existing vector memories via LLM.
    4. Upsert/Update/Delete vectors in Pinecone index.
    """
    if not messages or (not user_id and not store_id):
        return {"user_ok": True, "store_ok": True, "multi_store_ok": True}

    extracted = await extract_multi_level_memory(messages)

    user_prefs = extracted.get("user_preferences", [])
    store_know = extracted.get("store_knowledge", [])
    multi_know = extracted.get("multi_store_knowledge", [])

    user_ok = True
    store_ok = True
    multi_store_ok = True

    # Process Level 1: User Preferences
    if user_prefs and user_id:
        user_items = [{"content": fact} for fact in user_prefs]
        user_ok = await add_user_memory(user_id, messages, curated_messages=user_items)

    # Process Level 2: Store Knowledge
    if store_know and store_id:
        store_items = [{"content": fact} for fact in store_know]
        store_ok = await add_store_memory(store_id, messages, curated_messages=store_items)

    # Process Level 3: Multi-Store Knowledge
    if multi_know and user_id:
        s_ids = store_ids or [store_id]
        multi_items = [{"content": fact} for fact in multi_know]
        multi_store_ok = await add_multi_store_memory(user_id, s_ids, messages, curated_messages=multi_items)

    LOGGER.info(
        "pinecone_memory_pipeline_complete",
        user_id=user_id,
        store_id=store_id,
        user_prefs_processed=len(user_prefs),
        store_know_processed=len(store_know),
        multi_know_processed=len(multi_know),
        user_ok=user_ok,
        store_ok=store_ok,
        multi_store_ok=multi_store_ok,
    )

    return {
        "user_ok": user_ok,
        "store_ok": store_ok,
        "multi_store_ok": multi_store_ok,
    }


# ---------------------------------------------------------------------------
# LEVEL 1: User Preference Memory (user_id)
# ---------------------------------------------------------------------------

async def add_user_memory(
    user_id: str,
    messages: list[dict],
    *,
    curated_messages: list[dict] | None = None,
) -> bool:
    """
    Extract, reconcile, and store durable user preferences into Pinecone.
    """
    if not messages and not curated_messages:
        return False

    curated = curated_messages if curated_messages is not None else await curate_memory_messages(messages)
    if not curated:
        return True

    filter_dict = {
        "memory_type": TYPE_USER_PREFERENCE,
        "user_id": str(user_id),
    }

    success_all = True
    for item in curated:
        content = item.get("content", "") if isinstance(item, dict) else str(item)
        if content:
            ok = await _reconcile_and_update_memory(
                memory_type=TYPE_USER_PREFERENCE,
                new_candidate_text=content,
                filter_dict=filter_dict,
                metadata_fields={"user_id": str(user_id)},
            )
            if not ok:
                success_all = False

    return success_all


async def search_user_memory(user_id: str, query: str, top_k: int = 5) -> list[dict]:
    """
    Search Pinecone for user preferences (language, tone, goals).
    """
    if not user_id:
        return []

    filter_dict = {
        "memory_type": TYPE_USER_PREFERENCE,
        "user_id": str(user_id),
    }
    return await _query_memory_vectors(filter_dict, query, top_k=top_k)


# ---------------------------------------------------------------------------
# LEVEL 2: Store Memory (store_id)
# ---------------------------------------------------------------------------

async def add_store_memory(
    store_id: str,
    messages: list[dict],
    *,
    curated_messages: list[dict] | None = None,
) -> bool:
    """
    Extract, reconcile, and store store-specific domain facts into Pinecone (isolated per store_id).
    """
    if (not messages and not curated_messages) or not store_id:
        return False

    curated = curated_messages if curated_messages is not None else await curate_memory_messages(messages)
    if not curated:
        return True

    filter_dict = {
        "memory_type": TYPE_STORE_MEMORY,
        "store_id": str(store_id),
    }

    success_all = True
    for item in curated:
        content = item.get("content", "") if isinstance(item, dict) else str(item)
        if content:
            ok = await _reconcile_and_update_memory(
                memory_type=TYPE_STORE_MEMORY,
                new_candidate_text=content,
                filter_dict=filter_dict,
                metadata_fields={"store_id": str(store_id)},
            )
            if not ok:
                success_all = False

    return success_all


async def search_store_memory(store_id: str, query: str, top_k: int = 5) -> list[dict]:
    """
    Search Pinecone for store-specific domain knowledge and past decisions.
    """
    if not store_id:
        return []

    filter_dict = {
        "memory_type": TYPE_STORE_MEMORY,
        "store_id": str(store_id),
    }
    return await _query_memory_vectors(filter_dict, query, top_k=top_k)


# ---------------------------------------------------------------------------
# LEVEL 3: Multi-Store Memory (user_id + list of store_ids)
# ---------------------------------------------------------------------------

async def add_multi_store_memory(
    user_id: str,
    store_ids: list[str],
    messages: list[dict],
    *,
    curated_messages: list[dict] | None = None,
) -> bool:
    """
    Extract, reconcile, and store multi-store enterprise knowledge into Pinecone.
    """
    if (not messages and not curated_messages) or not user_id:
        return False

    curated = curated_messages if curated_messages is not None else await curate_memory_messages(messages)
    if not curated:
        return True

    store_ids_str = ",".join(str(s) for s in store_ids) if store_ids else ""
    filter_dict = {
        "memory_type": TYPE_MULTI_STORE_MEMORY,
        "user_id": str(user_id),
    }

    success_all = True
    for item in curated:
        content = item.get("content", "") if isinstance(item, dict) else str(item)
        if content:
            ok = await _reconcile_and_update_memory(
                memory_type=TYPE_MULTI_STORE_MEMORY,
                new_candidate_text=content,
                filter_dict=filter_dict,
                metadata_fields={
                    "user_id": str(user_id),
                    "store_ids": store_ids_str,
                },
            )
            if not ok:
                success_all = False

    return success_all


async def search_multi_store_memory(user_id: str, query: str, top_k: int = 5) -> list[dict]:
    """
    Search Pinecone for multi-store / enterprise level strategies and insights.
    """
    if not user_id:
        return []

    filter_dict = {
        "memory_type": TYPE_MULTI_STORE_MEMORY,
        "user_id": str(user_id),
    }
    return await _query_memory_vectors(filter_dict, query, top_k=top_k)


# ---------------------------------------------------------------------------
# Combined retrieval for the think node
# ---------------------------------------------------------------------------

async def load_memory_context(
    *,
    user_id: str,
    store_id: str,
    user_prompt: str,
    current_goal: str = "",
    store_ids: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """
    Load user-preference, single-store, and multi-store memory from Pinecone.

    Always loads basic user preferences (language, format, detail level) unconditionally,
    plus query-relevant store & multi-store facts.

    Returns:
        Tuple of (user_preferences, store_knowledge, multi_store_knowledge) dicts.
    """
    memory_query = await build_memory_query(user_prompt, current_goal=current_goal)

    # 1. Fetch user preferences (query-specific + general preference fallback)
    user_results = await search_user_memory(user_id, memory_query)
    if not user_results or len(user_results) < 2:
        general_user = await search_user_memory(user_id, "user preference language tone format explanation detail business goals")
        seen_user_ids = {u["id"] for u in user_results if "id" in u}
        for g in general_user:
            if g.get("id") not in seen_user_ids:
                user_results.append(g)

    # 2. Fetch store and multi-store knowledge
    store_results = await search_store_memory(store_id, memory_query)
    if not store_results or len(store_results) < 2:
        general_store = await search_store_memory(store_id, "store facts rules supplier schedule restock inventory")
        seen_store_ids = {s["id"] for s in store_results if "id" in s}
        for s in general_store:
            if s.get("id") not in seen_store_ids:
                store_results.append(s)

    multi_store_results = await search_multi_store_memory(user_id, memory_query)

    # Keep user preferences intact without harsh threshold filtering
    user_results = _cap_results(user_results)
    store_results = _cap_results(_filter_relevant_results(store_results))
    multi_store_results = _cap_results(_filter_relevant_results(multi_store_results))

    user_prefs = {
        "raw": user_results,
        "summary": await summarize_memory_results(user_results, memory_query),
    }
    store_knowledge = {
        "raw": store_results,
        "summary": await summarize_memory_results(store_results, memory_query),
    }
    multi_store_knowledge = {
        "raw": multi_store_results,
        "summary": await summarize_memory_results(multi_store_results, memory_query),
    }

    return user_prefs, store_knowledge, multi_store_knowledge


def should_retrieve_memory(user_prompt: str) -> bool:
    """Return whether the request needs durable memory rather than live tools."""
    prompt = (user_prompt or "").lower()
    memory_signals = (
        "remember", "last time", "previous", "earlier", "we decided",
        "our decision", "usually", "prefer", "preference", "my language",
        "my style", "what did we discuss", "multi store", "across stores",
    )
    return any(signal in prompt for signal in memory_signals)


async def build_memory_query(user_prompt: str, *, current_goal: str = "") -> str:
    """Distill request into a focused semantic search query."""
    prompt = (user_prompt or "").strip()
    if not prompt:
        return "retail store inventory sales forecasting restocking preferences"

    query_input = f"Current request: {prompt}"
    if current_goal:
        query_input += f"\nCurrent goal: {current_goal}"

    query = await summarize(
        query_input,
        instruction=MEMORY_QUERY_INSTRUCTION,
        max_tokens=48,
    )
    query = " ".join((query or "").split())
    if not query or query.upper() == "NONE":
        return prompt[:300]
    return query[:300]


async def curate_memory_messages(messages: list[dict]) -> list[dict]:
    """Convert exchange into durable memory records across all levels."""
    if not messages:
        return []

    extracted = await extract_multi_level_memory(messages)
    all_facts = (
        extracted.get("user_preferences", [])
        + extracted.get("store_knowledge", [])
        + extracted.get("multi_store_knowledge", [])
    )

    if not all_facts:
        return []

    return [{"role": "user", "content": fact} for fact in all_facts]


def _filter_relevant_results(results: list[dict]) -> list[dict]:
    """Filter low similarity score results."""
    if not results:
        return []

    filtered = []
    for result in results:
        score = result.get("score") if isinstance(result, dict) else None
        if isinstance(score, (int, float)) and score < 0.40:
            continue
        filtered.append(result)
    return filtered


_MAX_MEMORY_ENTRIES = 5
_MAX_MEMORY_CHARS = 2_000
_MAX_MEMORY_ENTRY_CHARS = 400


def _sort_user_source_first(results: list[dict]) -> list[dict]:
    """
    Prioritize memories originating from the USER over memories from the ASSISTANT.
    User-stated facts and preferences take strict precedence.
    """
    if not results:
        return []

    def _sort_key(item: dict) -> tuple[int, float]:
        meta = item.get("metadata", {}) if isinstance(item, dict) else {}
        source_role = meta.get("source_role", "") if isinstance(meta, dict) else ""
        mem_type = item.get("memory_type", "") if isinstance(item, dict) else ""

        # Priority 0: Explicit user source or user_preference
        if source_role == "user" or mem_type == "user_preference":
            user_priority = 0
        elif source_role == "assistant":
            user_priority = 2
        else:
            user_priority = 1

        score = item.get("score", 0.0) if isinstance(item, dict) else 0.0
        return (user_priority, -score)

    return sorted(results, key=_sort_key)


def _cap_results(results: list[dict]) -> list[dict]:
    """Trim memory result list to a bounded size, prioritizing user sources."""
    if not results:
        return []

    sorted_results = _sort_user_source_first(results)
    capped = list(sorted_results[:_MAX_MEMORY_ENTRIES])
    for i, r in enumerate(capped):
        if isinstance(r, dict):
            text = r.get("memory") or r.get("text")
            if isinstance(text, str) and len(text) > _MAX_MEMORY_ENTRY_CHARS:
                r = dict(r)
                r["memory"] = text[:_MAX_MEMORY_ENTRY_CHARS] + "...[truncated]"
                capped[i] = r
    return capped


async def summarize_memory_results(results: list[dict], query: str) -> str:
    """Produce compact summary of vector memory search results."""
    results = _cap_results(results)
    if not results:
        return ""

    flat = _flatten_results(results)
    if len(flat) <= _MAX_MEMORY_CHARS:
        return flat

    summary = await summarize(
        flat,
        instruction=SUMMARIZER_MEMORY_INSTRUCTION,
        max_tokens=200,
    )
    return summary or flat


def _flatten_results(results: list[dict]) -> str:
    """Flatten results into plain text."""
    parts = []
    for r in results:
        text = r.get("memory") or r.get("text") or str(r)
        parts.append(str(text) if not isinstance(text, str) else text)
    return "\n".join(parts)


def get_memory_status() -> dict[str, Any]:
    """Return Pinecone Multi-Level LTM configuration and health status."""
    from app.config.settings import get_settings
    settings = get_settings()
    index_ok = _get_pinecone_index() is not None
    return {
        "provider": "pinecone",
        "has_pinecone_api_key": bool(settings.pinecone_api_key),
        "has_gemini_api_key": bool(settings.gemini_api_key),
        "index_name": getattr(settings, "pinecone_index", "vyapar-sathi"),
        "pinecone_available": index_ok,
        "embedding_model": "models/gemini-embedding-001",
        "dimension": 3072,
        "status": "healthy" if index_ok else "degraded",
    }