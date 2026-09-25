"""
tests/test_pinecone_memory.py
==============================
Unit and integration tests for the Custom Pinecone Multi-Level Memory System.
"""
from __future__ import annotations

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.memory import (
    TYPE_USER_PREFERENCE,
    TYPE_STORE_MEMORY,
    TYPE_MULTI_STORE_MEMORY,
    add_user_memory,
    search_user_memory,
    add_store_memory,
    search_store_memory,
    add_multi_store_memory,
    search_multi_store_memory,
    load_memory_context,
    get_memory_status,
)


@pytest.mark.asyncio
async def test_memory_status_returns_pinecone_info():
    status = get_memory_status()
    assert status["provider"] == "pinecone"
    assert "has_pinecone_api_key" in status
    assert "has_gemini_api_key" in status
    assert status["index_name"] == "vyapar-sathi"


@pytest.mark.asyncio
@patch("app.agent.memory.retriever._reconcile_and_update_memory")
@patch("app.agent.memory.retriever.curate_memory_messages")
async def test_add_user_memory(mock_curate, mock_reconcile):
    mock_curate.return_value = [{"role": "user", "content": "User prefers Hindi responses."}]
    mock_reconcile.return_value = True

    messages = [{"role": "user", "content": "Hi, answer in Hindi."}]
    success = await add_user_memory("user-123", messages)

    assert success is True
    mock_reconcile.assert_called_once_with(
        memory_type=TYPE_USER_PREFERENCE,
        new_candidate_text="User prefers Hindi responses.",
        filter_dict={"memory_type": TYPE_USER_PREFERENCE, "user_id": "user-123"},
        metadata_fields={"user_id": "user-123"},
    )


@pytest.mark.asyncio
@patch("app.agent.memory.retriever._reconcile_and_update_memory")
@patch("app.agent.memory.retriever.curate_memory_messages")
async def test_add_store_memory(mock_curate, mock_reconcile):
    mock_curate.return_value = [{"role": "user", "content": "Rice restock lead time is 2 days."}]
    mock_reconcile.return_value = True

    messages = [{"role": "user", "content": "Rice restock lead time is 2 days."}]
    success = await add_store_memory("store-456", messages)

    assert success is True
    mock_reconcile.assert_called_once_with(
        memory_type=TYPE_STORE_MEMORY,
        new_candidate_text="Rice restock lead time is 2 days.",
        filter_dict={"memory_type": TYPE_STORE_MEMORY, "store_id": "store-456"},
        metadata_fields={"store_id": "store-456"},
    )


@pytest.mark.asyncio
@patch("app.agent.memory.retriever._reconcile_and_update_memory")
@patch("app.agent.memory.retriever.curate_memory_messages")
async def test_add_multi_store_memory(mock_curate, mock_reconcile):
    mock_curate.return_value = [{"role": "user", "content": "Cross-store inventory transfer strategy enabled."}]
    mock_reconcile.return_value = True

    messages = [{"role": "user", "content": "Transfer stock between store A and store B."}]
    success = await add_multi_store_memory("user-123", ["store-A", "store-B"], messages)

    assert success is True
    mock_reconcile.assert_called_once_with(
        memory_type=TYPE_MULTI_STORE_MEMORY,
        new_candidate_text="Cross-store inventory transfer strategy enabled.",
        filter_dict={"memory_type": TYPE_MULTI_STORE_MEMORY, "user_id": "user-123"},
        metadata_fields={"user_id": "user-123", "store_ids": "store-A,store-B"},
    )


@pytest.mark.asyncio
@patch("app.agent.memory.retriever._query_memory_vectors")
async def test_search_memories(mock_query):
    mock_query.return_value = [
        {"id": "mem_1", "memory": "Prefers Hindi", "score": 0.88}
    ]

    user_res = await search_user_memory("user-123", "language preference")
    assert len(user_res) == 1
    assert user_res[0]["memory"] == "Prefers Hindi"

    mock_query.assert_called_with(
        {"memory_type": TYPE_USER_PREFERENCE, "user_id": "user-123"},
        "language preference",
        top_k=5,
    )


@pytest.mark.asyncio
@patch("app.agent.memory.retriever.search_user_memory")
@patch("app.agent.memory.retriever.search_store_memory")
@patch("app.agent.memory.retriever.search_multi_store_memory")
@patch("app.agent.memory.retriever.build_memory_query")
async def test_load_memory_context(mock_build_q, mock_multi, mock_store, mock_user):
    mock_build_q.return_value = "inventory restock strategy"
    mock_user.return_value = [{"id": "m1", "memory": "User prefers brief format", "score": 0.9}]
    mock_store.return_value = [{"id": "m2", "memory": "Store orders on Mondays", "score": 0.85}]
    mock_multi.return_value = [{"id": "m3", "memory": "Chain discount rate 5%", "score": 0.82}]

    user_prefs, store_knowledge, multi_store_knowledge = await load_memory_context(
        user_id="user-1",
        store_id="store-1",
        user_prompt="When should I order?",
    )

    assert "raw" in user_prefs
    assert "summary" in user_prefs
    assert user_prefs["raw"][0]["memory"] == "User prefers brief format"
    assert store_knowledge["raw"][0]["memory"] == "Store orders on Mondays"
    assert multi_store_knowledge["raw"][0]["memory"] == "Chain discount rate 5%"


@pytest.mark.asyncio
@patch("app.agent.nodes.memory_write.process_and_persist_memory")
async def test_memory_write_node(mock_process):
    from app.agent.nodes.memory_write import memory_write_node

    mock_process.return_value = {"user_ok": True, "store_ok": True, "multi_store_ok": True}

    state = {
        "should_persist_memory": True,
        "user_id": "user-101",
        "store_id": "store-202",
        "store_ids": ["store-202", "store-203"],
        "user_prompt": "What are my restock options?",
        "final_answer": "Here are your urgent restock items...",
    }

    res = await memory_write_node(state)
    assert res == {}

    mock_process.assert_called_once_with(
        user_id="user-101",
        store_id="store-202",
        messages=[
            {"role": "user", "content": "What are my restock options?"},
            {"role": "assistant", "content": "Here are your urgent restock items..."},
        ],
        store_ids=["store-202", "store-203"],
    )


@pytest.mark.asyncio
@patch("app.agent.memory.retriever._delete_memory_vector")
@patch("app.agent.memory.retriever._upsert_memory_vector")
@patch("app.lib.llm.get_llm")
@patch("app.agent.memory.retriever._query_memory_vectors")
async def test_reconcile_and_update_memory(mock_query, mock_get_llm, mock_upsert, mock_delete):
    from app.agent.memory import _reconcile_and_update_memory, TYPE_USER_PREFERENCE

    mock_llm = AsyncMock()
    mock_res = MagicMock()
    mock_res.content = '{"actions": [{"action": "UPDATE", "id": "mem_old_1", "text": "User prefers English"}]}'
    mock_llm.ainvoke.return_value = mock_res
    mock_get_llm.return_value = mock_llm

    mock_query.return_value = [{"id": "mem_old_1", "text": "User prefers Hindi"}]
    mock_upsert.return_value = True
    mock_delete.return_value = True

    ok = await _reconcile_and_update_memory(
        memory_type=TYPE_USER_PREFERENCE,
        new_candidate_text="User now prefers English responses.",
        filter_dict={"memory_type": TYPE_USER_PREFERENCE, "user_id": "u1"},
        metadata_fields={"user_id": "u1"},
    )

    assert ok is True
    mock_upsert.assert_called_once_with(
        TYPE_USER_PREFERENCE,
        "User prefers English",
        {"user_id": "u1"},
        existing_id="mem_old_1",
    )


@pytest.mark.asyncio
@patch("app.lib.llm.get_llm")
async def test_extract_multi_level_memory(mock_get_llm):
    from app.agent.memory import extract_multi_level_memory

    mock_llm = AsyncMock()
    mock_res = MagicMock()
    mock_res.content = (
        '{"user_preferences": ["Prefers Hindi"], "store_knowledge": ["Amul arrives Tuesdays"], "multi_store_knowledge": []}'
    )
    mock_llm.ainvoke.return_value = mock_res
    mock_get_llm.return_value = mock_llm

    messages = [{"role": "user", "content": "Please answer in Hindi. Amul comes on Tuesdays."}]
    res = await extract_multi_level_memory(messages)

    assert res["user_preferences"] == ["Prefers Hindi"]
    assert res["store_knowledge"] == ["Amul arrives Tuesdays"]
    assert res["multi_store_knowledge"] == []


@pytest.mark.asyncio
@patch("app.agent.memory.service.write_new_memory")
@patch("app.agent.memory.service.reconcile_memory")
@patch("app.agent.memory.service.extract_memories")
async def test_process_and_persist_memory(mock_extract, mock_reconcile, mock_write):
    from app.agent.memory import process_and_persist_memory
    from app.agent.memory.models import ExtractedMemory
    from app.agent.memory.conflict import ResolutionAction

    mock_extract.return_value = [
        ExtractedMemory(
            memory_type="user_preference",
            content="User prefers English",
            subject="User",
            source_role="user",
            evidence="User requested English",
            confidence=0.9,
            importance=0.8,
        )
    ]
    mock_reconcile.return_value = ResolutionAction(action="ADD", reason="New fact")
    mock_write.return_value = True

    messages = [{"role": "user", "content": "Sample"}]
    res = await process_and_persist_memory(user_id="u1", store_id="s1", messages=messages)

    assert res["user_ok"] is True
    mock_extract.assert_called_once()
    mock_write.assert_called_once()

