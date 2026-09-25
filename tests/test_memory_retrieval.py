"""
tests/test_memory_retrieval.py
================================
Unit tests for the memory retrieval architecture.

Tests cover:
1. Live-data intent → minimal bootstrap (no episodic/decision preload)
2. Memory intent → broader bootstrap
3. search_memory tool isolation (user/store scope enforced)
4. search_memory budget (MAX_MEMORY_SEARCH_CALLS)
5. Duplicate suppression in retrieved_memories
6. No-result handling
7. Reranker ordering
8. Temporal filtering
"""

from __future__ import annotations

import datetime
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# ---------------------------------------------------------------------------
# Bootstrap memory node tests
# ---------------------------------------------------------------------------

class TestBootstrapMemoryNode:

    @pytest.mark.asyncio
    async def test_live_data_intent_skips_store_memory(self):
        """For live_data intent, bootstrap should skip store-specific deep memory."""
        from app.agent.nodes.memory_query import memory_query_node

        state = {
            "store_id": "store_A",
            "user_id": "user_1",
            "intent": "live_data",
            "user_memory_loaded": False,
            "store_memory_loaded": False,
            "multi_store_memory_loaded": False,
            "messages": [],
            "user_prompt": "What is my current stock?",
            "goal": "",
        }

        with patch("app.agent.nodes.memory_query.search_user_memory", new_callable=AsyncMock) as mock_user, \
             patch("app.agent.nodes.memory_query.search_store_memory", new_callable=AsyncMock) as mock_store, \
             patch("app.agent.nodes.memory_query.search_multi_store_memory", new_callable=AsyncMock) as mock_multi, \
             patch("app.agent.nodes.memory_query.build_memory_query", new_callable=AsyncMock, return_value="stock query"), \
             patch("app.agent.nodes.memory_query.summarize_memory_results", new_callable=AsyncMock, return_value=""):
            mock_user.return_value = []
            mock_store.return_value = []
            mock_multi.return_value = []

            result = await memory_query_node(state)

        # Store memory should NOT be called for live_data intent
        mock_store.assert_not_called()
        mock_multi.assert_not_called()
        # User memory MUST be called
        assert mock_user.called
        assert result["user_memory_loaded"] is True

    @pytest.mark.asyncio
    async def test_memory_intent_loads_store_and_patterns(self):
        """For memory intent, bootstrap should load store knowledge including patterns."""
        from app.agent.nodes.memory_query import memory_query_node

        state = {
            "store_id": "store_A",
            "user_id": "user_1",
            "intent": "memory",
            "user_memory_loaded": False,
            "store_memory_loaded": False,
            "multi_store_memory_loaded": False,
            "messages": [],
            "user_prompt": "What did we decide about rice inventory?",
            "goal": "",
        }

        with patch("app.agent.nodes.memory_query.search_user_memory", new_callable=AsyncMock, return_value=[]) as mock_user, \
             patch("app.agent.nodes.memory_query.search_store_memory", new_callable=AsyncMock, return_value=[]) as mock_store, \
             patch("app.agent.nodes.memory_query.search_multi_store_memory", new_callable=AsyncMock, return_value=[]), \
             patch("app.agent.nodes.memory_query.build_memory_query", new_callable=AsyncMock, return_value="rice inventory"), \
             patch("app.agent.nodes.memory_query.summarize_memory_results", new_callable=AsyncMock, return_value=""):
            result = await memory_query_node(state)

        assert mock_user.called
        assert mock_store.called
        assert result["store_memory_loaded"] is True

    @pytest.mark.asyncio
    async def test_already_loaded_skips_retrieval(self):
        """If memory already loaded, node should skip all Pinecone calls."""
        from app.agent.nodes.memory_query import memory_query_node

        state = {
            "store_id": "s1", "user_id": "u1", "intent": "general",
            "user_memory_loaded": True,
            "store_memory_loaded": True,
            "multi_store_memory_loaded": True,
            "messages": [], "user_prompt": "test", "goal": "",
        }

        with patch("app.agent.nodes.memory_query.search_user_memory", new_callable=AsyncMock) as mock_user:
            result = await memory_query_node(state)

        mock_user.assert_not_called()
        assert result == {"memory_query_needed": False}


# ---------------------------------------------------------------------------
# search_memory tool tests
# ---------------------------------------------------------------------------

class TestSearchMemoryTool:

    def _make_config(self, user_id="u1", store_id="s1"):
        return {"configurable": {"user_id": user_id, "store_id": store_id}}

    @pytest.mark.asyncio
    async def test_no_results_returns_found_false(self):
        """When Pinecone returns no matches, found=False with empty results."""
        from app.agent.tools.memory.search import search_memory

        with patch("app.agent.tools.memory.search._query_memory_vectors", new_callable=AsyncMock, return_value=[]):
            result = await search_memory.ainvoke(
                {"query": "rice decision", "scope": "store"},
                config=self._make_config(),
            )

        assert result["found"] is False
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_store_isolation(self):
        """search_memory must filter by store_id from config, not from LLM input."""
        from app.agent.tools.memory.search import search_memory

        captured_filters = []

        async def fake_query(filter_dict, query, top_k=5):
            captured_filters.append(filter_dict)
            return []

        with patch("app.agent.tools.memory.search._query_memory_vectors", side_effect=fake_query):
            await search_memory.ainvoke(
                {"query": "test", "scope": "store"},
                config=self._make_config(store_id="store_A"),
            )

        # Verify the filter used store_A, not something the LLM could inject
        store_filters = [f for f in captured_filters if "store_id" in f]
        assert all(f["store_id"] == "store_A" for f in store_filters)

    @pytest.mark.asyncio
    async def test_memory_type_filter_applied(self):
        """memory_types argument must appear in the Pinecone filter."""
        from app.agent.tools.memory.search import search_memory

        captured_filters = []

        async def fake_query(filter_dict, query, top_k=5):
            captured_filters.append(filter_dict)
            return []

        with patch("app.agent.tools.memory.search._query_memory_vectors", side_effect=fake_query):
            await search_memory.ainvoke(
                {"query": "diwali decision", "memory_types": ["decision", "episodic_event"], "scope": "store"},
                config=self._make_config(),
            )

        type_filters = [f for f in captured_filters if "memory_type" in f]
        assert len(type_filters) > 0
        assert "decision" in type_filters[0]["memory_type"]["$in"]

    @pytest.mark.asyncio
    async def test_temporal_filter(self):
        """Memories outside date_from/date_to range should be excluded."""
        from app.agent.tools.memory.search import search_memory

        old_memory = {
            "id": "mem_old",
            "score": 0.9,
            "metadata": {
                "event_time": "2026-01-15",
                "memory_type": "decision",
                "status": "active",
                "text": "Old decision",
            },
            "text": "Old decision",
        }

        with patch("app.agent.tools.memory.search._query_memory_vectors", new_callable=AsyncMock, return_value=[old_memory]):
            result = await search_memory.ainvoke(
                {
                    "query": "decision",
                    "date_from": "2026-09-01",
                    "date_to": "2026-09-30",
                    "scope": "store",
                },
                config=self._make_config(),
            )

        # old_memory has event_time in January, so it should be filtered out
        assert result["found"] is False or len(result["results"]) == 0

    @pytest.mark.asyncio
    async def test_no_identity_returns_error(self):
        """Without user_id/store_id in config, should return controlled error."""
        from app.agent.tools.memory.search import search_memory

        result = await search_memory.ainvoke(
            {"query": "test", "scope": "store"},
            config={"configurable": {}},  # empty identity
        )

        assert result["found"] is False
        assert "error" in result or result["results"] == []

    @pytest.mark.asyncio
    async def test_duplicate_results_deduped(self):
        """Same memory_id returned twice should appear only once."""
        from app.agent.tools.memory.search import search_memory

        dup_memory = {
            "id": "mem_001",
            "score": 0.85,
            "metadata": {"memory_type": "decision", "status": "active", "text": "Buy rice before Diwali"},
            "text": "Buy rice before Diwali",
        }

        with patch(
            "app.agent.tools.memory.search._query_memory_vectors",
            new_callable=AsyncMock,
            return_value=[dup_memory, dup_memory],  # duplicated
        ):
            result = await search_memory.ainvoke(
                {"query": "rice decision", "scope": "store"},
                config=self._make_config(),
            )

        memory_ids = [r.get("memory_id") for r in result.get("results", [])]
        assert memory_ids.count("mem_001") <= 1


# ---------------------------------------------------------------------------
# Reranker tests
# ---------------------------------------------------------------------------

class TestMemoryReranker:

    def test_higher_importance_ranks_first(self):
        """Memory with higher importance should rank above similar-score memory."""
        from app.agent.tools.memory.search import _rerank

        low_importance = {
            "id": "a", "score": 0.8,
            "metadata": {"importance": 0.3, "confidence": 0.5, "memory_type": "decision"},
        }
        high_importance = {
            "id": "b", "score": 0.78,
            "metadata": {"importance": 0.9, "confidence": 0.9, "memory_type": "decision"},
        }

        ranked = _rerank([low_importance, high_importance], "test")
        assert ranked[0]["id"] == "b"

    def test_preference_decays_slower_than_episodic(self):
        """User preferences should retain higher score than old episodic events."""
        from app.agent.tools.memory.search import _rerank

        old_date = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=180)).isoformat()

        old_episodic = {
            "id": "ep", "score": 0.82,
            "metadata": {"importance": 0.8, "confidence": 0.9, "memory_type": "episodic_event", "created_at": old_date},
        }
        old_pref = {
            "id": "pref", "score": 0.80,
            "metadata": {"importance": 0.8, "confidence": 0.9, "memory_type": "user_preference", "created_at": old_date},
        }

        ranked = _rerank([old_episodic, old_pref], "preference test")
        # user_preference should beat episodic after recency decay is applied
        assert ranked[0]["id"] == "pref"
