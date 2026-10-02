import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_voice_memory_bootstrap_falls_back_to_empty_lists_on_pinecone_error():
    from app.routes.voice_routes import _bootstrap_voice_memory_context

    with patch("app.routes.voice_routes._bootstrap_user_memory", new_callable=AsyncMock, side_effect=Exception("pinecone ssl failure")), \
         patch("app.routes.voice_routes._bootstrap_store_memory", new_callable=AsyncMock, side_effect=Exception("pinecone ssl failure")):
        user_memories, store_memories = await _bootstrap_voice_memory_context("user-1", "store-1")

    assert user_memories == []
    assert store_memories == []
