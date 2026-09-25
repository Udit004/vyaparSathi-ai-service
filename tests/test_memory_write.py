import pytest
from app.agent.memory.extractor import extract_memories

@pytest.mark.asyncio
async def test_trivial_conversation():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"}
    ]
    memories = await extract_memories("user1", "store1", messages)
    assert len(memories) == 0

@pytest.mark.asyncio
async def test_user_preference():
    messages = [
        {"role": "user", "content": "Please always answer me in Hindi."}
    ]
    memories = await extract_memories("user1", "store1", messages)
    assert len(memories) > 0
    assert any(m.memory_type == "user_preference" and m.source_role == "user" for m in memories)

@pytest.mark.asyncio
async def test_assistant_statement():
    messages = [
        {"role": "assistant", "content": "I think you prefer Hindi."}
    ]
    memories = await extract_memories("user1", "store1", messages)
    # Should not be user_preference since it's from assistant
    assert not any(m.memory_type == "user_preference" and m.source_role == "user" for m in memories)

@pytest.mark.asyncio
async def test_business_decision():
    messages = [
        {"role": "user", "content": "Before Diwali, I have decided to increase rice stock."}
    ]
    memories = await extract_memories("user1", "store1", messages)
    assert len(memories) > 0
    assert any(m.memory_type == "decision" and m.source_role == "user" for m in memories)

@pytest.mark.asyncio
async def test_assistant_recommendation():
    messages = [
        {"role": "assistant", "content": "Based on historical demand, I recommend increasing rice inventory before Diwali."}
    ]
    memories = await extract_memories("user1", "store1", messages)
    assert any(m.memory_type == "assistant_recommendation" and m.source_role == "assistant" for m in memories)
