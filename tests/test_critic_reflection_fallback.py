"""
tests/test_critic_reflection_fallback.py
=========================================
Unit tests verifying that critic_node and reflection_node support model fallbacks.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.nodes.critic_node import critic_node
from app.agent.nodes.reflection_node import reflection_node
from app.lib.llm import _FallbackLLM


@pytest.mark.asyncio
async def test_critic_node_fallback_behavior():
    """Verify critic_node executes structured output and config chaining across _FallbackLLM."""
    state = {
        "goal": "Verify sales report",
        "final_answer": "Sales for today is 50,000 INR.",
        "tool_results": [{"tool_name": "get_sales_report", "success": True}],
    }

    # Test 1: Fallback LLM succeeds
    mock_member1 = MagicMock()
    mock_member1.with_structured_output.side_effect = RuntimeError("Rate limit 429")

    mock_member2 = MagicMock()
    mock_structured = MagicMock()
    mock_structured.with_config.return_value = mock_structured
    mock_result = MagicMock()
    mock_result.status = "PASS"
    mock_result.reason = "Goal fully satisfied."
    mock_result.missing_information = []
    mock_result.needs_retry = False
    mock_structured.ainvoke = AsyncMock(return_value=mock_result)
    mock_member2.with_structured_output.return_value = mock_structured

    fallback = _FallbackLLM([mock_member1, mock_member2], ["gemini/flash", "groq/llama"])

    with patch("app.agent.nodes.critic_node.get_llm", return_value=fallback):
        res = await critic_node(state)
        assert res["goal_status"] == "complete"
        assert res["critic_status"] == "PASS"


@pytest.mark.asyncio
async def test_reflection_node_fallback_behavior():
    """Verify reflection_node executes ainvoke across _FallbackLLM."""
    state = {
        "critic_reason": "Missing sales forecast data",
        "critic_missing_information": ["sales_forecast"],
        "loop_count": 1,
    }

    mock_member1 = MagicMock()
    mock_member1.ainvoke = AsyncMock(side_effect=RuntimeError("Quota 429"))

    mock_member2 = MagicMock()
    mock_res = MagicMock()
    mock_res.content = "Fetch sales forecast next using the forecast tool."
    mock_member2.ainvoke = AsyncMock(return_value=mock_res)

    fallback = _FallbackLLM([mock_member1, mock_member2], ["gemini/flash", "groq/llama"])

    with patch("app.agent.nodes.reflection_node.get_llm", return_value=fallback):
        res = await reflection_node(state)
        assert res["goal_status"] == "in_progress"
        assert "forecast" in res["reflection"]
        assert res["loop_count"] == 2
