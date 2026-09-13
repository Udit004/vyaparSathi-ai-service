"""
app/agent/nodes/think_node.py
==============================
Think node — the LLM decides which tools to call or declares the goal complete.
"""

from __future__ import annotations

import time
from typing import Dict, Any

import structlog

from app.agent.state import VyaparAgentState, ToolCall
from app.lib.llm import get_llm
from app.agent.tools.registry import VYAPAR_TOOLS
from langchain_core.messages import SystemMessage

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.think")


async def think_node(state: VyaparAgentState) -> Dict[str, Any]:
    loop = state.get("loop_count", 0)
    store_id = state.get("store_id", "unknown")
    user_id = state.get("user_id", "unknown")

    LOGGER.info(
        "think_node_start",
        loop=loop,
        store_id=store_id,
        user_id=user_id,
        goal=state.get("goal"),
        goal_status=state.get("goal_status"),
        pending_tool_calls_count=len(state.get("pending_tool_calls", [])),
        tool_results_accumulated=len(state.get("tool_results", [])),
    )

    # Check memory status
    user_memory_loaded = state.get("user_memory_loaded", False)
    store_memory_loaded = state.get("store_memory_loaded", False)

    # If memory hasn't been loaded yet and we're not on the final loop,
    # skip LLM invocation entirely — route directly to memory_query node.
    # The LLM should only reason AFTER it has access to long-term memory.
    if not user_memory_loaded and not store_memory_loaded:
        max_loops = state.get("max_loops", 3)
        is_final_loop = loop >= max_loops
        if not is_final_loop:
            LOGGER.info(
                "think_node_skip_memory_not_loaded",
                loop=loop,
                store_id=store_id,
                reason="routing to memory_query node before LLM invocation",
            )
            return {"memory_query_needed": True}

    llm = get_llm()
    if not llm:
        LOGGER.error("think_node_no_llm", store_id=store_id)
        return {"error": "LLM not configured — check GEMINI_API_KEY in .env", "goal_status": "failed"}

    t0 = time.perf_counter()
    
    # Force the LLM to output a direct response (no tools) if we hit the loop ceiling
    is_final_loop = loop >= state.get("max_loops", 3)
    
    if is_final_loop:
        LOGGER.info("think_node_max_loops_reached", loop=loop, store_id=store_id)
        llm_with_tools = llm # No tools bound, forces a text response
    else:
        llm_with_tools = llm.bind_tools(VYAPAR_TOOLS)

    # Build system prompt — richer on later loops to guide synthesis
    tools_called = state.get("tools_called_this_loop", [])
    available = [t for t in state.get("available_tools", []) if t not in tools_called]
    results_so_far = state.get("tool_results", [])
    max_loops = state.get("max_loops", 3)

    # Extract/normalize goal on first loop
    current_goal = state.get("goal", "")
    goal_status = state.get("goal_status", "pending")
    if not current_goal and goal_status == "pending":
        current_goal = f"Answer the user's question: {state.get('user_prompt', '')}"
        goal_status = "in_progress"

    sys_content = (
        f"You are Vyapar Copilot, an AI assistant for Indian retail stores.\n"
        f"Store ID: {store_id}\n"
        f"User's question: \"{state.get('user_prompt', '')}\"\n"
        f"Goal: {current_goal}\n"
        f"Current loop: {loop + 1} / {max_loops}\n\n"
    )

    # Inject memory context if available
    user_prefs = state.get("user_preferences", {})
    store_knowledge = state.get("store_knowledge", {})
    user_mem_summary = user_prefs.get("summary", "") if isinstance(user_prefs, dict) else ""
    store_mem_summary = store_knowledge.get("summary", "") if isinstance(store_knowledge, dict) else ""

    if user_mem_summary or store_mem_summary:
        sys_content += "Long-term memory context:\n"
        if user_mem_summary:
            sys_content += f"<user_preferences>\n{user_mem_summary}\n</user_preferences>\n\n"
        if store_mem_summary:
            sys_content += f"<store_knowledge>\n{store_mem_summary}\n</store_knowledge>\n\n"

    if results_so_far:
        import json
        sys_content += (
            f"You have already gathered data from {len(results_so_far)} tool call(s). "
            f"If you have enough information to fully answer the user's question, "
            f"respond directly WITHOUT calling any more tools. "
            f"Synthesize the gathered data into a comprehensive, actionable response. "
            f"Be specific — reference actual numbers from the data. "
            f"Format clearly with bullet points or sections if appropriate.\n\n"
        )
        # Inject the context buckets for the LLM to synthesize
        inv = state.get("inventory_context", {})
        sal = state.get("sales_context", {})
        fct = state.get("forecast_context", {})
        ins = state.get("insights_context", [])
        
        sys_content += (
            f"<inventory_data>\n{json.dumps(inv, default=str)}\n</inventory_data>\n\n"
            f"<sales_data>\n{json.dumps(sal, default=str)}\n</sales_data>\n\n"
            f"<forecast_data>\n{json.dumps(fct, default=str)}\n</forecast_data>\n\n"
            f"<insights>\n{json.dumps(ins, default=str)}\n</insights>\n\n"
        )
    else:
        sys_content += (
            f"Use the available tools to gather the data you need. "
            f"Call only the tools that are relevant — do not over-fetch.\n\n"
        )

    if available and not is_final_loop:
        sys_content += f"Available tools this loop: {', '.join(available)}\n"

    sys_msg = SystemMessage(content=sys_content)
    messages = [sys_msg] + state["messages"]

    LOGGER.debug(
        "think_node_invoking_llm",
        loop=loop,
        message_count=len(messages),
        available_tools=available,
        is_final_loop=is_final_loop
    )

    try:
        response = await llm_with_tools.ainvoke(messages)
    except Exception as exc:
        LOGGER.error("think_node_llm_error", loop=loop, error=str(exc), exc_info=True)
        return {"error": str(exc), "goal_status": "failed"}

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    if hasattr(response, "tool_calls") and response.tool_calls:
        pending_calls = [
            ToolCall(
                tool_name=tc["name"],
                arguments=tc["args"],
                call_id=tc["id"],
            )
            for tc in response.tool_calls
        ]
        tool_names = [c["tool_name"] for c in pending_calls]

        LOGGER.info(
            "think_node_decided_tools",
            loop=loop,
            tools_requested=tool_names,
            llm_latency_ms=elapsed_ms,
        )

        return {
            "pending_tool_calls": pending_calls,
            "tools_called_this_loop": tool_names,
            "messages": [response],
            "goal": current_goal,
            "goal_status": goal_status,
        }
    else:
        LOGGER.info(
            "think_node_decided_complete",
            loop=loop,
            llm_latency_ms=elapsed_ms,
        )
        final_text = response.content if hasattr(response, "content") else str(response)
        tools_used = list({r["tool_name"] for r in state.get("tool_results", [])})
        
        # Determine if this conversation is worth persisting to mem0.
        # Skip trivial exchanges ("hi", "hello", "ok", etc.) to avoid
        # polluting long-term memory with noise.
        should_persist = _is_conversation_meaningful(
            user_prompt=state.get("user_prompt", ""),
            final_answer=final_text,
            tools_used=tools_used,
            loop_count=loop,
        )
        
        return {
            "goal_status": "complete",
            "goal": current_goal,
            "messages": [response],
            "final_answer": final_text,
            "should_persist_memory": should_persist,
            "response_metadata": {
                "loops_taken": loop,
                "tools_used": tools_used,
                "goal_status": "complete",
                "goal": current_goal,
                "forced_stop": is_final_loop,
                "memory_loaded": {
                    "user": state.get("user_memory_loaded", False),
                    "store": state.get("store_memory_loaded", False),
                },
            }
        }


# ---------------------------------------------------------------------------
# Conversation importance classifier
# ---------------------------------------------------------------------------

# Trivial greetings / acknowledgements that should NOT be persisted
_TRIVIAL_PATTERNS = [
    "hi", "hello", "hey", "hi there", "hello there",
    "ok", "okay", "yes", "no", "yeah", "nope",
    "thanks", "thank you", "thank", "thx",
    "good morning", "good afternoon", "good evening",
    "how are you", "are you there", "you there",
    "help", "help me",
]

_MIN_MEANINGFUL_LENGTH = 15
# Minimum characters for a response to be considered substantive


def _is_conversation_meaningful(
    *,
    user_prompt: str,
    final_answer: str,
    tools_used: list[str],
    loop_count: int,
) -> bool:
    """
    Determine whether a conversation contains enough substance to persist
    to long-term mem0 memory.

    Returns False for trivial exchanges ("hi", "hello", "ok") to avoid
    polluting memory with noise.

    Returns True when:
        - The agent used tools (real data was gathered)
        - The user prompt is substantive (not a greeting)
        - The final answer is substantive (long enough, not just "sure")
    """
    # If tools were used, the conversation has real data — always persist
    if tools_used:
        return True

    # Multiple loops means the agent did real reasoning work
    if loop_count > 0:
        return True

    # Check if user prompt is trivial
    prompt_lower = (user_prompt or "").strip().lower()
    if prompt_lower in _TRIVIAL_PATTERNS:
        return False

    # Short prompts (< 15 chars) that aren't greetings are likely trivial
    if len(prompt_lower) < _MIN_MEANINGFUL_LENGTH:
        return False

    # Check if the final answer is substantive
    answer_lower = (final_answer or "").strip().lower()
    if len(answer_lower) < _MIN_MEANINGFUL_LENGTH:
        return False

    # If the answer is just a greeting back, skip
    if answer_lower in _TRIVIAL_PATTERNS:
        return False

    return True
