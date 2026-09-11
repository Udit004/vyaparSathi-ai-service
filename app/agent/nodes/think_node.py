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

    llm = get_llm()
    if not llm:
        LOGGER.error("think_node_no_llm", store_id=store_id)
        return {"error": "LLM not configured — check GEMINI_API_KEY in .env", "goal_status": "failed"}

    t0 = time.perf_counter()
    
    # Force the LLM to output a direct response (no tools) if we hit the loop ceiling
    is_final_loop = loop >= state.get("max_loops", 5)
    
    if is_final_loop:
        LOGGER.info("think_node_max_loops_reached", loop=loop, store_id=store_id)
        llm_with_tools = llm # No tools bound, forces a text response
    else:
        llm_with_tools = llm.bind_tools(VYAPAR_TOOLS)

    # Build system prompt — richer on later loops to guide synthesis
    tools_called = state.get("tools_called_this_loop", [])
    available = [t for t in state.get("available_tools", []) if t not in tools_called]
    results_so_far = state.get("tool_results", [])

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
        f"Current loop: {loop + 1} / {state.get('max_loops', 5)}\n\n"
    )

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
        
        return {
            "goal_status": "complete",
            "messages": [response],
            "final_answer": final_text,
            "response_metadata": {
                "loops_taken": loop,
                "tools_used": tools_used,
                "goal_status": "complete",
                "forced_stop": is_final_loop,
                "memory_loaded": {
                    "user": state.get("user_memory_loaded", False),
                    "store": state.get("store_memory_loaded", False),
                },
            }
        }
