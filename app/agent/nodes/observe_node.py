"""
app/agent/nodes/observe_node.py
================================
Observe node — converts tool results to ToolMessages and increments the loop counter.
"""

from __future__ import annotations

import json
from typing import Dict, Any

import structlog

from app.agent.state import VyaparAgentState
from langchain_core.messages import ToolMessage

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.observe")


async def observe_node(state: VyaparAgentState) -> Dict[str, Any]:
    tool_results = state.get("tool_results", [])
    current_loop = state.get("loop_count", 0)
    store_id = state.get("store_id", "unknown")

    # Only create ToolMessages for results produced in this exact loop iteration
    this_loop_results = [r for r in tool_results if r.get("loop_index") == current_loop]

    LOGGER.info(
        "observe_node_start",
        loop=current_loop,
        store_id=store_id,
        results_this_loop=len(this_loop_results),
        total_results_accumulated=len(tool_results),
    )

    new_messages = []
    for res in this_loop_results:
        if res["success"]:
            content = json.dumps(res["data"], default=str)
            LOGGER.debug(
                "observe_node_tool_message",
                tool_name=res["tool_name"],
                loop=current_loop,
                data_keys=list(res["data"].keys()) if isinstance(res["data"], dict) else "non-dict",
            )
        else:
            content = f"Error from {res['tool_name']}: {res['error']}"
            LOGGER.warning(
                "observe_node_tool_error_message",
                tool_name=res["tool_name"],
                loop=current_loop,
                error=res["error"],
            )

        new_messages.append(
            ToolMessage(
                content=content,
                name=res["tool_name"],
                tool_call_id=res["call_id"],
            )
        )

    next_loop = current_loop + 1
    LOGGER.info(
        "observe_node_complete",
        loop=current_loop,
        next_loop=next_loop,
        messages_appended=len(new_messages),
    )

    return {
        "loop_count": next_loop,
        "tools_called_this_loop": [],
        "messages": new_messages,
    }
