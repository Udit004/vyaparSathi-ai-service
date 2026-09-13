"""
app/agent/nodes/observe_node.py
================================
Observe node — converts tool results to ToolMessages and increments the loop counter.

Tool payloads can be very large (full inventory lists, sales histories,
forecasts). To keep the LLM context window bounded, each payload is
compressed with a small/fast summarizer (GROQ/NVIDIA) before it becomes
a ToolMessage. The full raw data still lives in the *_context buckets
and the persisted state — only the message the LLM actually reads is
summarized.
"""

from __future__ import annotations

import json
from typing import Dict, Any

import structlog

from app.agent.state import VyaparAgentState
from app.lib.summarizer import summarize
from app.agent.prompts.summarizer_prompts import build_summarizer_tool_instruction
from langchain_core.messages import ToolMessage

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.observe")

# Payloads smaller than this are passed through verbatim — no need to
# spend a summarizer call on trivial data.
_MIN_SUMMARIZE_CHARS = 600


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
            raw = res["data"]
            content_json = json.dumps(raw, default=str)

            # Compress large payloads for the LLM context window.
            # The full raw data remains in *_context buckets (set by the
            # tool node) and in the persisted state, so nothing is lost.
            if len(content_json) > _MIN_SUMMARIZE_CHARS:
                summary = await summarize(
                    content_json,
                    instruction=build_summarizer_tool_instruction(res["tool_name"]),
                    max_tokens=300,
                )
                if summary:
                    content = f"{summary}\n\n[Full data retained in state; summarized for context]"
                    LOGGER.debug(
                        "observe_node_summarized",
                        tool_name=res["tool_name"],
                        loop=current_loop,
                        in_chars=len(content_json),
                        out_chars=len(content),
                    )
                else:
                    content = content_json
            else:
                content = content_json

            LOGGER.debug(
                "observe_node_tool_message",
                tool_name=res["tool_name"],
                loop=current_loop,
                data_keys=list(raw.keys()) if isinstance(raw, dict) else "non-dict",
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
