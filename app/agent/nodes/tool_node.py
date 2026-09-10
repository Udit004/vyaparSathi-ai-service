"""
app/agent/nodes/tool_node.py
=============================
Tool node — executes tool calls requested by the think node.
"""

from __future__ import annotations

import time
from typing import Dict, Any

import structlog

from app.agent.state import VyaparAgentState, make_tool_result
from app.agent.tools.registry import get_tool_by_name

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.tool")


async def tool_node(state: VyaparAgentState) -> Dict[str, Any]:
    pending = state.get("pending_tool_calls", [])
    loop = state.get("loop_count", 0)
    store_id = state.get("store_id", "unknown")

    LOGGER.info(
        "tool_node_start",
        loop=loop,
        store_id=store_id,
        tools_to_run=[c["tool_name"] for c in pending],
    )

    results = []
    updates: Dict[str, Any] = {"pending_tool_calls": []}

    for call in pending:
        tool_name = call["tool_name"]
        tool = get_tool_by_name(tool_name)

        if not tool:
            LOGGER.warning("tool_node_unknown_tool", tool_name=tool_name, loop=loop)
            results.append(
                make_tool_result(
                    call_id=call["call_id"],
                    tool_name=tool_name,
                    data=None,
                    error=f"Tool '{tool_name}' not found in registry",
                    loop_index=loop,
                )
            )
            continue

        t0 = time.perf_counter()
        try:
            # Ensure store_id is always passed to every tool
            args = dict(call["arguments"])
            if "store_id" not in args:
                args["store_id"] = store_id

            LOGGER.debug("tool_node_invoking", tool_name=tool_name, args=args, loop=loop)
            data = await tool.ainvoke(args)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

            raw_data = data.model_dump() if hasattr(data, "model_dump") else data

            LOGGER.info(
                "tool_node_success",
                tool_name=tool_name,
                loop=loop,
                latency_ms=elapsed_ms,
            )

            results.append(
                make_tool_result(
                    call_id=call["call_id"],
                    tool_name=tool_name,
                    data=raw_data,
                    loop_index=loop,
                )
            )

            # Route data into the correct context bucket
            if "inventory" in tool_name or "stock" in tool_name:
                ctx = dict(state.get("inventory_context", {}))
                ctx[tool_name] = raw_data
                updates["inventory_context"] = ctx
            elif "sales" in tool_name or "selling" in tool_name:
                ctx = dict(state.get("sales_context", {}))
                ctx[tool_name] = raw_data
                updates["sales_context"] = ctx
            elif "forecast" in tool_name or "restock" in tool_name:
                ctx = dict(state.get("forecast_context", {}))
                ctx[tool_name] = raw_data
                updates["forecast_context"] = ctx
            elif "insights" in tool_name:
                ctx = list(state.get("insights_context", []))
                if isinstance(raw_data, list):
                    ctx.extend(raw_data)
                else:
                    ctx.append(raw_data)
                updates["insights_context"] = ctx

        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            LOGGER.error(
                "tool_node_error",
                tool_name=tool_name,
                loop=loop,
                error=str(exc),
                latency_ms=elapsed_ms,
                exc_info=True,
            )
            results.append(
                make_tool_result(
                    call_id=call["call_id"],
                    tool_name=tool_name,
                    data=None,
                    error=str(exc),
                    loop_index=loop,
                )
            )

    LOGGER.info(
        "tool_node_complete",
        loop=loop,
        results_count=len(results),
        successes=sum(1 for r in results if r["success"]),
        failures=sum(1 for r in results if not r["success"]),
    )

    updates["tool_results"] = results
    return updates
