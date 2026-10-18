"""
app/mcp/http_handler.py
=======================
Handles direct HTTP POST JSON-RPC requests for MCP clients (like ChatGPT).
Supports:
- initialize
- ping
- notifications/initialized
- tools/list
- tools/call
"""
from typing import Any, Dict
from fastapi import Request
from fastapi.responses import JSONResponse
import structlog
from app.mcp.server import mcp

LOGGER = structlog.get_logger("vyaparsathi.mcp.http")

async def handle_mcp_jsonrpc(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}},
            status_code=400,
        )

    method = body.get("method")
    req_id = body.get("id")
    params = body.get("params", {})

    LOGGER.info("mcp_jsonrpc_request", method=method, req_id=req_id)

    if method == "initialize":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {
                        "listChanged": False
                    }
                },
                "serverInfo": {
                    "name": "VyaparSathi-Assistant",
                    "version": "1.0.0"
                }
            }
        })

    elif method in ("notifications/initialized", "initialized"):
        if req_id is not None:
            return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": {}})
        return JSONResponse({}, status_code=204)

    elif method == "ping":
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": {}})

    elif method == "tools/list":
        tools = await mcp.list_tools()
        tools_dict = [t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in tools]
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": tools_dict
            }
        })

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            call_res = await mcp.call_tool(tool_name, arguments)
            content_list = []
            if hasattr(call_res, "content") and call_res.content:
                for c in call_res.content:
                    content_list.append({"type": getattr(c, "type", "text"), "text": getattr(c, "text", str(c))})
            else:
                content_list.append({"type": "text", "text": str(call_res)})

            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": content_list,
                    "isError": getattr(call_res, "is_error", False)
                }
            })
        except Exception as e:
            LOGGER.exception("tool_call_failed", tool=tool_name, error=str(e))
            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error executing tool: {str(e)}"}],
                    "isError": True
                }
            })

    return JSONResponse({
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": -32601,
            "message": f"Method '{method}' not found"
        }
    })
