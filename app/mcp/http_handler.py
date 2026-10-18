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
from bson import ObjectId
import structlog
from app.config.database import get_database
from app.mcp.server import mcp, current_store_context

LOGGER = structlog.get_logger("vyaparsathi.mcp.http")


async def resolve_store_from_request(request: Request) -> str:
    """
    Resolve which store to query based on request parameters:
    1. Query param `?store=...` or `?store_id=...`
    2. Query param `?user=...` or `?email=...` (maps user email/name to their owned store in MongoDB)
    3. Header `Authorization: Bearer <email_or_store>`
    4. Custom headers `x-store-id` or `x-user-id`
    """
    db = get_database()

    # Direct store query parameter
    store_param = request.query_params.get("store") or request.query_params.get("store_id") or request.headers.get("x-store-id")
    if store_param:
        return store_param.strip()

    # User identifier (email, user ID, or Bearer token)
    user_param = (
        request.query_params.get("user")
        or request.query_params.get("email")
        or request.headers.get("x-user-id")
        or request.headers.get("x-user-email")
    )

    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth.replace("Bearer ", "").strip()
        # 1. Check if token is an OAuth JWT token
        from app.mcp.oauth import decode_access_token
        jwt_claims = decode_access_token(token)
        if jwt_claims:
            target = jwt_claims.get("store_name") or jwt_claims.get("store_id")
            if target:
                LOGGER.info("authenticated_via_oauth_jwt", store=target, user=jwt_claims.get("email"))
                return str(target)

        if token and not user_param:
            user_param = token

    if user_param:
        cleaned_user = user_param.strip()
        # 1. Try finding user by ObjectId or email or name
        user = None
        try:
            user = await db["users"].find_one({"_id": ObjectId(cleaned_user)})
        except Exception:
            pass

        if not user:
            user = await db["users"].find_one({
                "$or": [
                    {"email": cleaned_user.lower()},
                    {"name": {"$regex": f"^{cleaned_user}$", "$options": "i"}}
                ]
            })

        if user:
            store = await db["stores"].find_one({"owner": user["_id"]})
            if store:
                return str(store.get("name") or store["_id"])

        # 2. Try directly matching store name
        store = await db["stores"].find_one({"name": {"$regex": f"^{cleaned_user}$", "$options": "i"}})
        if store:
            return str(store.get("name") or store["_id"])

    return ""


async def handle_mcp_jsonrpc(request: Request) -> JSONResponse:
    # Resolve store context from request query params / headers
    try:
        resolved_store = await resolve_store_from_request(request)
        if resolved_store:
            current_store_context.set(resolved_store)
            LOGGER.info("resolved_tenant_store", store=resolved_store)
    except Exception as exc:
        LOGGER.warning("failed_to_resolve_tenant_store", error=str(exc))

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
        tools_dict = []
        for t in tools:
            schema = t.inputSchema.model_dump() if hasattr(t.inputSchema, "model_dump") else t.inputSchema
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
            if "type" not in schema:
                schema["type"] = "object"
            tools_dict.append({
                "name": t.name,
                "description": (t.description or "").strip(),
                "inputSchema": schema,
            })
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

            # FastMCP call_tool returns a tuple (list_of_blocks, meta_dict) or list of blocks
            blocks = call_res
            is_error = False
            if isinstance(call_res, tuple):
                if len(call_res) > 0:
                    blocks = call_res[0]
                if len(call_res) > 1 and isinstance(call_res[1], dict):
                    is_error = call_res[1].get("is_error", False)
            elif hasattr(call_res, "content"):
                blocks = call_res.content
                is_error = getattr(call_res, "is_error", False)

            if isinstance(blocks, (list, tuple)):
                for c in blocks:
                    if hasattr(c, "text"):
                        content_list.append({"type": getattr(c, "type", "text"), "text": c.text})
                    elif isinstance(c, dict) and "text" in c:
                        content_list.append(c)
                    else:
                        content_list.append({"type": "text", "text": str(c)})
            elif hasattr(blocks, "text"):
                content_list.append({"type": getattr(blocks, "type", "text"), "text": blocks.text})
            elif isinstance(blocks, str):
                content_list.append({"type": "text", "text": blocks})
            else:
                import json
                try:
                    content_list.append({"type": "text", "text": json.dumps(blocks, default=str)})
                except Exception:
                    content_list.append({"type": "text", "text": str(blocks)})

            return JSONResponse({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": content_list,
                    "isError": is_error
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
