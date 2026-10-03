"""
app/routes/voice_routes.py
==========================
WebSocket endpoint for Gemini Realtime API (Multimodal Live API).

This route acts as a relay between the Vyapar Sathi frontend and the
Google Gemini Multimodal API via raw websockets.
"""

import json
import os
import structlog
import asyncio
import base64
import math
import struct
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import websockets

from app.config.settings import get_settings
from app.agent.tools.registry import VYAPAR_TOOLS
from app.agent.nodes.memory_query import _bootstrap_user_memory, _bootstrap_store_memory

router = APIRouter()
LOGGER = structlog.get_logger("vyaparsathi.ai.voice")


def _decode_gemini_message(msg):
    """Decode Gemini messages safely. Some frames are binary audio chunks, not JSON."""
    if isinstance(msg, (bytes, bytearray)):
        payload = bytes(msg)
        if not payload:
            return None
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"binary": payload}

    if isinstance(msg, str):
        text = msg.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"binary": text.encode("utf-8")}

    return None


def _clean_gemini_schema(node: Any) -> Any:
    """Recursively clean JSON Schema for Gemini Live API compatibility.

    Gemini Live API rejects unknown fields such as additionalProperties, title,
    default, $defs, allOf, etc.
    """
    if not isinstance(node, dict):
        if isinstance(node, list):
            return [_clean_gemini_schema(item) for item in node]
        return node

    cleaned: Dict[str, Any] = {}

    # Handle anyOf / oneOf (e.g. Optional[T] / Union[T, None])
    if "anyOf" in node or "oneOf" in node:
        sub_schemas = node.get("anyOf") or node.get("oneOf", [])
        non_null = [s for s in sub_schemas if isinstance(s, dict) and s.get("type") != "null"]
        if non_null:
            merged = _clean_gemini_schema(non_null[0])
            if isinstance(merged, dict):
                cleaned.update(merged)
                if any(isinstance(s, dict) and s.get("type") == "null" for s in sub_schemas):
                    cleaned["nullable"] = True
        else:
            cleaned["type"] = "string"

    # Copy allowed schema fields
    for k, v in node.items():
        if k in (
            "additionalProperties", "title", "$defs", "$ref", "default", "examples",
            "prefixItems", "discriminator", "readOnly", "writeOnly", "xml", "externalDocs",
            "example", "deprecated"
        ):
            continue
        if k == "properties" and isinstance(v, dict):
            cleaned["properties"] = {
                prop_name: _clean_gemini_schema(prop_val)
                for prop_name, prop_val in v.items()
            }
        elif k == "items":
            cleaned["items"] = _clean_gemini_schema(v)
        elif k not in ("anyOf", "oneOf"):
            cleaned[k] = _clean_gemini_schema(v) if isinstance(v, (dict, list)) else v

    # Ensure valid type on objects with properties
    if "properties" in cleaned and "type" not in cleaned:
        cleaned["type"] = "object"

    return cleaned


def _get_gemini_tools():
    gemini_tools = []
    for tool in VYAPAR_TOOLS:
        args_schema = getattr(tool, "args_schema", None)
        raw_parameters = args_schema.model_json_schema() if args_schema else {
            "type": "object",
            "properties": {},
        }
        parameters = _clean_gemini_schema(raw_parameters)
        gemini_tools.append({
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        })
    return [{"functionDeclarations": gemini_tools}]


def _get_live_model_candidates():
    """Return the officially supported Gemini Live model names in priority order.

    Google currently documents Gemini 3.8 Live as the default Live API model and
    lists 2.5 Flash Live as a valid native-audio model. Older model names such as
    gemini-2.0-flash-live-001 are no longer valid and fail with 1008 policy
    violations.
    """
    return [
        "models/gemini-3.8-live",
        "models/gemini-2.5-flash-native-audio-preview-12-2025",
        "models/gemini-3.1-flash-live-preview",
    ]


def _build_gemini_audio_message(audio_bytes: bytes):
    """Build the Live API realtime audio message for raw 16 kHz PCM."""
    return {
        "realtimeInput": {
            "audio": {
                "mimeType": "audio/pcm;rate=16000",
                "data": base64.b64encode(audio_bytes).decode("ascii"),
            }
        }
    }


def _build_gemini_audio_stream_end_message():
    """Build the Live API signal that ends an explicitly managed audio turn."""
    return {"realtimeInput": {"activityEnd": {}}}


def _build_gemini_activity_start_message():
    """Build the Live API signal that starts an explicitly managed audio turn."""
    return {"realtimeInput": {"activityStart": {}}}


def _pcm_rms(audio_bytes: bytes):
    """Return the RMS level of little-endian 16-bit PCM for transport diagnostics."""
    sample_count = len(audio_bytes) // 2
    if sample_count == 0:
        return 0.0
    samples = struct.unpack(f"<{sample_count}h", audio_bytes[:sample_count * 2])
    return math.sqrt(sum(sample * sample for sample in samples) / sample_count)


async def _connect_gemini_live(api_key: str, system_prompt: str):
    """Connect to Gemini Live using the supported model set, with fallback retries."""
    last_error = None
    gemini_ws = None
    for model_name in _get_live_model_candidates():
        ws_url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={api_key}"
        try:
            gemini_ws = await websockets.connect(ws_url)
            setup_msg = {
                "setup": {
                    "model": model_name,
                    "systemInstruction": {
                        "parts": [{"text": system_prompt}]
                    },
                    "generationConfig": {
                        "responseModalities": ["AUDIO"],
                        "speechConfig": {
                            "voiceConfig": {
                                "prebuiltVoiceConfig": {"voiceName": "Kore"}
                            }
                        }
                    },
                    "inputAudioTranscription": {},
                    "outputAudioTranscription": {},
                    "realtimeInputConfig": {
                        "automaticActivityDetection": {"disabled": True}
                    },
                    "tools": _get_gemini_tools()
                }
            }
            await gemini_ws.send(json.dumps(setup_msg))

            setup_response = await gemini_ws.recv()
            LOGGER.info("gemini_realtime_connected", model=model_name, response=setup_response)
            return gemini_ws
        except Exception as exc:  # Keep retrying on unsupported/legacy model names.
            last_error = exc
            if gemini_ws is not None:
                await gemini_ws.close()
                gemini_ws = None
            LOGGER.warning(
                "gemini_live_model_retry",
                model=model_name,
                error=str(exc),
            )

    raise last_error or RuntimeError("Unable to establish Gemini Live websocket connection")


from app.agent.memory.redis_cache import (
    get_cached_store_memory,
    get_cached_user_memory,
    set_cached_store_memory,
    set_cached_user_memory,
)


async def _bootstrap_voice_memory_context(user_id: str, store_id: str):
    """
    Multi-Tier Memory Bootstrap:
    1. Check fast Redis cache first.
    2. Fall back to Pinecone vector DB if cache misses.
    3. Cache retrieved memories back into Redis for sub-millisecond future turns.
    """
    user_memories = None
    store_memories = None

    # 1. Check Redis Cache
    try:
        user_memories = await get_cached_user_memory(user_id)
        store_memories = await get_cached_store_memory(store_id)
    except Exception as exc:
        LOGGER.warning("voice_redis_cache_read_failed", error=str(exc))

    # 2. Fall back to Pinecone for user memories if not cached
    if user_memories is None:
        try:
            user_memories = await _bootstrap_user_memory(user_id, "user preferences and details")
            if user_memories:
                await set_cached_user_memory(user_id, user_memories)
        except Exception as exc:
            LOGGER.warning("voice_user_memory_bootstrap_failed", user_id=user_id, error=str(exc))
            user_memories = []

    # 3. Fall back to Pinecone for store memories if not cached
    if store_memories is None:
        try:
            store_memories = await _bootstrap_store_memory(store_id, "store details and facts", intent="general")
            if store_memories:
                await set_cached_store_memory(store_id, store_memories)
        except Exception as exc:
            LOGGER.warning("voice_store_memory_bootstrap_failed", store_id=store_id, error=str(exc))
            store_memories = []

    return user_memories or [], store_memories or []


def _build_voice_system_prompt(user_id: str, store_id: str, memory_context: str):
    """
    Build the voice assistant system prompt with role allocation, proactive reasoning,
    memory tool guidelines, and few-shot voice conversational examples.
    """
    return f"""You are Vyapar Sathi (व्यापार साथी) — an expert, proactive AI Retail Business Partner and Inventory Co-Pilot for Indian store owners.

==================================================
ROLE & PERSONALITY
==================================================
- Tone: Warm, energetic, sharp, decisive, and natural (like a smart, trusted business partner).
- Language: Speak fluent natural Hinglish, Hindi, or English matching the store owner's speech.
- Brevity for Voice: Keep voice responses concise, punchy, and conversational (1 to 3 short sentences per turn).
- Monetary Values: Always express money in Rupees (₹), e.g., "₹450", "₹1,200".

==================================================
PROACTIVE REASONING & INVENTORY MANAGEMENT
==================================================
1. Do not just report raw numbers passively — explain what they mean for business:
   - When stock is low, calculate daily burn rate, factor in distributor lead time (usually 2-3 days), and recommend exact reorder quantities.
   - When asked about fast-moving items or revenue, highlight profit margins and stockout prevention.
2. Inventory & Product Actions:
   - When the owner asks to ADD stock (e.g. "Add 20 to Coca Cola stock", "20 packets aur add kar do", "20 pieces aaye hain"):
     ALWAYS call `tool_adjust_stock` with `quantity_to_add` and `product_name` (or `tool_update_product` with `add_quantity`).
     CRITICAL: NEVER pass `quantity=20` to replace stock when the owner asks to add stock. Always use `quantity_to_add` / `add_quantity` so it adds on top of current stock.
   - When the owner asks to set the absolute stock count explicitly (e.g. "Dukaan me exact 50 piece bache hain count set kar do"): Use `quantity=50` in `tool_update_product`.
   - When asked to update prices or other details, use `tool_update_product`.
3. Purchases & Supplier Orders:
   - When the owner asks to place an order or record a purchase: execute `tool_create_purchase`.
   - When the owner asks what products are in a purchase order, what needs to be ordered, or inquires about a PO (e.g. "PO-20261003-ADF4 me kaunse products hain?", "Iss PO se kya order karna hai?"):
     First call `search_purchases` with `query="PO-20261003-ADF4"`. If not found or if it was a draft restock order, call `read_scratchpad_notes` with `query="PO-20261003-ADF4"`. Explicitly list the product names, quantities, and supplier details.
   - When the owner asks to email a purchase order or message to a supplier/seller (e.g. "Ye PO Ramesh Traders ko bhej do", "Supplier ko mail kar do"):
     - If the owner DID NOT specify the language (English, Hindi, or Hinglish) in their current turn: DO NOT call `send_store_email` immediately.
     - FIRST ASK: "Aap ye email kaunsi language me bhejna chahte hain — English, Hindi, ya Hinglish?"
     - Once the owner replies (e.g. "Hindi me", "English", "Hinglish"): execute `send_store_email` with `language="<chosen_language>"` formatting the email in that language.
   - When the owner asks to update an existing purchase order or invoice (e.g. "Tiwari Traders ke invoice PO-20261003-ADF4 me payment update kar do", "Mark PO-XXX as paid", "Change purchase notes"): execute `tool_update_purchase` with `purchase_identifier="PO-20261003-ADF4"`.
   - When the owner asks to delete or cancel a purchase order: execute `tool_delete_purchase` with `purchase_identifier="PO-20261003-ADF4"`.

==================================================
STORE & OWNER MEMORY TOOLS (HOW TO USE PROPERLY)
==================================================
Always supply the correct query and parameters when calling memory tools:
1. `get_owner_goals_and_preferences`:
   - When to call: When the owner asks about their own targets, sales goals, profit margin aims, or personal preferences (e.g., "Mera iss mahine ka target kya hai?", "Meri kya preferences hain?").
   - Arguments: `topic="business goals targets preferences"`, `user_id="{user_id}"`, `store_id="{store_id}"`.
2. `set_owner_goal_or_preference`:
   - When to call: When the owner states a personal target or preference (e.g., "Mera goal hai iss mahine ₹5 lakh sales karna", "Mujhe Hindi me brief alert bheja karo").
   - Arguments: `goal_or_preference="<the goal or preference statement>"`, `category="business_goal"`, `user_id="{user_id}"`, `store_id="{store_id}"`.
3. `search_memory`:
   - When to call: When looking up past decisions, supplier discount agreements, customer credit rules, or general store history.
   - Arguments:
     - `query`: A concise descriptive search phrase (e.g. "Ramesh supplier discount", "customer credit rule", "monthly revenue target", or "*" to list all).
     - `scope`: "all" (search both owner & store facts), "user" (for owner personal targets/preferences), or "store" (for store-level business facts).
     - `user_id`: "{user_id}"
     - `store_id`: "{store_id}"
4. `remember_store_fact`:
   - When to call: When the owner shares a store-wide business rule or supplier term (e.g., "Supplier ABC gives 5% discount on cash orders above 10 boxes").
   - Arguments: `content="<the business rule>"`, `memory_type="store_fact"`, `user_id="{user_id}"`, `store_id="{store_id}"`.

==================================================
IMMUTABLE SESSION CONTEXT
==================================================
user_id={user_id}
store_id={store_id}
For every tool call, use exactly this store_id and user_id.

==================================================
STORE & OWNER MEMORY BANK (REDIS & PINECONE)
==================================================
{memory_context}

==================================================
FEW-SHOT VOICE CONVERSATIONAL EXAMPLES
==================================================
[Example 1: Proactive Restock with Lead-Time Warning]
Owner: "Kitna Parle-G bacha hai dukaan me?"
Assistant: (Calls get_product_details) -> "Aapke paas Parle-G ke sirf 6 packets bache hain aur roz lagbhag 15 packets bikte hain. Distributor ko turant 50 packets ka order de dein taaki shaam tak stockout na ho."

[Example 2: Memory Recall for Supplier Pricing]
Owner: "Fortune Oil kaunse supplier se lena chahiye?"
Assistant: (Calls search_memory with query='Fortune Oil supplier terms discount', scope='store') -> "Gupta Traders aapko 10 dabbe lene par 4% cash discount dete hain aur 2 din me delivery kar dete hain. Unhe order karna best rahega."

[Example 3: Setting a Personal Business Goal]
Owner: "Mera goal hai iss mahine ₹4 lakh ka revenue hit karna."
Assistant: (Calls set_owner_goal_or_preference with goal_or_preference='Monthly sales target is ₹4,00,000') -> "Bilkul! Aapka monthly revenue target ₹4,00,000 save kar liya hai. Main aapki inventory aur sales strategy ko isi target ke hisaab se align rakhunga."

[Example 5: Smart Purchase Order & Diary Draft]
Owner: "Low stock items ke liye distributor ka PO bana do."
Assistant: (Calls create_smart_purchase_order) -> "Maine 3-day lead time ke hisaab se 6 critical items ka ₹14,800 ka Purchase Order draft bana kar Merchant Diary me save kar diya hai. Kya main ise distributor ko email kar doon?"

[Example 6: Emailing Purchase Order to Supplier - Asking Language First]
Owner: "Haan, ye PO ramesh.traders@gmail.com par bhej do."
Assistant: "Aap ye email Ramesh Traders ko kaunsi language me bhejna chahte hain — English, Hindi, ya Hinglish?"
Owner: "Hindi me bhej do."
Assistant: (Calls send_store_email with recipient_email="ramesh.traders@gmail.com", language="Hindi") -> "Maine Ramesh Traders ko Hindi me Purchase Order email kar diya hai. Ye purchase order aapke Purchases page par record ho chuka hai aur stock sync ho gaya hai."

[Example 7: Placing an Order / Purchasing from a Seller]
Owner: "Global Traders se 20 packets Tata Salt order kar do."
Assistant: (Calls tool_create_purchase) -> "Maine Global Traders se 20 packets Tata Salt ka ₹560 ka purchase order place kar diya hai. Stock update ho gaya hai aur ye aapke Purchases page par dikhai de raha hai."

[Example 8: Adding Stock to Existing Inventory]
Owner: "Coca-Cola me 20 piece aur add kar do."
Assistant: (Calls tool_adjust_stock with product_name="Coca-Cola", quantity_to_add=20) -> "Maine Coca-Cola me 20 units add kar diye hain. Pehle 10 units the, ab kul 30 units ka stock ho gaya hai."

[Example 9: Updating an Existing Purchase Order / Bill]
Owner: "Tiwari Traders ka invoice PO-20261003-ADF4 me ₹5,000 paid mark kar do."
Assistant: (Calls tool_update_purchase with purchase_identifier="PO-20261003-ADF4", paid_amount=5000) -> "Maine Tiwari Traders ke invoice PO-20261003-ADF4 me ₹5,000 paid mark kar diya hai. Due balance update ho gaya hai aur Purchases page par sync ho chuka hai."

[Example 10: Inspecting Products in a Purchase Order / PO Number]
Owner: "PO-20261003-ADF4 me kaunsa product order karna hai?"
Assistant: (Calls search_purchases with query="PO-20261003-ADF4", or read_scratchpad_notes with query="PO-20261003-ADF4") -> "PO-20261003-ADF4 me aapke paas 50 packets Tata Salt (₹28/unit) aur 20 packets Fortune Oil (₹145/unit) hain, kul total ₹4,300 Tiwari Traders ke naam par hai."
"""


def _get_tool_status_label(fn_name: str, args: dict) -> tuple[str, str]:
    if fn_name == "tool_create_purchase":
        s = args.get("seller_name", "seller")
        return (f"Recording official purchase order from {s} & updating inventory...", "Purchase order created & added to Purchases page!")
    elif fn_name == "tool_update_purchase":
        p = args.get("purchase_identifier", "invoice")
        return (f"Updating purchase order {p}...", "Purchase order updated successfully!")
    elif fn_name == "tool_delete_purchase":
        p = args.get("purchase_identifier", "invoice")
        return (f"Deleting purchase order {p} & restoring stock...", "Purchase order deleted!")
    elif fn_name == "tool_adjust_stock":
        p = args.get("product_name") or "product"
        qty = args.get("quantity_to_add", "")
        return (f"Adding {qty} units to {p} stock...", f"{p} stock updated successfully!")
    elif fn_name == "search_sellers":
        q = args.get("query", "")
        return (f"Searching connected sellers and vendors {f'for {q}' if q else ''}...", "Seller details retrieved!")
    elif fn_name == "search_buyers":
        q = args.get("query", "")
        return (f"Searching customer and buyer records {f'for {q}' if q else ''}...", "Buyer details retrieved!")
    elif fn_name == "search_purchases":
        s = args.get("seller_name", "")
        return (f"Searching purchase orders and supplier bills {f'from {s}' if s else ''}...", "Purchase orders retrieved!")
    elif fn_name == "create_smart_purchase_order":
        return ("Calculating lead-time forecast & generating Purchase Order...", "Purchase Order draft ready & saved to Diary!")
    elif fn_name == "send_store_email":
        rec = args.get("recipient_email") or args.get("to") or "recipient"
        return (f"Sending official email to {rec}...", "Email dispatched successfully!")
    elif fn_name == "write_scratchpad_note":
        return ("Writing note to your Merchant Diary...", "Note saved to Merchant Diary!")
    elif fn_name == "read_scratchpad_notes":
        return ("Reading notes from your Merchant Diary...", "Diary notes retrieved!")
    elif fn_name == "update_scratchpad_note":
        return ("Updating note in your Merchant Diary...", "Diary note updated!")
    elif fn_name == "delete_scratchpad_note":
        return ("Removing note from your Merchant Diary...", "Note removed!")
    elif fn_name == "search_memory":
        q = args.get("query", "")
        return (f"Searching store memory for '{q[:30]}'...", "Memory retrieved!")
    elif fn_name == "remember_store_fact":
        return ("Saving new fact to store memory...", "Memory saved and cached!")
    elif fn_name == "get_owner_goals_and_preferences":
        return ("Retrieving your personal business goals...", "Owner goals loaded!")
    elif fn_name == "set_owner_goal_or_preference":
        return ("Updating your personal business goal...", "Goal saved and cached!")
    elif fn_name == "get_daily_action_checklist":
        return ("Preparing your daily action checklist...", "Daily action plan ready!")
    elif fn_name == "analyze_customer_credit_risk":
        return ("Analyzing customer credit & udhaar risk...", "Credit risk analysis ready!")
    elif fn_name == "calculate_restock_budget":
        return ("Calculating restock budget & distributor payouts...", "Restock budget ready!")
    elif fn_name == "generate_deal_bundle_recommendation":
        return ("Generating profitable promotional combos...", "Bundle deals ready!")
    elif fn_name == "get_store_goal_progress_report":
        return ("Calculating progress towards your monthly goal...", "Goal progress report ready!")
    elif fn_name in ("get_low_stock_products", "get_inventory_summary"):
        return ("Scanning inventory levels...", "Inventory data ready!")
    elif fn_name in ("get_sales_summary", "get_daily_sales_trend", "get_top_selling_products"):
        return ("Analyzing sales performance...", "Sales analytics ready!")
    elif fn_name in ("get_demand_forecast", "get_restock_priorities"):
        return ("Calculating demand forecast...", "Forecast complete!")
    elif fn_name == "tool_update_product":
        return (f"Updating product {args.get('name', '')}...", "Product updated!")
    else:
        clean_name = fn_name.replace("get_", "").replace("tool_", "").replace("_", " ").title()
        return (f"Executing {clean_name}...", f"{clean_name} complete!")


@router.websocket("/ws/voice")
async def voice_assistant_websocket(
    websocket: WebSocket,
    user_id: str = Query(..., description="The user's ID"),
    store_id: str = Query(..., description="The store's ID")
):
    await websocket.accept()

    settings = get_settings()
    api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        await websocket.close(code=1008)
        return

    # 1. Pre-load Memory (best-effort only; Pinecone outages should not break voice chat)
    LOGGER.info("voice_loading_memory", user_id=user_id, store_id=store_id)
    user_memories, store_memories = await _bootstrap_voice_memory_context(user_id, store_id)

    if user_memories or store_memories:
        memory_context = (
            "Here are your long-term memories about the user and their store:\n\n"
            "USER MEMORIES:\n" + "\n".join(m.get("content", "") for m in user_memories) + "\n\n"
            "STORE MEMORIES:\n" + "\n".join(m.get("content", "") for m in store_memories)
        )
    else:
        memory_context = (
            "No long-term memory is currently available for this user/store. "
            "Proceed with a fresh voice conversation using the current context only."
        )

    system_prompt = _build_voice_system_prompt(user_id, store_id, memory_context)

    gemini_ws = None
    try:
        # 2. Connect via raw websockets using the valid Gemini Live models.
        gemini_ws = await _connect_gemini_live(api_key, system_prompt)
        LOGGER.info("setup_complete")
        # Signal the frontend that Gemini is ready
        await websocket.send_json({"type": "ready"})

        # Send client audio to Gemini
        async def receive_from_client():
            audio_chunk_count = 0
            activity_started = False
            try:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        raise WebSocketDisconnect

                    if message.get("bytes") is not None:
                        if not activity_started:
                            await gemini_ws.send(json.dumps(_build_gemini_activity_start_message()))
                            activity_started = True
                            LOGGER.info("voice_activity_start_forwarded_to_gemini")
                        # Forward raw 16 kHz PCM using Gemini Live's current audio shape.
                        chunk = _build_gemini_audio_message(message["bytes"])
                        await gemini_ws.send(json.dumps(chunk))
                        audio_chunk_count += 1
                        if audio_chunk_count == 1 or audio_chunk_count % 25 == 0:
                            LOGGER.info(
                                "voice_audio_forwarded_to_gemini",
                                chunks=audio_chunk_count,
                                bytes=len(message["bytes"]),
                                pcm_rms=round(_pcm_rms(message["bytes"]), 2),
                            )
                        continue

                    if message.get("text"):
                        control = json.loads(message["text"])
                        if control.get("type") == "audio_stream_end":
                            if activity_started:
                                await gemini_ws.send(json.dumps(_build_gemini_audio_stream_end_message()))
                                activity_started = False
                            LOGGER.info(
                                "voice_activity_end_forwarded_to_gemini",
                                chunks=audio_chunk_count,
                            )
                            await websocket.send_json({
                                "type": "audio_received",
                                "chunks": audio_chunk_count,
                            })
            except WebSocketDisconnect:
                LOGGER.info("client_disconnected")
            except Exception as e:
                LOGGER.error("client_receive_error", error=str(e))

        # Receive Gemini audio/tool-calls and send back to client
        async def receive_from_gemini():
            from app.agent.tools.registry import get_tool_by_name
            try:
                while True:
                    msg = await gemini_ws.recv()
                    LOGGER.info("gemini_frame_received", type=type(msg).__name__, preview=str(msg)[:500])
                    decoded = _decode_gemini_message(msg)

                    if decoded is None:
                        continue

                    if "binary" in decoded:
                        LOGGER.info("gemini_binary_frame_forwarded", size=len(decoded["binary"]))
                        await websocket.send_bytes(decoded["binary"])
                        continue

                    data = decoded

                    if "serverContent" in data:
                        server_content = data["serverContent"]
                        LOGGER.info(
                            "gemini_server_content_received",
                            has_model_turn="modelTurn" in server_content,
                            has_input_transcription="inputTranscription" in server_content,
                            has_output_transcription="outputTranscription" in server_content,
                            turn_complete=server_content.get("turnComplete"),
                        )

                        # Handle Audio parts and text parts
                        if "modelTurn" in server_content:
                            for part in server_content["modelTurn"].get("parts", []):
                                if "text" in part and part["text"]:
                                    await websocket.send_json({
                                        "type": "ai_text",
                                        "text": part["text"],
                                    })
                                if "inlineData" in part:
                                    pcm_base64 = part["inlineData"].get("data", "")
                                    if pcm_base64:
                                        pcm_bytes = base64.b64decode(pcm_base64)
                                        await websocket.send_bytes(pcm_bytes)

                        if "outputTranscription" in server_content:
                            out_text = server_content["outputTranscription"].get("text", "")
                            if out_text:
                                await websocket.send_json({
                                    "type": "ai_transcript",
                                    "text": out_text,
                                })

                        if "inputTranscription" in server_content:
                            in_text = server_content["inputTranscription"].get("text", "")
                            if in_text:
                                await websocket.send_json({
                                    "type": "user_transcript",
                                    "text": in_text,
                                })

                        if server_content.get("turnComplete"):
                            await websocket.send_json({
                                "type": "turn_complete",
                            })

                    # Handle tool calls
                    if "toolCall" in data:
                        tool_call = data["toolCall"]
                        fn_calls = tool_call.get("functionCalls", [])
                        tool_responses = []
                        for fn in fn_calls:
                            fn_name = fn.get("name")
                            fn_args = fn.get("args", {})
                            fn_id = fn.get("id", fn_name)
                            LOGGER.info("voice_tool_call", tool=fn_name, args=fn_args)

                            start_label, complete_label = _get_tool_status_label(fn_name, fn_args)
                            # Notify client that tool execution has started
                            await websocket.send_json({
                                "type": "tool_start",
                                "tool": fn_name,
                                "label": start_label,
                            })

                            try:
                                tool = get_tool_by_name(fn_name)
                                if tool:
                                    # Inject context that tools normally get from state
                                    fn_args.setdefault("user_id", user_id)
                                    fn_args.setdefault("store_id", store_id)
                                    result = await tool.ainvoke(
                                        fn_args,
                                        config={"configurable": {"user_id": user_id, "store_id": store_id}},
                                    )
                                    if isinstance(result, (dict, list)):
                                        serialized_result = json.dumps(result, ensure_ascii=False)
                                    else:
                                        serialized_result = str(result)
                                else:
                                    serialized_result = f"Tool '{fn_name}' not found."
                            except Exception as exc:
                                LOGGER.error("voice_tool_error", tool=fn_name, error=str(exc), exc_info=True)
                                serialized_result = f"Error executing {fn_name}: {exc}"

                            # Notify client that tool execution is complete
                            await websocket.send_json({
                                "type": "tool_complete",
                                "tool": fn_name,
                                "label": complete_label,
                            })

                            tool_responses.append({
                                "id": fn_id,
                                "name": fn_name,
                                "response": {"output": serialized_result},
                            })

                        # Send all tool results back to Gemini
                        response_msg = {
                            "toolResponse": {
                                "functionResponses": tool_responses
                            }
                        }
                        await gemini_ws.send(json.dumps(response_msg))

            except websockets.exceptions.ConnectionClosed:
                LOGGER.info("gemini_disconnected")
            except asyncio.CancelledError:
                LOGGER.info("gemini_receive_cancelled")
            except Exception as e:
                LOGGER.error("gemini_receive_error", error=str(e), exc_info=True)

        client_task = asyncio.create_task(receive_from_client())
        gemini_task = asyncio.create_task(receive_from_gemini())

        done, pending = await asyncio.wait(
            {client_task, gemini_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

        for task in done:
            exc = task.exception()
            if exc is not None:
                LOGGER.error("voice_loop_task_error", error=str(exc), exc_info=True)

    except Exception as e:
        LOGGER.error("voice_websocket_error", error=str(e), exc_info=True)
    finally:
        if gemini_ws is not None:
            try:
                await gemini_ws.close()
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass
