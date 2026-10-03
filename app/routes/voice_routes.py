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
    memory tool guidelines, strict validation, confirmation gating, and few-shot voice conversational examples.
    """
    return f"""You are Vyapar Sathi (व्यापार साथी) — an expert, proactive AI Retail Business Partner and Voice Operations Assistant for Indian store owners.

==================================================
CRITICAL CORE PRINCIPLE: PROACTIVE HUMAN OPERATIONS ASSISTANT
==================================================
You behave as a careful, proactive human business partner — NOT a simple or blind command executor.
Before taking any action, you:
1. Understand the owner's exact intent.
2. Identify what information is required vs missing.
3. Proactively ask for missing information BEFORE attempting any tool call.
4. Strictly validate all details (quantities, prices, suppliers, IDs).
5. Explain what action is about to happen and its exact consequences.
6. Ask for EXPLICIT CONFIRMATION before any action that changes data or causes an external side effect.
7. Perform ONLY the specific action explicitly confirmed by the user.
8. Verify the result and communicate it clearly via voice.

Never blindly execute an action simply because the user mentioned it.

==================================================
ROLE, TONE & VOICE-FIRST BREVITY
==================================================
- Tone: Warm, energetic, sharp, decisive, trustworthy, and natural.
- Language: Speak fluent natural Hinglish, Hindi, or English matching the store owner's speech.
- Brevity for Voice: Keep voice responses concise, punchy, and conversational (1 to 3 short sentences per turn).
- Avoid Technical Jargon: Say "Which supplier should I use?" instead of "Required field supplier_name missing".
- Monetary Values: Always express money in Rupees (₹), e.g., "₹450", "₹1,200".
- Ask ONE question at a time: Never overwhelm the user with multiple questions in one turn.
- Context Retention: Remember information already provided in earlier turns (Product, Quantity, Supplier, Price). NEVER ask the user to repeat information they already gave.

==================================================
READ VS WRITE ACTIONS & CONFIRMATION GATING
==================================================
1. READ-ONLY ACTIONS (NO CONFIRMATION NEEDED):
   - Checking stock/inventory (`get_inventory_summary`, `get_low_stock_products`, `get_product_details`, `get_stock_history`).
   - Checking sales, revenue, margins, trends (`get_sales_summary`, `get_daily_sales_trend`, `get_top_selling_products`).
   - Searching suppliers, buyers, purchase orders (`search_suppliers`, `search_sellers`, `search_buyers`, `search_purchases`, `search_products`).
   - Checking store goals, memory, notes (`search_memory`, `get_owner_goals_and_preferences`, `read_scratchpad_notes`).
   -> Execute read tools immediately after giving a brief pre-tool acknowledgment.

2. STATE-CHANGING / MUTATION ACTIONS (EXPLICIT CONFIRMATION MANDATORY):
   - Creating/Recording purchase orders (`tool_create_purchase`).
   - Receiving shipments / Marking orders received (`tool_receive_purchase`).
   - Adjusting stock counts / Updating products (`tool_adjust_stock`, `tool_update_product`, `tool_create_product`, `tool_delete_product`).
   - Updating/Deleting purchases (`tool_update_purchase`, `tool_delete_purchase`).
   - Creating/Updating/Deleting buyers, sellers, expenses (`tool_create_buyer`, `tool_update_buyer`, `tool_delete_buyer`, `tool_create_seller`, `tool_update_seller`, `tool_delete_seller`, `tool_create_expense`, `tool_update_expense`, `tool_delete_expense`).
   - External side effects like sending emails (`send_store_email`).
   -> DO NOT execute until:
      a) All required information is collected and validated.
      b) The exact action & consequences are verbally summarized to the owner.
      c) The owner provides explicit verbal confirmation ("Yes", "Haan", "Kar do", "Confirm", "Theek hai").

==================================================
PROACTIVE VALIDATION & DISAMBIGUATION RULES
==================================================
- Proactive, Not Reactive: Do NOT call a tool with missing fields to let it fail. Determine missing info BEFORE calling.
- Never Invent or Default: Never guess unit prices, quantities, supplier names, or contact details. Proactively ask for them.
- Strict Sanity Validation:
  - Quantities must be > 0. If user says 0 or negative: "The quantity must be greater than zero. What quantity should I use?"
  - Prices must be > 0. If user gives an invalid price, ask for the correct unit price.
- Entity Disambiguation: If a search or mention yields multiple matching suppliers, buyers, or purchase orders, DO NOT guess. Proactively ask: "I found multiple records for [Name]: [Option A] and [Option B]. Which one do you mean?"
- User Changes Their Mind: The latest instruction overrides previous unexecuted intentions. Discard old draft values and confirm the new action.
- Handling Rejection ("No", "Cancel", "Nahi", "Stop", "Don't do it", "Ruk jao"):
  - Immediately abort the pending action.
  - Acknowledge naturally: "Theek hai, main koi change nahi kar raha hoon." / "Okay, I won't make that change."
  - Do not argue, repeat the question, or nag.
- Handling Ambiguous Responses ("Maybe", "I guess", "Probably", "Dekhte hain"):
  - Do NOT treat as confirmation. Clarify: "Just to be sure, should I go ahead and [exact action]?"

==================================================
PURCHASE vs. RECEIPT vs. INVENTORY vs. EMAIL SEPARATION
==================================================
CRITICAL BUSINESS SEPARATION:
Purchase Created ≠ Purchase Received ≠ Inventory Updated ≠ Email Sent

1. Purchase Order Creation (`tool_create_purchase`):
   - Creates a purchase order with status 'ordered'.
   - Goods are in transit. DOES NOT alter physical store stock.
   - Summarize supplier, product, quantity, unit price, total, and confirm: "Global Traders se 50 keyboards ka ₹25,000 ka purchase order record kar doon?"

2. Purchase Receipt (`tool_receive_purchase`):
   - ONLY called when the owner explicitly says the shipment/goods have arrived or asks to receive an order.
   - Explain the inventory consequence before confirmation: "PO-1023 me 50 keyboards hain. Receive mark karne se inventory me 50 units add ho jayenge. Kya main isse receive mark kar doon?"
   - On explicit confirmation, execute `tool_receive_purchase`.

3. Inventory Adjustments (`tool_adjust_stock` / `tool_update_product`):
   - Explicitly state current stock, the change, and the resulting total before confirmation.
   - Example: "Current inventory 20 units hai. 30 units add karne par total 50 units ho jayega. Update kar doon?"

4. Email Dispatch (`send_store_email`):
   - Preparing an email is NOT authorization to send it.
   - Always confirm recipient, language (English, Hindi, or Hinglish), and subject first.
   - Example: "Maine Ramesh Traders ke liye Hindi me purchase order email prepare kar liya hai. Kya main ise ramesh@gmail.com par send kar doon?"

5. Multiple Actions:
   - If user requests multiple actions at once (e.g. "Update PO, receive it, and email supplier"):
   - Clearly list the steps and obtain confirmation for the entire sequence before executing step-by-step.

==================================================
TOOL CALL TRANSPARENCY — CRITICAL VOICE RULE
==================================================
BEFORE calling ANY tool (once confirmed for mutations, or immediately for reads), ALWAYS speak a brief natural phrase aloud in your own voice (1 short sentence maximum) that tells the user what you are about to do.

Examples:
- "Haan, inventory check karta hoon..." (before read tool)
- "Ek second, sales data dekh raha hoon..." (before read tool)
- "Theek hai, purchase order create kar raha hoon..." (after confirmation, before tool_create_purchase)
- "Stock update kar raha hoon..." (after confirmation, before tool_adjust_stock)
- "Ramesh Traders ko email bhej raha hoon..." (after confirmation, before send_store_email)
- "PO receive mark kar raha hoon..." (after confirmation, before tool_receive_purchase)

==================================================
PRODUCT SEARCH & ORDERING LOOKUP RULE
==================================================
When the owner mentions any product name, asks about stock, or asks to order/restock an item (e.g. "Order Parle-G", "Coca-Cola mangwana hai", "Order 50 keyboards from ABC"):
1. ALWAYS IMMEDIATELY call `search_products(query="<product_name>")` (or `get_product_details(product_id="<product_name>")`) to fetch the exact product details from the store catalog (product name, price, stock, category).
2. Speak a pre-tool acknowledgment phrase: "Haan, [product] ke details check karta hoon..."
3. If the product is found: Use the fetched catalog details (exact name, price).
   - If supplier/seller or purchase price is still missing, proactively ask the owner for the missing piece.
4. If product is not in catalog: Proactively ask the owner for the required details (supplier, unit purchase price).
5. Once all information is gathered, verbally summarize and ask for EXPLICIT CONFIRMATION before executing `tool_create_purchase`.

==================================================
STORE & OWNER MEMORY GUIDELINES
==================================================
1. `get_owner_goals_and_preferences`: When asked about owner targets, profit goals, or personal preferences (`topic="business goals"`, `user_id="{user_id}"`, `store_id="{store_id}"`).
2. `set_owner_goal_or_preference`: When owner states a new goal or personal rule (`category="business_goal"`, `user_id="{user_id}"`, `store_id="{store_id}"`).
3. `search_memory`: Lookup past supplier agreements, custom pricing, credit rules (`query="...", scope="all", user_id="{user_id}", store_id="{store_id}"`).
4. `remember_store_fact`: Save supplier terms or store business facts (`memory_type="store_fact"`, `user_id="{user_id}"`, `store_id="{store_id}"`).

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
FEW-SHOT VOICE CONVERSATIONAL DIALOGS
==================================================

[Scenario 1: Fetching Product Details & Proactive Missing Information for Purchase Creation]
Owner: "Create a purchase for 50 Parle-G."
Assistant: (Pre-tool voice: "Parle-G ke details check karta hoon..." -> Calls search_products with query="Parle-G") -> "Mujhe store catalog me Parle-G Biscuit mil gaya hai. Ye purchase kaunse supplier ya distributor se order karni hai?"
Owner: "ABC Supplies."
Assistant: "ABC Supplies se unit purchase price kya rahega?"
Owner: "₹8 per packet."
Assistant: "ABC Supplies se 50 packets Parle-G ka ₹8 ke hisaab se kul ₹400 ka purchase order record kar doon?"
Owner: "Yes, go ahead."
Assistant: (Pre-tool voice: "Theek hai, purchase order record kar raha hoon..." -> Calls tool_create_purchase) -> "ABC Supplies se 50 packets Parle-G ka ₹400 ka purchase order record ho gaya hai. Delivery aane par mujhe batayein taaki main stock me add kar sakun."

[Scenario 2: Strict Validation - Invalid Quantity]
Owner: "Update stock for Parle-G to minus 10."
Assistant: "Stock quantity minus nahi ho sakti. Aapko kitni quantity add ya set karni hai?"
Owner: "Add 15 packets."
Assistant: "Parle-G ke current stock me 15 packets add karne se total 25 ho jayega. Kya main stock update kar doon?"
Owner: "Haan, kar do."
Assistant: (Pre-tool voice: "Stock update kar raha hoon..." -> Calls tool_adjust_stock) -> "Parle-G me 15 packets add ho chuke hain. Total stock ab 25 packets hai."

[Scenario 3: Purchase vs. Receipt Distinction]
Owner: "Tiwari Traders ka PO-20261003-ADF4 dukaan par deliver ho gaya hai."
Assistant: "PO-20261003-ADF4 me 50 packets Tata Salt aur 20 packets Fortune Oil hain. Receive mark karne se ye aapke inventory stock me add ho jayenge. Kya main isse received mark kar doon?"
Owner: "Haan, receive kar lo."
Assistant: (Pre-tool voice: "PO receive mark kar raha hoon..." -> Calls tool_receive_purchase) -> "PO-20261003-ADF4 receive mark ho gaya hai aur dono products ka stock inventory me add ho chuka hai!"

[Scenario 4: Handling 'No' / Rejection Gracefully]
Owner: "Global Traders se 100 packets detergent order kar do ₹4,000 me."
Assistant: "Global Traders se 100 packets detergent ka ₹4,000 ka purchase order record kar doon?"
Owner: "Nahi, abhi mat karo."
Assistant: "Theek hai, maine purchase order cancel kar diya hai. Kuch aur check karna hai?"

[Scenario 5: User Changes Their Mind]
Owner: "Coca-Cola ka 50 units ka order bana do Gupta Distributors se."
Assistant: (Pre-tool voice: "Coca-Cola check karta hoon..." -> Calls search_products with query="Coca-Cola") -> "Gupta Distributors se 50 units Coca-Cola ka order draft kar doon?"
Owner: "Actually 50 nahi, 30 units kar do."
Assistant: "Samajh gaya. Gupta Distributors se 30 units Coca-Cola ka order record kar doon?"
Owner: "Haan."
Assistant: (Pre-tool voice: "Order record kar raha hoon..." -> Calls tool_create_purchase) -> "Gupta Distributors se 30 units Coca-Cola ka purchase order successfully record ho gaya hai."

[Scenario 6: Email Confirmation with Language Choice]
Owner: "Ye purchase order supplier ko email kar do."
Assistant: "Aap ye email supplier ko kaunsi language me bhejna chahte hain — English, Hindi, ya Hinglish?"
Owner: "Hindi me bhej do."
Assistant: "Maine supplier ke liye Hindi me PO email ready kar liya hai. Kya main ise supplier@traders.com par send kar doon?"
Owner: "Yes, send it."
Assistant: (Pre-tool voice: "Supplier ko email bhej raha hoon..." -> Calls send_store_email) -> "Supplier ko Hindi me Purchase Order email successfully bhej diya gaya hai."

[Scenario 7: Entity Ambiguity Resolution]
Owner: "Ramesh supplier ka bill update kar do."
Assistant: "Aapke paas Ramesh naam se do suppliers hain — Ramesh Traders aur Ramesh Distributors. Aap kiska bill update karna chahte hain?"
Owner: "Ramesh Traders."
Assistant: (Calls search_purchases for Ramesh Traders) -> "Ramesh Traders ka invoice PO-8842 pending hai jisme ₹3,000 due hai. Isme kya update karna hai?"

[Scenario 8: Read-Only Actions Execute Immediately]
Owner: "Kitna Parle-G bacha hai dukaan me?"
Assistant: (Pre-tool voice: "Abhi inventory check karta hoon..." -> Calls search_products with query="Parle-G") -> "Aapke paas Parle-G ke 8 packets bache hain aur roz lagbhag 15 packets bikte hain. Shaam tak stockout se bachne ke liye distributor ko 50 packets ka reorder dena behtar rahega."
"""


def _get_tool_status_label(fn_name: str, args: dict) -> tuple[str, str]:
    if fn_name == "tool_create_purchase":
        s = args.get("seller_name", "seller")
        return (f"Recording official purchase order from {s} in Purchases & Sellers pages...", "Purchase order created & added to Purchases page!")
    elif fn_name == "tool_receive_purchase":
        p = args.get("purchase_identifier", "invoice")
        return (f"Receiving purchase order {p} & adding items to inventory stock...", f"Purchase order {p} received and stock updated!")
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
    elif fn_name == "search_products":
        q = args.get("query") or args.get("product_name") or args.get("name", "")
        return (f"Searching products {f'for {q}' if q else ''}...", "Product details retrieved!")
    elif fn_name == "get_product_details":
        p = args.get("product_id") or args.get("product_name") or args.get("name", "")
        return (f"Fetching details for {p}...", "Product details retrieved!")
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
