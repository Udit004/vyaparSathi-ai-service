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


def _get_gemini_tools():
    gemini_tools = []
    for tool in VYAPAR_TOOLS:
        args_schema = getattr(tool, "args_schema", None)
        parameters = args_schema.model_json_schema() if args_schema else {
            "type": "object",
            "properties": {},
        }
        parameters.pop("title", None)
        parameters.pop("$defs", None)
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


async def _bootstrap_voice_memory_context(user_id: str, store_id: str):
    """Load voice memory safely. Pinecone failures should degrade to an empty context."""
    try:
        user_memories = await _bootstrap_user_memory(user_id, "user preferences and details")
    except Exception as exc:
        LOGGER.warning(
            "voice_user_memory_bootstrap_failed",
            user_id=user_id,
            store_id=store_id,
            error=str(exc),
        )
        user_memories = []

    try:
        store_memories = await _bootstrap_store_memory(store_id, "store details and facts", intent="general")
    except Exception as exc:
        LOGGER.warning(
            "voice_store_memory_bootstrap_failed",
            user_id=user_id,
            store_id=store_id,
            error=str(exc),
        )
        store_memories = []

    return user_memories, store_memories


def _build_voice_system_prompt(user_id: str, store_id: str, memory_context: str):
    """Build the voice prompt with immutable tenant context for tool calls."""
    return (
        "You are Vyapar Sathi, a helpful business assistant. "
        "Keep responses short, natural, and conversational for voice. "
        "Respond only in English or Hindi. Match the user's language when it is clear; "
        "if the language is unclear, use simple English. Do not respond in any other language. "
        "The active session context is immutable: "
        f"user_id={user_id}; store_id={store_id}. "
        "For every tool call that accepts store_id, use exactly this store_id. "
        "For every tool call that accepts user_id, use exactly this user_id. "
        "Never invent, omit, or ask the user for these IDs. "
        f"{memory_context}"
    )


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

                        # Handle Audio parts
                        if "modelTurn" in server_content:
                            for part in server_content["modelTurn"].get("parts", []):
                                if "inlineData" in part:
                                    pcm_base64 = part["inlineData"].get("data", "")
                                    if pcm_base64:
                                        pcm_bytes = base64.b64decode(pcm_base64)
                                        await websocket.send_bytes(pcm_bytes)

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
                            try:
                                tool = get_tool_by_name(fn_name)
                                if tool:
                                    # Inject context that tools normally get from state
                                    fn_args.setdefault("user_id", user_id)
                                    fn_args.setdefault("store_id", store_id)
                                    result = await tool.ainvoke(fn_args)
                                else:
                                    result = f"Tool '{fn_name}' not found."
                            except Exception as exc:
                                LOGGER.error("voice_tool_error", tool=fn_name, error=str(exc), exc_info=True)
                                result = f"Error executing {fn_name}: {exc}"

                            tool_responses.append({
                                "id": fn_id,
                                "name": fn_name,
                                "response": {"output": str(result)},
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
