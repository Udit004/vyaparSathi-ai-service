import json

from app.routes.voice_routes import (
    _build_gemini_audio_message,
    _build_gemini_audio_stream_end_message,
    _build_gemini_activity_start_message,
    _decode_gemini_message,
    _pcm_rms,
)


def test_decode_gemini_json_message():
    payload = json.dumps({"serverContent": {"modelTurn": {"parts": []}}}).encode("utf-8")
    assert _decode_gemini_message(payload) == {"serverContent": {"modelTurn": {"parts": []}}}


def test_decode_gemini_binary_audio_message():
    payload = b"\x00\x01\x02\x03"
    decoded = _decode_gemini_message(payload)
    assert decoded is not None and "binary" in decoded
    assert decoded["binary"] == payload


def test_build_gemini_realtime_audio_message_uses_live_api_shape():
    payload = b"\x00\x01\x02\x03"

    assert _build_gemini_audio_message(payload) == {
        "realtimeInput": {
            "audio": {
                "mimeType": "audio/pcm;rate=16000",
                "data": "AAECAw==",
            }
        }
    }


def test_build_gemini_audio_stream_end_message_flushes_paused_audio():
    assert _build_gemini_audio_stream_end_message() == {
        "realtimeInput": {"activityEnd": {}}
    }


def test_build_gemini_activity_start_message():
    assert _build_gemini_activity_start_message() == {
        "realtimeInput": {"activityStart": {}}
    }


def test_pcm_rms_detects_audio_signal():
    assert _pcm_rms(b"\x00\x00\xe8\x03\x18\xfc") > 0
