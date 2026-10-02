from app.routes.voice_routes import (
    _build_voice_system_prompt,
    _get_gemini_tools,
    _get_live_model_candidates,
)


def test_live_model_candidates_prefer_current_google_live_model():
    candidates = _get_live_model_candidates()

    assert candidates[0] == "models/gemini-3.8-live"
    assert "models/gemini-2.5-flash-native-audio-preview-12-2025" in candidates
    assert "models/gemini-2.0-flash-live-001" not in candidates


def test_live_tools_include_argument_schemas():
    declarations = _get_gemini_tools()[0]["functionDeclarations"]

    assert declarations
    assert all(declaration["parameters"]["type"] == "object" for declaration in declarations)


def test_voice_prompt_contains_exact_session_ids_for_tool_calls():
    prompt = _build_voice_system_prompt("user-123", "store-456", "memory")

    assert "user_id=user-123" in prompt
    assert "store_id=store-456" in prompt
    assert "use exactly this store_id" in prompt
