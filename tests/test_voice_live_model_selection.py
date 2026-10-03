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


def test_live_tools_contain_no_unsupported_schema_keys():
    """Verify Gemini Live functionDeclarations contain no illegal keys rejected by websocket proto."""
    declarations = _get_gemini_tools()[0]["functionDeclarations"]

    def _assert_clean_schema(schema_dict, path=""):
        if isinstance(schema_dict, dict):
            for k, v in schema_dict.items():
                current_path = f"{path}.{k}" if path else k
                assert k not in (
                    "additionalProperties", "title", "$defs", "$ref", "default",
                    "examples", "prefixItems"
                ), f"Found illegal schema keyword '{k}' at {current_path}"
                if k == "properties" and isinstance(v, dict):
                    for prop_name, prop_schema in v.items():
                        _assert_clean_schema(prop_schema, f"{current_path}.{prop_name}")
                elif k == "items":
                    _assert_clean_schema(v, current_path)
                elif isinstance(v, (dict, list)) and k != "properties":
                    _assert_clean_schema(v, current_path)
        elif isinstance(schema_dict, list):
            for idx, item in enumerate(schema_dict):
                _assert_clean_schema(item, f"{path}[{idx}]")

    for decl in declarations:
        _assert_clean_schema(decl["parameters"], f"{decl['name']}.parameters")


def test_voice_prompt_contains_exact_session_ids_for_tool_calls():
    prompt = _build_voice_system_prompt("user-123", "store-456", "memory")

    assert "user_id=user-123" in prompt
    assert "store_id=store-456" in prompt
    assert "use exactly this store_id" in prompt

