"""
tests/test_gemini_keys.py
==========================
Unit tests for Gemini API key rotation and rate-limit handling.
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch

from app.lib.gemini_keys import (
    get_gemini_api_keys,
    get_next_gemini_key,
    mark_key_rate_limited,
    RotatingGoogleGenerativeAIEmbeddings,
)


def test_get_gemini_api_keys_comma_separated():
    with patch.dict(os.environ, {"GEMINI_API_KEYS": "key_a, key_b, key_c"}, clear=True), \
         patch("app.lib.gemini_keys.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = None
        keys = get_gemini_api_keys()
        assert keys == ["key_a", "key_b", "key_c"]


def test_get_gemini_api_keys_numbered():
    env = {
        "GEMINI_API_KEY": "key_1",
        "GEMINI_API_KEY_2": "key_2",
        "GEMINI_API_KEY_3": "key_3",
    }
    with patch.dict(os.environ, env, clear=True), \
         patch("app.lib.gemini_keys.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = None
        keys = get_gemini_api_keys()
        assert "key_1" in keys
        assert "key_2" in keys
        assert "key_3" in keys


def test_key_rotation_skips_rate_limited():
    with patch.dict(os.environ, {"GEMINI_API_KEYS": "k1, k2"}, clear=True), \
         patch("app.lib.gemini_keys.get_settings") as mock_settings:
        mock_settings.return_value.gemini_api_key = None
        k_first = get_next_gemini_key()
        mark_key_rate_limited(k_first, cooldown_seconds=60)

        # Next key should be k2 because k_first is on cooldown
        k_next = get_next_gemini_key()
        assert k_next != k_first
        from app.lib.gemini_keys import _cooldowns
        _cooldowns.clear()
