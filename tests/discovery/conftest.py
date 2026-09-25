"""
tests/discovery/conftest.py
============================
Mocks out motor/pymongo before any test imports to allow
testing pure Python modules (like discovery/filters.py) without
a live MongoDB connection.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

try:
    import motor
    import pymongo
except ImportError:
    for mod_name in [
        "motor",
        "motor.motor_asyncio",
        "pymongo",
        "pymongo.cursor",
        "pymongo.errors",
    ]:
        if mod_name not in sys.modules:
            mock_mod = MagicMock()
            mock_mod.__path__ = []
            sys.modules[mod_name] = mock_mod

# Stub app.config.database so import chain doesn't break
db_mock = MagicMock()
sys.modules.setdefault("app.config.database", db_mock)
