"""
app/agent/memory/__init__.py
============================
Public API for the Vyapar Sathi long-term memory package.

Exposes all symbols that the rest of the application imports from
`app.agent.memory` so that neither the graph nor external callers need
to know which internal module provides each function.
"""

# ---------------------------------------------------------------------------
# Write pipeline (new structured system)
# ---------------------------------------------------------------------------
from app.agent.memory.service import process_and_persist_memory

# ---------------------------------------------------------------------------
# Read / retrieval pipeline (from the original retriever.py module)
# ---------------------------------------------------------------------------
from app.agent.memory.retriever import (
    # Public read helpers
    load_memory_context,
    should_retrieve_memory,
    build_memory_query,
    summarize_memory_results,

    # Per-level search
    search_user_memory,
    search_store_memory,
    search_multi_store_memory,

    # Per-level write helpers (legacy path — kept for backward compat)
    add_user_memory,
    add_store_memory,
    add_multi_store_memory,
    curate_memory_messages,
    extract_multi_level_memory,

    # Memory level constants
    TYPE_USER_PREFERENCE,
    TYPE_STORE_MEMORY,
    TYPE_MULTI_STORE_MEMORY,

    # Diagnostics / health
    get_memory_client,
    is_enabled,
    get_memory_status,

    # Private helpers re-exported so tests/nodes can call them
    _query_memory_vectors,
    _cap_results,
    _filter_relevant_results,
    _reconcile_and_update_memory,
    _upsert_memory_vector,
    _delete_memory_vector,
)

__all__ = [
    # Write
    "process_and_persist_memory",
    # Read
    "load_memory_context",
    "should_retrieve_memory",
    "build_memory_query",
    "summarize_memory_results",
    "search_user_memory",
    "search_store_memory",
    "search_multi_store_memory",
    # Legacy write
    "add_user_memory",
    "add_store_memory",
    "add_multi_store_memory",
    "curate_memory_messages",
    "extract_multi_level_memory",
    # Constants
    "TYPE_USER_PREFERENCE",
    "TYPE_STORE_MEMORY",
    "TYPE_MULTI_STORE_MEMORY",
    # Diagnostics
    "get_memory_client",
    "is_enabled",
    "get_memory_status",
    # Low-level helpers
    "_query_memory_vectors",
    "_cap_results",
    "_filter_relevant_results",
    "_reconcile_and_update_memory",
    "_upsert_memory_vector",
    "_delete_memory_vector",
]
