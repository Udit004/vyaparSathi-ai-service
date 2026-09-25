"""
Vyapar Copilot — Agentic Core
==============================
Goal-driven, looping LangGraph agent for Vyapar Copilot.

Package layout
--------------
state.py        — VyaparAgentState TypedDict + factories
checkpointer.py — MongoDBSaver singleton (short-term memory)
memory.py       — Pinecone multi-level user, store, & multi-store memory helpers
utils.py        — SSE streaming helpers and text formatting
graph.py        — LangGraph graph assembly — build_graph()
nodes/          — Individual node implementations:
                      grader        — guardrail (safety check), runs first
                      think         — LLM reasoning + tool selection
                      memory_query  — conditionally fetches Pinecone vector context
                      tool          — executes tool calls
                      observe       — converts results to messages, increments loop
tools/          — Read-only tool implementations grouped by domain

Public API
----------
>>> from app.agent import build_graph, get_checkpointer, make_initial_state
"""

from app.agent.state import VyaparAgentState, make_initial_state
from app.agent.graph import build_graph
from app.agent.checkpointer import (
    get_checkpointer,
    close_checkpointer,
)
from app.agent.memory import (
    get_memory_client,
    is_enabled,
    add_user_memory,
    search_user_memory,
    add_store_memory,
    search_store_memory,
    add_multi_store_memory,
    search_multi_store_memory,
    load_memory_context,
)

__all__ = [
    "VyaparAgentState",
    "make_initial_state",
    "build_graph",
    "get_checkpointer",
    "close_checkpointer",
    "get_memory_client",
    "is_enabled",
    "add_user_memory",
    "search_user_memory",
    "add_store_memory",
    "search_store_memory",
    "add_multi_store_memory",
    "search_multi_store_memory",
    "load_memory_context",
]