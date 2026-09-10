"""
Vyapar Copilot — Agentic Core
==============================
Goal-driven, looping LangGraph agent for Vyapar Copilot.

Package layout
--------------
state.py        — VyaparAgentState TypedDict + factories
checkpointer.py — AsyncMongoDBSaver singleton (short-term memory)
memory.py       — mem0 user + store memory helpers (long-term)
utils.py        — SSE streaming helpers and text formatting
graph.py        — LangGraph graph assembly — build_graph()
nodes/          — Individual node implementations (think, tool, observe, respond)
tools/          — Read-only tool implementations grouped by domain

Public API
----------
>>> from app.agent import build_graph, get_checkpointer, make_initial_state
"""

from app.agent.state import VyaparAgentState, make_initial_state
from app.agent.graph import build_graph
from app.agent.checkpointer import get_checkpointer, close_checkpointer

__all__ = [
    "VyaparAgentState",
    "make_initial_state",
    "build_graph",
    "get_checkpointer",
    "close_checkpointer",
]
