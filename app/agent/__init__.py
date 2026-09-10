"""
Vyapar Copilot — Agentic Core
==============================
Goal-driven, looping LangGraph agent for Vyapar Copilot.

Package layout
--------------
state.py        — VyaparAgentState TypedDict (the single source of truth)
memory.py       — mem0 user + store memory helpers
checkpointer.py — MongoDB LangGraph checkpoint (cross-session persistence)
tools/          — read-only tool implementations
graph.py        — LangGraph graph assembly (nodes, edges, loop logic)
"""
