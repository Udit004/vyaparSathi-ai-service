"""
app/agent/prompts/summarizer_prompts/__init__.py
=================================================
Instructions for the small summarizer LLM (GROQ/NVIDIA).

These prompts are used to compress large payloads — mem0 memory data,
tool responses, and sliding-window history — before they enter the
main LLM's context window.
"""

from app.agent.prompts.summarizer_prompts.memory import (
    MEMORY_EXTRACTION_INSTRUCTION,
    MEMORY_QUERY_INSTRUCTION,
    SUMMARIZER_MEMORY_INSTRUCTION,
    MULTI_LEVEL_MEMORY_EXTRACTION_INSTRUCTION,
    MEMORY_RECONCILIATION_INSTRUCTION,
)
from app.agent.prompts.summarizer_prompts.tool import build_summarizer_tool_instruction
from app.agent.prompts.summarizer_prompts.history import SUMMARIZER_HISTORY_INSTRUCTION
from app.agent.prompts.summarizer_prompts.title import TITLE_GENERATION_INSTRUCTION

__all__ = [
    "SUMMARIZER_MEMORY_INSTRUCTION",
    "MEMORY_QUERY_INSTRUCTION",
    "MEMORY_EXTRACTION_INSTRUCTION",
    "MULTI_LEVEL_MEMORY_EXTRACTION_INSTRUCTION",
    "MEMORY_RECONCILIATION_INSTRUCTION",
    "build_summarizer_tool_instruction",
    "SUMMARIZER_HISTORY_INSTRUCTION",
    "TITLE_GENERATION_INSTRUCTION",
]