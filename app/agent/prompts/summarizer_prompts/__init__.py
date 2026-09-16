"""
app/agent/prompts/summarizer_prompts/__init__.py
=================================================
Instructions for the small summarizer LLM (GROQ/NVIDIA).

These prompts are used to compress large payloads — mem0 memory data,
tool responses, and sliding-window history — before they enter the
main LLM's context window.
"""

from app.agent.prompts.summarizer_prompts.memory import SUMMARIZER_MEMORY_INSTRUCTION
from app.agent.prompts.summarizer_prompts.tool import build_summarizer_tool_instruction
from app.agent.prompts.summarizer_prompts.history import SUMMARIZER_HISTORY_INSTRUCTION
from app.agent.prompts.summarizer_prompts.title import TITLE_GENERATION_INSTRUCTION

__all__ = [
    "SUMMARIZER_MEMORY_INSTRUCTION",
    "build_summarizer_tool_instruction",
    "SUMMARIZER_HISTORY_INSTRUCTION",
    "TITLE_GENERATION_INSTRUCTION",
]