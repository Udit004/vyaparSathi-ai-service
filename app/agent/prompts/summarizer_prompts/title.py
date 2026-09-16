"""
app/agent/prompts/summarizer_prompts/title.py
=============================================
Instruction for generating a short chat title from a user's first message.

Used by ``app/services/chat_history_service.py`` to label a chat session
using the small/fast summarizer (GROQ openai/gpt-oss-20b / NVIDIA
nvidia/llama-3.1-8b-instruct) instead of the main Gemini model, so Gemini
quota is reserved for retail-agent reasoning.
"""

TITLE_GENERATION_INSTRUCTION = (
    "You generate short chat titles for a retail store AI assistant. Given "
    "the user's first message, produce a concise title (3-7 words) that "
    "captures the topic. Return ONLY the title, no quotes, no trailing "
    "punctuation, no explanation, no extra text."
)
