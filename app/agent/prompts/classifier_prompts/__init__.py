"""
app/agent/prompts/classifier_prompts/__init__.py
================================================
Instructions for the lightweight harm-classifier LLM (GROQ / NVIDIA).

These prompts steer a small/fast model that the guardrail node uses to decide
whether a user's message should be refused before the main agent engages.
"""

from app.agent.prompts.classifier_prompts.harm_check import CLASSIFIER_HARM_INSTRUCTION

__all__ = ["CLASSIFIER_HARM_INSTRUCTION"]
