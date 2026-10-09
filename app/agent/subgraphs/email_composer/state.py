"""
app/agent/subgraphs/email_composer/state.py
============================================
State definition for the Email Composer Subgraph.
"""

from typing import Any, Literal, Optional
from typing_extensions import TypedDict


class EmailComposerState(TypedDict, total=False):
    """
    Isolated state for the Email Composer Subgraph.
    Carries the context from intent classification, template specification,
    HTML/text rendering, validation, and bounded repair loop.
    """
    # Scope & Request context
    user_id: str
    store_id: str
    request_prompt: str
    recipient_name: Optional[str]
    recipient_email: Optional[str]
    category: Optional[str]
    language: Optional[str]

    # Store & Merchant context
    merchant_context: dict[str, Any]
    business_context: dict[str, Any]

    # Classification output
    classification: dict[str, Any]

    # Specification & Rendering outputs
    draft_spec: dict[str, Any]
    rendered_html: str
    rendered_text: str

    # Validation & Bounded Repair
    validation_result: dict[str, Any]
    validation_errors: list[str]
    repair_feedback: str
    retry_count: int
    max_retries: int

    # Final Subgraph Output
    status: Literal["pending", "in_progress", "success", "failed"]
    error: Optional[str]
    email_output: dict[str, Any]
