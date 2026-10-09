"""
app/agent/subgraphs/email_composer/__init__.py
==============================================
Exports Email Composer Subgraph and schemas.
"""

from app.agent.subgraphs.email_composer.graph import email_composer_graph
from app.agent.subgraphs.email_composer.schemas import (
    EmailCategory,
    EmailLanguage,
    EmailClassification,
    EmailDraftSpec,
)
from app.agent.subgraphs.email_composer.templates import (
    render_html_email,
    render_plain_text,
)

__all__ = [
    "email_composer_graph",
    "EmailCategory",
    "EmailLanguage",
    "EmailClassification",
    "EmailDraftSpec",
    "render_html_email",
    "render_plain_text",
]
