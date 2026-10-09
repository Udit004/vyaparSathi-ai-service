"""
app/agent/tools/subgraphs/invoke_email_composer.py
=================================================
Tool stub that signals the subgraph_router to execute the Email Composer Subgraph.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool


class InvokeEmailComposerInput(BaseModel):
    request_prompt: str = Field(
        ...,
        description="Detailed description or context of what email to draft (e.g. 'Draft a formal Purchase Order for 50 crates of Amul Milk from Global Traders', 'Compose payment reminder for customer Rahul due ₹5,000').",
    )
    category: Optional[Literal[
        "purchase_order",
        "payment_reminder",
        "quotation_inquiry",
        "restock_alert",
        "customer_announcement",
        "general_correspondence",
    ]] = Field(
        None,
        description="Optional category of email if known.",
    )
    recipient_name: Optional[str] = Field(
        None,
        description="Name of the seller, distributor, customer, or recipient (e.g. 'Global Traders', 'Ramesh Tiwari', 'Rahul').",
    )
    recipient_email: Optional[str] = Field(
        None,
        description="Direct email address of recipient if provided.",
    )
    language: Optional[Literal["English", "Hindi", "Hinglish"]] = Field(
        "English",
        description="Language for the email: 'English', 'Hindi', or 'Hinglish'.",
    )


@tool("invoke_email_composer", args_schema=InvokeEmailComposerInput)
async def invoke_email_composer(
    request_prompt: str,
    category: Optional[str] = None,
    recipient_name: Optional[str] = None,
    recipient_email: Optional[str] = None,
    language: Optional[str] = "English",
) -> dict:
    """
    [SUBGRAPH TOOL] Draft a high-quality, professional, beautifully styled HTML & plain-text business email
    using the multi-tier Email Composer Agent (SLM classifier + LLM structured composer + validation loop).

    Supports:
    - Purchase Orders (itemized tables, PO IDs, quantity/rates, delivery instructions)
    - Payment Reminders (due amounts, payment links/UPI, invoice references)
    - Quotation Inquiries (item specifications, bulk price inquiries)
    - Restock Alerts (urgent product requirements, timelines)
    - Customer Announcements / Offers (festive sales, new arrivals)
    - General Business Correspondence

    OUTPUT:
      success        : bool   — True if draft succeeded quality validation
      category       : str    — Email category
      recipient_name : str    — Resolved recipient name
      recipient_email: str    — Resolved recipient email
      subject        : str    — Professional subject line
      preview_text   : str    — Inbox preview snippet
      html_content   : str    — Fully styled responsive HTML email
      plain_text     : str    — Plain-text fallback email
      summary_cards  : list   — Key highlight metric cards
      draft_spec     : dict   — Raw structured draft spec
    """
    return {
        "__subgraph__": "email_composer",
        "request_prompt": request_prompt,
        "category": category,
        "recipient_name": recipient_name,
        "recipient_email": recipient_email,
        "language": language or "English",
    }
