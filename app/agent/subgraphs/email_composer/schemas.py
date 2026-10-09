"""
app/agent/subgraphs/email_composer/schemas.py
=============================================
Pydantic schemas and structured output models for Email Composer Subgraph.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


EmailCategory = Literal[
    "purchase_order",
    "payment_reminder",
    "quotation_inquiry",
    "restock_alert",
    "customer_announcement",
    "inventory_report",
    "sales_summary",
    "general_correspondence",
]

EmailLanguage = Literal["English", "Hindi", "Hinglish"]


class EmailClassification(BaseModel):
    category: EmailCategory = Field(
        ...,
        description="The specific business category for the email.",
    )
    recipient_name: Optional[str] = Field(
        None,
        description="Name of the seller, distributor, customer, or recipient.",
    )
    recipient_email: Optional[str] = Field(
        None,
        description="Direct email address of recipient if mentioned.",
    )
    language: EmailLanguage = Field(
        default="English",
        description="Language to write the email in: English, Hindi, or Hinglish.",
    )
    tone: Literal["formal", "polite_urgent", "friendly_promotional", "professional"] = Field(
        default="professional",
        description="Tone of the email.",
    )
    key_intent: str = Field(
        ...,
        description="Short 1-sentence summary of what this email achieves.",
    )


class SummaryCard(BaseModel):
    label: str = Field(..., description="Label like 'Total Amount', 'Due Date', 'Order ID', 'Invoice #'.")
    value: str = Field(..., description="Value like '₹14,500', '15 Oct 2026', 'PO-9821'.")
    color: Optional[str] = Field(None, description="Accent color e.g. 'emerald', 'indigo', 'amber'.")


class ActionButton(BaseModel):
    text: str = Field(..., description="Button call-to-action text e.g. 'Confirm Order', 'Pay via UPI', 'View Catalog'.")
    url: Optional[str] = Field(None, description="Target URL if applicable.")


class SenderSignature(BaseModel):
    store_name: str = Field(..., description="Store business name.")
    owner_name: str = Field(..., description="Merchant / Owner full name.")
    email: str = Field(..., description="Store official email.")
    phone: Optional[str] = Field(None, description="Contact phone number.")
    address: Optional[str] = Field(None, description="Store city or address.")
    gstin: Optional[str] = Field(None, description="GST Number if applicable.")


class EmailDraftSpec(BaseModel):
    subject: str = Field(
        ...,
        description="Clear, professional subject line with relevant tags e.g. '[Purchase Order #PO-2026-081] - Fresh Dairy Restock from NexaStore'.",
    )
    preview_text: str = Field(
        ...,
        description="Short 1-sentence inbox preview snippet.",
    )
    badge_title: str = Field(
        default="OFFICIAL BUSINESS MESSAGE",
        description="Top badge text e.g. 'PURCHASE ORDER', 'PAYMENT REMINDER', 'QUOTATION REQUEST'.",
    )
    headline: str = Field(
        ...,
        description="Main headline title e.g. 'Purchase Order Request', 'Outstanding Balance Reminder'.",
    )
    salutation: str = Field(
        ...,
        description="Greeting e.g. 'Dear Ramesh Ji (Global Traders),', 'Respected Partner,'.",
    )
    intro_paragraphs: List[str] = Field(
        ...,
        description="1-2 paragraphs introducing the context clearly and politely.",
    )
    summary_cards: List[SummaryCard] = Field(
        default_factory=list,
        description="2-4 key metrics or highlight cards (e.g. Total Amount, Due Date, Delivery Timeline).",
    )
    table_headers: Optional[List[str]] = Field(
        default=None,
        description="Columns for table e.g. ['Item Description', 'Quantity', 'Unit Rate', 'Subtotal'].",
    )
    table_rows: Optional[List[List[str]]] = Field(
        default=None,
        description="Row items e.g. [['Amul Taaza 500ml', '50 pkts', '₹28.00', '₹1,400.00'], ...].",
    )
    table_footer: Optional[Dict[str, str]] = Field(
        default=None,
        description="Footer summary e.g. {'Subtotal': '₹1,400.00', 'GST (5%)': '₹70.00', 'Grand Total': '₹1,470.00'}.",
    )
    instruction_paragraphs: List[str] = Field(
        default_factory=list,
        description="Next steps, delivery instructions, payment details, or terms.",
    )
    action_button: Optional[ActionButton] = Field(
        default=None,
        description="Primary CTA button.",
    )
    closing_text: str = Field(
        default="Thank you for your partnership.",
        description="Polite closing phrase.",
    )
    sender: SenderSignature = Field(
        ...,
        description="Sender merchant details.",
    )
