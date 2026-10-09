"""
app/agent/subgraphs/email_composer/nodes.py
===========================================
Node implementations for the Email Composer Subgraph.
Combines Fast SLM (classification/enrichment) + Live MongoDB Store Data Extraction
+ Quality LLM (structured draft composition with real numbers & tables)
+ Pre-baked Responsive HTML Engine + Bounded Repair Loop.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from bson import ObjectId
import structlog
from langchain_core.messages import SystemMessage, HumanMessage

from app.lib.llm import get_small_llm, get_llm
from app.config.database import get_database
from app.agent.tools.communication.send_email import (
    _resolve_recipient_email,
    _fetch_store_owner_info,
)
from app.agent.service.sales.summary import fetch_sales_summary, _resolve_store_id
from app.agent.subgraphs.email_composer.schemas import (
    EmailClassification,
    EmailDraftSpec,
    SenderSignature,
    SummaryCard,
    ActionButton,
)
from app.agent.subgraphs.email_composer.templates import (
    render_html_email,
    render_plain_text,
)
from app.agent.subgraphs.email_composer.state import EmailComposerState

LOGGER = structlog.get_logger("vyaparsathi.ai.email_composer.nodes")


# ---------------------------------------------------------------------------
# Helper: Fetch Live Business Data for Email Content
# ---------------------------------------------------------------------------

async def _fetch_live_business_data(db, store_id: str, prompt_text: str, category: str) -> Dict[str, Any]:
    """
    Fetches real live business records from MongoDB to supply real numbers and product rows for the email.
    """
    store_oid = await _resolve_store_id(db, store_id)
    ist = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist)
    current_date_str = now_ist.strftime("%d %B %Y")
    current_month_str = now_ist.strftime("%B %Y")

    data: Dict[str, Any] = {
        "current_date": current_date_str,
        "current_month": current_month_str,
        "store_id": store_id,
    }

    if not store_oid:
        return data

    prompt_lower = prompt_text.lower()
    is_inventory = category in ("inventory_report", "restock_alert") or any(w in prompt_lower for w in ["inventory", "stock", "product", "item", "catalog"])
    is_sales = category in ("sales_summary", "general_correspondence") or any(w in prompt_lower for w in ["sale", "sell", "revenue", "order", "income", "profit"])

    # 1. Fetch Inventory & Product Stats
    if is_inventory or category == "general_correspondence":
        try:
            total_products = await db["products"].count_documents({"store": store_oid, "isActive": {"$ne": False}})
            low_stock_cursor = db["products"].find({
                "store": store_oid,
                "isActive": {"$ne": False},
                "$expr": {"$lte": ["$stock", {"$ifNull": ["$minStock", 10]}]}
            }).limit(15)
            low_stock_items = await low_stock_cursor.to_list(length=15)

            all_products_cursor = db["products"].find(
                {"store": store_oid, "isActive": {"$ne": False}},
                {"name": 1, "productName": 1, "stock": 1, "sellingPrice": 1, "purchasePrice": 1, "category": 1, "unit": 1}
            ).limit(20)
            sample_products = await all_products_cursor.to_list(length=20)

            data["inventory"] = {
                "total_catalog_products": total_products,
                "low_stock_count": len(low_stock_items),
                "low_stock_items": [
                    {
                        "name": p.get("productName") or p.get("name", "Product"),
                        "current_stock": p.get("stock", 0),
                        "min_stock": p.get("minStock", 10),
                        "price": f"₹{p.get('sellingPrice', 0):,.2f}",
                        "category": p.get("category", "General"),
                    }
                    for p in low_stock_items
                ],
                "sample_catalog": [
                    {
                        "name": p.get("productName") or p.get("name", "Product"),
                        "stock": p.get("stock", 0),
                        "price": f"₹{p.get('sellingPrice', 0):,.2f}",
                        "cost": f"₹{p.get('purchasePrice', 0):,.2f}",
                        "unit": p.get("unit", "pcs"),
                    }
                    for p in sample_products
                ]
            }
        except Exception as inv_err:
            LOGGER.warning("email_composer_inv_fetch_error", error=str(inv_err))

    # 2. Fetch Sales Summary Stats
    if is_sales:
        try:
            sales_kpi = await fetch_sales_summary(store_id, days_lookback=30)
            data["sales_30d"] = {
                "total_revenue": f"₹{sales_kpi.get('total_revenue', 0.0):,.2f}",
                "total_orders": sales_kpi.get("total_sales_count", 0),
                "average_order_value": f"₹{sales_kpi.get('average_order_value', 0.0):,.2f}",
                "top_products": sales_kpi.get("top_products", [])[:5],
            }

            # Today's Sales
            start_of_day = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
            today_sales_cursor = db["sales"].find({
                "store": store_oid,
                "createdAt": {"$gte": start_of_day}
            })
            today_sales = await today_sales_cursor.to_list(length=100)
            today_rev = sum(s.get("totalAmount", 0) for s in today_sales)
            data["today_sales"] = {
                "orders_count": len(today_sales),
                "revenue": f"₹{today_rev:,.2f}",
            }
        except Exception as sale_err:
            LOGGER.warning("email_composer_sales_fetch_error", error=str(sale_err))

    return data


# ---------------------------------------------------------------------------
# Node 1: Classify and Enrich Context
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM_PROMPT = """You are an expert business communication classifier for an Indian MSME / Retail store (Vyapar Sathi).
Analyze the merchant's email drafting request and classify:
1. category: 'purchase_order', 'payment_reminder', 'quotation_inquiry', 'restock_alert', 'customer_announcement', 'inventory_report', 'sales_summary', or 'general_correspondence'
2. recipient_name: name of the seller, distributor, customer, or recipient (or null)
3. recipient_email: explicit email address if provided (or null)
4. language: 'English', 'Hindi', or 'Hinglish'. CRITICAL: Default to 'English' for all reports, summaries, and formal business emails UNLESS the user explicitly asked for Hindi (e.g. 'Hindi me likho', 'send in Hindi').
5. tone: 'formal', 'polite_urgent', 'friendly_promotional', or 'professional'
6. key_intent: A concise 1-sentence summary of what this email achieves.
"""


async def classify_and_enrich_node(state: EmailComposerState) -> dict[str, Any]:
    """
    Step 1: Uses fast Small LLM to classify intent, extracts store/owner details,
    and loads live business metrics from MongoDB.
    """
    request_prompt = state.get("request_prompt", "")
    store_id = state.get("store_id", "")
    user_id = state.get("user_id", "")

    LOGGER.info("email_composer.classify_and_enrich", store_id=store_id, prompt=request_prompt[:80])

    prompt_lower = request_prompt.lower()

    # Determine default language: English unless explicitly requested otherwise
    explicit_hindi = any(w in prompt_lower for w in ["hindi me", "in hindi", "हिंदी में", "हिंदी मे", "hindi mein"])
    default_language = "Hindi" if explicit_hindi else "English"

    # 1. Fast SLM Classification
    classification_data: Dict[str, Any] = {}
    try:
        small_llm = get_small_llm()
        structured_classifier = small_llm.with_structured_output(EmailClassification)
        result: EmailClassification = await structured_classifier.ainvoke([
            SystemMessage(content=_CLASSIFY_SYSTEM_PROMPT),
            HumanMessage(content=f"Merchant Request: {request_prompt}"),
        ])
        classification_data = result.model_dump()
    except Exception as exc:
        LOGGER.warning("email_composer.classification_fallback", error=str(exc))
        cat = "general_correspondence"
        if any(w in prompt_lower for w in ["inventory", "stock report", "catalog report"]):
            cat = "inventory_report"
        elif any(w in prompt_lower for w in ["sales report", "sales summary", "today's sales", "daily sales"]):
            cat = "sales_summary"
        elif any(w in prompt_lower for w in ["po", "purchase order", "order items", "buy from"]):
            cat = "purchase_order"
        elif any(w in prompt_lower for w in ["udhar", "due", "payment reminder", "pending balance"]):
            cat = "payment_reminder"
        elif any(w in prompt_lower for w in ["restock", "low stock alert"]):
            cat = "restock_alert"

        classification_data = {
            "category": state.get("category") or cat,
            "recipient_name": state.get("recipient_name"),
            "recipient_email": state.get("recipient_email"),
            "language": default_language,
            "tone": "professional",
            "key_intent": request_prompt[:120],
        }

    # Override with explicitly provided values in state if any
    category = state.get("category") or classification_data.get("category", "general_correspondence")
    recipient_name = state.get("recipient_name") or classification_data.get("recipient_name")
    recipient_email = state.get("recipient_email") or classification_data.get("recipient_email")
    language = state.get("language") or (default_language if not explicit_hindi else classification_data.get("language", "English"))

    # 2. Fetch Store & Owner Context from DB
    db = get_database()
    store_name, owner_name, owner_email, owner_phone = await _fetch_store_owner_info(
        db=db,
        store_id=store_id,
        user_id=user_id,
    )

    # 3. Resolve Recipient Email from database if name provided
    resolved_email, resolved_name, _ = await _resolve_recipient_email(
        store_id=store_id,
        recipient_email=recipient_email,
        recipient_name=recipient_name,
    )

    merchant_context = {
        "store_name": store_name,
        "owner_name": owner_name,
        "owner_email": owner_email or "support@vyaparsathi.in",
        "owner_phone": owner_phone,
        "store_id": store_id,
    }

    # 4. Fetch Live Business Metrics from MongoDB
    business_data = await _fetch_live_business_data(db, store_id, request_prompt, category)

    return {
        "classification": classification_data,
        "category": category,
        "recipient_name": resolved_name or recipient_name,
        "recipient_email": resolved_email or recipient_email,
        "language": language,
        "merchant_context": merchant_context,
        "business_context": business_data,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 2),
        "status": "in_progress",
    }


# ---------------------------------------------------------------------------
# Node 2: Compose Email Specification (LLM Quality Stage)
# ---------------------------------------------------------------------------

_COMPOSE_SYSTEM_PROMPT = """You are an elite business email composer for Indian retail store merchants and businesses (Vyapar Sathi).
Your task is to craft a complete, highly structured EmailDraftSpec JSON according to the merchant's request, category, and real store data.

STRICT DESIGN & DATA GUIDELINES:
1. LANGUAGE:
   - Always write the email in clean, professional English by default.
   - Only write in Hindi/Hinglish if the user explicitly requested Hindi.
   - Even if the merchant's prompt or speech is in Hindi/Hinglish, draft the email content in English unless instructed otherwise.

2. REAL DATA INJECTION (NO PLACEHOLDERS, NO EMPTY BOILERPLATES):
   - You MUST populate summary_cards and itemized data tables with the REAL NUMBERS and PRODUCTS from LIVE STORE BUSINESS DATA.
   - Use current real dates (e.g., 2026). NEVER use old hallucinated dates like 2023.
   - For Inventory Reports:
     * Summary Cards: Total Active Products, Low Stock Count, Total Catalog Categories, Report Date.
     * Line Items Table: Columns = ['Product Name', 'Category', 'Current Stock', 'Unit Price', 'Status']. Include actual products from the data!
   - For Sales Summaries:
     * Summary Cards: Total Revenue, Total Orders, Average Order Value, Reporting Period.
     * Line Items Table (if top products available): Columns = ['Product Name', 'Units Sold', 'Revenue Generated'].
   - For Purchase Orders & Restock:
     * Summary Cards: Total Order Value, Total Items, Expected Delivery.
     * Line Items Table: Columns = ['Item Description', 'Quantity', 'Estimated Rate', 'Subtotal'] with footer Grand Total.

3. STRUCTURED FIELDS:
   - Subject: Clear, professional subject line (e.g., '[Inventory Report] - Sharma Kirana Stock Overview as of October 2026' or 'Daily Sales Summary Report - NexaStore').
   - Preview Text: 1-sentence teaser summarizing the core numbers.
   - Salutation: Polite, respectful (e.g. 'Dear Udit Kumar Tiwari,', 'Dear Partner,').
   - Intro Paragraphs: 1-2 clear paragraphs summarizing the purpose and key highlights.
   - Instruction Paragraphs: 2-3 bullet points with next steps, remarks, or review notes.
   - Sender Signature: Use the provided merchant identity accurately.
"""


async def compose_spec_node(state: EmailComposerState) -> dict[str, Any]:
    """
    Step 2: Uses High-Quality LLM to generate structured EmailDraftSpec with real data.
    """
    request_prompt = state.get("request_prompt", "")
    category = state.get("category", "general_correspondence")
    recipient_name = state.get("recipient_name") or "Valued Merchant"
    recipient_email = state.get("recipient_email") or ""
    language = state.get("language", "English")
    merchant_ctx = state.get("merchant_context", {})
    biz_data = state.get("business_context", {})
    repair_feedback = state.get("repair_feedback")

    prompt_content = f"""
MERCHANT CONTEXT:
- Store Name: {merchant_ctx.get('store_name', 'Vyapar Store')}
- Merchant / Owner: {merchant_ctx.get('owner_name', 'Store Owner')}
- Store Official Email: {merchant_ctx.get('owner_email', 'store@vyaparsathi.in')}
- Store Phone: {merchant_ctx.get('owner_phone', '')}

TARGET DETAILS:
- Category: {category}
- Recipient Name: {recipient_name}
- Recipient Email: {recipient_email}
- Output Language: {language}

LIVE STORE BUSINESS DATA:
{biz_data}

MERCHANT REQUEST / DETAILS:
{request_prompt}
"""

    if repair_feedback:
        prompt_content += f"\n\nPREVIOUS VALIDATION ERRORS (PLEASE FIX):\n{repair_feedback}\n"

    LOGGER.info("email_composer.composing_spec", category=category, recipient=recipient_name, language=language)

    llm = get_llm()
    structured_composer = llm.with_structured_output(EmailDraftSpec)

    try:
        spec: EmailDraftSpec = await structured_composer.ainvoke([
            SystemMessage(content=_COMPOSE_SYSTEM_PROMPT),
            HumanMessage(content=prompt_content),
        ])
        draft_dict = spec.model_dump()
    except Exception as exc:
        LOGGER.error("email_composer.compose_spec_error", error=str(exc))
        store_name = merchant_ctx.get("store_name", "Vyapar Store")
        owner_name = merchant_ctx.get("owner_name", "Store Owner")
        sender_email = merchant_ctx.get("owner_email", "store@vyaparsathi.in")

        draft_dict = EmailDraftSpec(
            subject=f"Business Report from {store_name}",
            preview_text=f"Official business summary for {store_name}",
            badge_title="BUSINESS REPORT",
            headline=f"Official Update - {store_name}",
            salutation=f"Dear {recipient_name},",
            intro_paragraphs=[request_prompt],
            summary_cards=[
                SummaryCard(label="Store", value=store_name),
                SummaryCard(label="Date", value=biz_data.get("current_date", "Today")),
            ],
            instruction_paragraphs=["Please review the attached details at your convenience."],
            closing_text="Best regards,",
            sender=SenderSignature(
                store_name=store_name,
                owner_name=owner_name,
                email=sender_email,
                phone=merchant_ctx.get("owner_phone"),
            ),
        ).model_dump()

    return {"draft_spec": draft_dict}


# ---------------------------------------------------------------------------
# Node 3: Render Responsive HTML & Text
# ---------------------------------------------------------------------------

async def render_email_node(state: EmailComposerState) -> dict[str, Any]:
    """
    Step 3: Renders structured EmailDraftSpec into bulletproof responsive HTML & plain text.
    """
    draft_spec_dict = state.get("draft_spec", {})
    category = state.get("category", "general_correspondence")

    try:
        spec = EmailDraftSpec.model_validate(draft_spec_dict)
        rendered_html = render_html_email(spec, category=category)
        rendered_text = render_plain_text(spec)
    except Exception as exc:
        LOGGER.error("email_composer.render_error", error=str(exc))
        rendered_html = f"<div><p>{draft_spec_dict.get('subject', 'Official Email')}</p></div>"
        rendered_text = draft_spec_dict.get("subject", "Official Email")

    return {
        "rendered_html": rendered_html,
        "rendered_text": rendered_text,
    }


# ---------------------------------------------------------------------------
# Node 4: Validate Draft
# ---------------------------------------------------------------------------

_PLACEHOLDER_REGEX = re.compile(r"(\[Insert [^\]]+\]|\{\{[^}]+\}\}|<Insert [^>]+>|\[TODO[^\]]*\])", re.IGNORECASE)


async def validate_email_node(state: EmailComposerState) -> dict[str, Any]:
    """
    Step 4: Quality validation of the drafted email.
    """
    errors: List[str] = []
    draft_spec = state.get("draft_spec", {})
    html_content = state.get("rendered_html", "")

    # 1. Subject validation
    subject = (draft_spec.get("subject") or "").strip()
    if not subject or len(subject) < 5:
        errors.append("Subject line is missing or too short.")

    # 2. Intro paragraphs
    intros = draft_spec.get("intro_paragraphs", [])
    if not intros or not any(p.strip() for p in intros):
        errors.append("Email has no body or intro paragraphs.")

    # 3. Unreplaced placeholder detection
    placeholders = _PLACEHOLDER_REGEX.findall(html_content)
    if placeholders:
        errors.append(f"Unresolved placeholders found in email text: {', '.join(set(placeholders)[:3])}")

    # 4. Sender check
    sender = draft_spec.get("sender", {})
    if not sender.get("store_name") or not sender.get("email"):
        errors.append("Sender signature missing store name or email.")

    is_valid = len(errors) == 0

    LOGGER.info(
        "email_composer.validation",
        is_valid=is_valid,
        error_count=len(errors),
        errors=errors,
    )

    return {
        "validation_result": {"valid": is_valid, "errors": errors},
        "validation_errors": errors,
    }


# ---------------------------------------------------------------------------
# Node 5: Bounded Repair
# ---------------------------------------------------------------------------

async def repair_email_node(state: EmailComposerState) -> dict[str, Any]:
    """
    Step 5: Increments retry count and sets repair instructions.
    """
    retry_count = state.get("retry_count", 0) + 1
    errors = state.get("validation_errors", [])
    repair_feedback = "Please fix the following issues in the next draft:\n" + "\n".join(f"- {e}" for e in errors)

    LOGGER.info("email_composer.repair_attempt", retry_count=retry_count)

    return {
        "retry_count": retry_count,
        "repair_feedback": repair_feedback,
    }


# ---------------------------------------------------------------------------
# Node 6: Finalize Output
# ---------------------------------------------------------------------------

async def finalize_email_node(state: EmailComposerState) -> dict[str, Any]:
    """
    Step 6: Packages the complete email draft payload.
    """
    draft_spec = state.get("draft_spec", {})
    rendered_html = state.get("rendered_html", "")
    rendered_text = state.get("rendered_text", "")
    category = state.get("category", "general_correspondence")
    recipient_name = state.get("recipient_name") or "Recipient"
    recipient_email = state.get("recipient_email") or ""
    validation_res = state.get("validation_result", {})

    output = {
        "success": validation_res.get("valid", True) or bool(rendered_html),
        "category": category,
        "recipient_name": recipient_name,
        "recipient_email": recipient_email,
        "subject": draft_spec.get("subject", ""),
        "preview_text": draft_spec.get("preview_text", ""),
        "html_content": rendered_html,
        "plain_text": rendered_text,
        "summary_cards": draft_spec.get("summary_cards", []),
        "draft_spec": draft_spec,
        "attempts": state.get("retry_count", 0) + 1,
    }

    LOGGER.info("email_composer.finalized", subject=output["subject"], recipient=recipient_name)

    return {
        "email_output": output,
        "status": "success" if output["success"] else "failed",
    }
