"""
app/agent/tools/communication/send_email.py
===========================================
Dispatches emails to suppliers, distributors, or customers via the Express backend
email service (/api/email/send), eliminating credential duplication across services.
Automatically resolves seller & customer email addresses by name from the store database.
Also logs email dispatches to the Merchant Diary in Redis for an audit trail.
"""

from __future__ import annotations

import datetime
import os
import re
import uuid
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from bson import ObjectId
from bson.errors import InvalidId
import httpx
import structlog

from app.agent.memory.redis_cache import get_redis
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.send_email")

_EXPRESS_URL_ENV = "EXPRESS_BACKEND_URL"
_DEFAULT_EXPRESS_URL = "http://localhost:5000"


class SendEmailInput(BaseModel):
    recipient_email: Optional[str] = Field(
        default=None,
        description="Direct recipient email address if known (e.g. distributor@gmail.com). Optional if recipient_name is provided.",
    )
    recipient_name: Optional[str] = Field(
        default=None,
        description="Name of the seller, distributor, or customer (e.g. 'Global Traders', 'Ramesh Tiwari', 'Udit'). The tool will automatically lookup their email from the store database.",
    )
    subject: str = Field(..., description="Subject line of the email (e.g. 'Purchase Order #PO-2026-001 from NexaStore').")
    body_html: Optional[str] = Field(None, description="Formatted HTML email content with styling, tables, or item lists.")
    body_text: Optional[str] = Field(None, description="Plain text email fallback message.")
    attachments: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional list of file attachments (each item: {'filename': 'po.pdf', 'content': 'base64...' or 'path': '...'}).",
    )
    language: Optional[str] = Field(
        default="English",
        description="The language for the email: 'English', 'Hindi', or 'Hinglish'. Always ask/confirm with the merchant before dispatching.",
    )
    store_id: Optional[str] = Field(default=None, description="Store ID for logging.")
    user_id: Optional[str] = Field(default=None, description="User ID for logging.")


async def _safe_find_one(collection, query, projection=None):
    try:
        res = collection.find_one(query, projection) if projection else collection.find_one(query)
        if hasattr(res, "__await__"):
            return await res
        return res
    except Exception:
        return None


async def _resolve_recipient_email(
    store_id: str,
    recipient_email: Optional[str],
    recipient_name: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Resolves recipient email address. Returns (email, resolved_name, error_reason).
    """
    raw_target = (recipient_email or "").strip()
    name_target = (recipient_name or "").strip()

    db = get_database()
    store_oid = None
    if store_id:
        try:
            store_oid = ObjectId(store_id)
        except (InvalidId, TypeError):
            doc = await _safe_find_one(db["stores"], {"name": store_id}, {"_id": 1})
            if doc and isinstance(doc, dict):
                store_oid = doc.get("_id")

    # Case 1: Direct email address provided
    if raw_target and "@" in raw_target and "." in raw_target:
        if store_oid:
            seller = await _safe_find_one(
                db["sellers"],
                {"store": store_oid, "email": {"$regex": f"^{re.escape(raw_target)}$", "$options": "i"}}
            )
            if seller and isinstance(seller, dict):
                s_name = seller.get("businessName") or seller.get("name")
                return raw_target, s_name or name_target or raw_target, None
        return raw_target, name_target or raw_target, None

    lookup_query = name_target or raw_target
    if not lookup_query:
        return None, None, "Neither recipient email nor seller/customer name was provided."

    if not store_oid:
        return None, None, f"Store '{store_id}' not found in database."

    # Search in Sellers Collection
    # Step 1: Exact or case-insensitive match
    seller = await _safe_find_one(
        db["sellers"],
        {
            "store": store_oid,
            "$or": [
                {"name": {"$regex": f"^{re.escape(lookup_query)}$", "$options": "i"}},
                {"businessName": {"$regex": f"^{re.escape(lookup_query)}$", "$options": "i"}},
                {"email": {"$regex": f"^{re.escape(lookup_query)}$", "$options": "i"}},
                {"phone": {"$regex": f"^{re.escape(lookup_query)}$", "$options": "i"}},
            ]
        }
    )

    # Step 2: Substring match
    if not seller:
        seller = await _safe_find_one(
            db["sellers"],
            {
                "store": store_oid,
                "$or": [
                    {"name": {"$regex": re.escape(lookup_query), "$options": "i"}},
                    {"businessName": {"$regex": re.escape(lookup_query), "$options": "i"}},
                ]
            }
        )

    # Step 3: Word token match
    if not seller:
        tokens = [w for w in re.split(r"[\s\-_]+", lookup_query) if len(w) >= 3]
        for t in tokens:
            seller = await _safe_find_one(
                db["sellers"],
                {
                    "store": store_oid,
                    "$or": [
                        {"name": {"$regex": re.escape(t), "$options": "i"}},
                        {"businessName": {"$regex": re.escape(t), "$options": "i"}},
                    ]
                }
            )
            if seller:
                break

    if seller and isinstance(seller, dict):
        s_name = seller.get("businessName") or seller.get("name", lookup_query)
        s_email = (seller.get("email") or "").strip()
        if s_email and "@" in s_email:
            return s_email, s_name, None
        return None, s_name, f"Seller '{s_name}' was found in your Sellers section, but does not have an email address registered. Please update this seller's profile with their email."

    # Search in Buyers / Customers Collection
    buyer = await _safe_find_one(
        db["buyers"],
        {
            "store": store_oid,
            "$or": [
                {"name": {"$regex": re.escape(lookup_query), "$options": "i"}},
                {"email": {"$regex": re.escape(lookup_query), "$options": "i"}},
            ]
        }
    )
    if buyer:
        b_name = buyer.get("name", lookup_query)
        b_email = (buyer.get("email") or "").strip()
        if b_email and "@" in b_email:
            return b_email, b_name, None
        return None, b_name, f"Customer '{b_name}' was found in your Customers section, but does not have an email address registered."

    return None, lookup_query, f"Could not find any seller or customer named '{lookup_query}' in your store. Please check the name or provide their email address."


async def _fetch_store_owner_info(
    db,
    store_id: Optional[str],
    user_id: Optional[str],
) -> tuple[str, str, str, str]:
    """
    Resolves store and owner info: returns (store_name, owner_name, owner_email, owner_phone).
    """
    store_name = "Vyapar Sakha Store"
    owner_name = "Store Owner"
    owner_email = ""
    owner_phone = ""

    # 1. Fetch from user_id if provided
    if user_id:
        try:
            if ObjectId.is_valid(user_id):
                doc = await _safe_find_one(db["users"], {"_id": ObjectId(user_id)})
                if doc and isinstance(doc, dict):
                    owner_name = doc.get("name") or owner_name
                    owner_email = doc.get("email") or owner_email
                    owner_phone = doc.get("phone") or owner_phone
        except Exception:
            pass

    # 2. Fetch from store_id if provided
    if store_id:
        try:
            s_doc = None
            if ObjectId.is_valid(store_id):
                s_doc = await _safe_find_one(db["stores"], {"_id": ObjectId(store_id)})
            else:
                s_doc = await _safe_find_one(db["stores"], {"name": store_id})

            if s_doc and isinstance(s_doc, dict):
                store_name = s_doc.get("name") or store_name
                if not owner_email:
                    owner_email = s_doc.get("email") or ""

                owner_ref = s_doc.get("owner")
                if owner_ref and (owner_name == "Store Owner" or not owner_email):
                    u_oid = owner_ref if isinstance(owner_ref, ObjectId) else (ObjectId(str(owner_ref)) if ObjectId.is_valid(str(owner_ref)) else None)
                    if u_oid:
                        u_doc = await _safe_find_one(db["users"], {"_id": u_oid})
                        if u_doc and isinstance(u_doc, dict):
                            owner_name = u_doc.get("name") or owner_name
                            owner_email = u_doc.get("email") or owner_email
                            owner_phone = u_doc.get("phone") or owner_phone
        except Exception:
            pass

    return store_name, owner_name, owner_email, owner_phone


@tool("send_store_email", args_schema=SendEmailInput)
async def send_store_email(
    recipient_email: Optional[str] = None,
    recipient_name: Optional[str] = None,
    subject: str = "Official Message from Vyapar Sakha",
    body_html: Optional[str] = None,
    body_text: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    language: Optional[str] = "English",
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Send an official business email (such as Purchase Orders, invoice copies, payment reminders, or restock requests)
    to a distributor, seller, supplier, or customer using the verified store email service.

    You can provide either:
    - `recipient_name`: The name of the seller or customer (e.g. "Global Traders", "Ramesh Tiwari"). The tool automatically looks up their registered email from the store database!
    - `recipient_email`: A direct email address (e.g. distributor@example.com).
    - `language`: 'English', 'Hindi', or 'Hinglish'. Always confirm with the merchant before dispatching!

    Use this when:
    - The merchant asks "Send this purchase order to Global Traders", "Email invoice to seller Ramesh", "Send restock list to distributor".
    - Dispatching an approved draft purchase order or report directly to a recipient.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")
    target_id = store_id or user_id

    # 1. Resolve Recipient Email by Name or Address
    target_email, resolved_name, err_reason = await _resolve_recipient_email(
        store_id=store_id,
        recipient_email=recipient_email,
        recipient_name=recipient_name,
    )

    if not target_email:
        LOGGER.warning("recipient_resolution_failed", error=err_reason, name=recipient_name or recipient_email)
        return {
            "success": False,
            "status": "NOT_FOUND",
            "recipient_name": resolved_name,
            "message": err_reason,
        }

    # 2. Check Redis Scratchpad for latest PO Draft if body is empty
    html_content = body_html
    text_content = body_text

    if (not html_content and not text_content) and target_id:
        try:
            redis = await get_redis()
            if redis:
                key = f"vyapar:scratchpad:{target_id}"
                raw = await redis.get(key)
                if raw:
                    notes = json.loads(raw)
                    for n in notes:
                        if n.get("category") in ("purchase_order_draft", "purchase_order", "email_draft") and n.get("draft_data", {}).get("email_html"):
                            html_content = n["draft_data"]["email_html"]
                            text_content = n.get("content")
                            break
        except Exception as sc_err:
            LOGGER.debug("scratchpad_body_lookup_err", error=str(sc_err))

    # Auto-generate rich HTML via Email Composer Subgraph if HTML is missing or simple text
    is_rich_html = html_content and ("<table" in html_content.lower() or "box-shadow" in html_content.lower() or "<!doctype html" in html_content.lower())
    if not is_rich_html:
        try:
            from app.agent.subgraphs.email_composer.graph import email_composer_graph
            composer_prompt = text_content or html_content or subject
            LOGGER.info("auto_composing_html_email_via_subgraph", prompt=composer_prompt[:80], recipient=resolved_name)
            composer_input = {
                "request_prompt": f"Subject: {subject}\nRecipient: {resolved_name} ({target_email})\nContent:\n{composer_prompt}",
                "recipient_name": resolved_name or recipient_name,
                "recipient_email": target_email,
                "language": language or "English",
                "store_id": store_id,
                "user_id": user_id,
            }
            comp_res = await email_composer_graph.ainvoke(composer_input)
            email_out = comp_res.get("email_output", {})
            if email_out.get("html_content"):
                html_content = email_out["html_content"]
                if not subject or subject == "Official Message from Vyapar Sakha" or subject == "Today's Sales Summary":
                    subject = email_out.get("subject", subject)
                if email_out.get("plain_text"):
                    text_content = email_out["plain_text"]
        except Exception as comp_err:
            LOGGER.warning("send_email_auto_composer_failed", error=str(comp_err))

    html_content = html_content or (f"<p>{text_content.replace(chr(10), '<br/>')}</p>" if text_content else f"<p>{subject}</p>")
    text_content = text_content or (html_content or subject)

    # 3. Resolve Store Owner Name & Email and append sender information
    db = get_database()
    store_name, owner_name, owner_email, owner_phone = await _fetch_store_owner_info(
        db=db,
        store_id=store_id,
        user_id=user_id,
    )

    # If owner details not already present in the email body, append a clean owner identity footer (for plain text / simple snippet HTML only)
    is_complete_html_doc = "<html" in html_content.lower() or "<!doctype html" in html_content.lower()
    owner_info_already_in_body = bool(
        is_complete_html_doc
        or (owner_email and (owner_email.lower() in html_content.lower() or owner_email.lower() in text_content.lower()))
    )

    if not owner_info_already_in_body:
        lang_str = (language or "English").lower()
        if "hindi" in lang_str and "hinglish" not in lang_str:
            card_title = "दुकान व स्वामी विवरण"
            lbl_store = "दुकान का नाम:"
            lbl_owner = "दुकानदार का नाम:"
            lbl_email = "ईमेल:"
            lbl_phone = "फ़ोन:"
        elif "hinglish" in lang_str:
            card_title = "Store & Owner Details"
            lbl_store = "Store ka Naam:"
            lbl_owner = "Owner ka Naam:"
            lbl_email = "Email ID:"
            lbl_phone = "Phone:"
        else:
            card_title = "Store & Owner Information"
            lbl_store = "Store Name:"
            lbl_owner = "Store Owner:"
            lbl_email = "Owner Email:"
            lbl_phone = "Phone:"

        owner_email_display = owner_email or "Store registered email"
        owner_card_html = f"""
        <div style="margin-top: 24px; padding: 14px; background: #f8fafc; border: 1px solid #e2e8f0; border-left: 4px solid #2563eb; border-radius: 6px; font-family: Arial, sans-serif; font-size: 13px; color: #1e293b;">
          <div style="font-weight: bold; font-size: 14px; margin-bottom: 6px; color: #0f172a;">{card_title}</div>
          <div style="margin-bottom: 4px;"><b>{lbl_store}</b> {store_name}</div>
          <div style="margin-bottom: 4px;"><b>{lbl_owner}</b> {owner_name}</div>
          <div style="margin-bottom: 4px;"><b>{lbl_email}</b> <a href="mailto:{owner_email_display}" style="color: #2563eb; text-decoration: none;">{owner_email_display}</a></div>
          {f'<div style="margin-bottom: 2px;"><b>{lbl_phone}</b> {owner_phone}</div>' if owner_phone else ''}
        </div>
        """
        html_content = html_content + owner_card_html
        text_content = (
            text_content
            + f"\n\n---\n{lbl_store} {store_name}\n{lbl_owner} {owner_name}\n{lbl_email} {owner_email_display}"
            + (f"\n{lbl_phone} {owner_phone}" if owner_phone else "")
        )

    express_base = os.getenv(_EXPRESS_URL_ENV, _DEFAULT_EXPRESS_URL).rstrip("/")
    endpoint = f"{express_base}/api/email/send"

    payload = {
        "to": target_email.strip(),
        "subject": subject.strip(),
        "html": html_content,
        "text": text_content,
    }
    if owner_email:
        payload["replyTo"] = owner_email.strip()
    if attachments:
        payload["attachments"] = attachments

    internal_key = os.getenv("INTERNAL_SERVICE_KEY", "vyapar-internal-ai-service-key")
    headers = {
        "x-internal-service-key": internal_key,
        "x-service-name": "vyapar-ai-service",
        "Authorization": f"Bearer {internal_key}",
    }

    LOGGER.info("dispatching_email_via_express", endpoint=endpoint, to=target_email, name=resolved_name, subject=subject)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

            if resp.status_code == 200 and data.get("success") is not False:
                LOGGER.info("email_sent_successfully", to=target_email, name=resolved_name)

                # Finalize draft PO into MongoDB Purchases & save email audit log into Redis
                purchase_rec_result = None
                try:
                    redis = await get_redis()
                    if redis and target_id:
                        log_key = f"vyapar:scratchpad:{target_id}"
                        existing = await redis.get(log_key)
                        notes = json.loads(existing) if existing else []

                        # If there is an active draft PO, convert it into an official purchase order
                        if store_id:
                            for n in notes:
                                if n.get("category") in ("purchase_order_draft", "purchase_order") and n.get("status") == "draft":
                                    d_data = n.get("draft_data", {})
                                    d_items = d_data.get("items", [])
                                    d_po_num = d_data.get("po_number") or f"PO-{datetime.datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"
                                    if d_items:
                                        from app.agent.service.purchases.write import create_purchase_order_record
                                        formatted_items = [
                                            {
                                                "product_name": it.get("product_name") or it.get("name"),
                                                "quantity": float(it.get("recommended_order_qty") or it.get("quantity", 1)),
                                                "purchase_price": float(it.get("unit_cost") or it.get("purchase_price", 100.0)),
                                            }
                                            for it in d_items
                                        ]
                                        rec_res = await create_purchase_order_record(
                                            store_id=store_id,
                                            seller_name=resolved_name or target_email,
                                            items=formatted_items,
                                            invoice_number=d_po_num,
                                            notes=f"Ordered & emailed via Vyapar Sakha AI to {target_email} on {datetime.datetime.now().strftime('%d %b %Y, %I:%M %p')}",
                                        )
                                        purchase_rec_result = rec_res
                                        n["status"] = "completed"
                                        n["draft_data"]["purchase_id"] = rec_res.get("purchase_id")
                                        n["draft_data"]["invoice_number"] = rec_res.get("invoice_number")
                                        LOGGER.info("draft_po_finalized_as_purchase", po=d_po_num, purchase_id=rec_res.get("purchase_id"))
                                    break

                        # Prepend email sent log
                        notes.insert(0, {
                            "note_id": f"mail_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                            "title": f"Email Sent to {resolved_name or target_email}: {subject}",
                            "content": f"Sent to {resolved_name} ({target_email}) on {datetime.datetime.now().strftime('%d %b %Y, %I:%M %p')}\n\n{text_content[:300]}...",
                            "category": "email_sent",
                            "status": "completed",
                            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        })
                        await redis.set(log_key, json.dumps(notes[:40]), ex=7 * 86400)
                except Exception as log_err:
                    LOGGER.warning("email_diary_and_purchase_sync_error", error=str(log_err))

                msg = f"Email successfully dispatched to {resolved_name} ({target_email})."
                if purchase_rec_result:
                    msg += f" Purchase Order {purchase_rec_result.get('invoice_number')} has been recorded in your Purchases page and inventory stock updated."

                return {
                    "success": True,
                    "recipient": target_email,
                    "recipient_name": resolved_name,
                    "subject": subject,
                    "status": "SENT",
                    "purchase_record": purchase_rec_result,
                    "message": msg,
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
            else:
                err_msg = data.get("message") or f"HTTP {resp.status_code}: {resp.text}"
                LOGGER.warning("express_email_failed", error=err_msg)
                return {
                    "success": False,
                    "recipient": target_email,
                    "recipient_name": resolved_name,
                    "status": "FAILED",
                    "message": f"Express email backend returned error: {err_msg}",
                }

    except Exception as exc:
        LOGGER.error("email_dispatch_exception", error=str(exc))
        return {
            "success": False,
            "recipient": target_email,
            "recipient_name": resolved_name,
            "status": "ERROR",
            "message": f"Could not connect to email service: {str(exc)}. Please verify Express backend is running on {express_base}.",
        }
