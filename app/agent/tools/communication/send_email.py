"""
app/agent/tools/communication/send_email.py
===========================================
Dispatches emails to suppliers, distributors, or customers via the Express backend
email service (/api/email/send), eliminating credential duplication across services.
Also logs email dispatches to the Merchant Diary in Redis for audit trail.
"""

from __future__ import annotations

import datetime
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import httpx
import structlog

from app.agent.memory.redis_cache import get_redis

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.send_email")

_EXPRESS_URL_ENV = "EXPRESS_BACKEND_URL"
_DEFAULT_EXPRESS_URL = "http://localhost:5000"


class SendEmailInput(BaseModel):
    recipient_email: str = Field(..., description="Valid recipient email address (e.g. distributor@gmail.com, customer@gmail.com).")
    subject: str = Field(..., description="Subject line of the email (e.g. 'Purchase Order #PO-2026-001 from NexaStore').")
    body_html: Optional[str] = Field(None, description="Formatted HTML email content with styling, tables, or item lists.")
    body_text: Optional[str] = Field(None, description="Plain text email fallback message.")
    attachments: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional list of file attachments (each item: {'filename': 'po.pdf', 'content': 'base64...' or 'path': '...'}).",
    )
    store_id: Optional[str] = Field(default=None, description="Store ID for logging.")
    user_id: Optional[str] = Field(default=None, description="User ID for logging.")


@tool("send_store_email", args_schema=SendEmailInput)
async def send_store_email(
    recipient_email: str,
    subject: str,
    body_html: Optional[str] = None,
    body_text: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Send an official business email (such as Purchase Orders, invoice copies, payment reminders, or restock requests)
    to a distributor, supplier, or customer using the verified store email service.

    Use this when:
    - The merchant asks "Send this purchase order to distributor@example.com", "Email invoice to Ramesh", or "Send restock list by email".
    - You need to dispatch an approved draft purchase order or report directly to a recipient.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")
    target_id = store_id or user_id

    # Fallback plain text / html if one is missing
    html_content = body_html or (f"<p>{body_text.replace(chr(10), '<br/>')}</p>" if body_text else f"<p>{subject}</p>")
    text_content = body_text or (body_html or subject)

    express_base = os.getenv(_EXPRESS_URL_ENV, _DEFAULT_EXPRESS_URL).rstrip("/")
    endpoint = f"{express_base}/api/email/send"

    payload = {
        "to": recipient_email.strip(),
        "subject": subject.strip(),
        "html": html_content,
        "text": text_content,
    }
    if attachments:
        payload["attachments"] = attachments

    internal_key = os.getenv("INTERNAL_SERVICE_KEY", "vyapar-internal-ai-service-key")
    headers = {
        "x-internal-service-key": internal_key,
        "x-service-name": "vyapar-ai-service",
        "Authorization": f"Bearer {internal_key}",
    }

    LOGGER.info("dispatching_email_via_express", endpoint=endpoint, to=recipient_email, subject=subject)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

            if resp.status_code == 200 and data.get("success") is not False:
                LOGGER.info("email_sent_successfully", to=recipient_email)

                # Save email log into Redis Merchant Diary as audit trail
                try:
                    redis = await get_redis()
                    if redis and target_id:
                        import json
                        log_key = f"vyapar:scratchpad:{target_id}"
                        existing = await redis.get(log_key)
                        notes = json.loads(existing) if existing else []
                        notes.insert(0, {
                            "note_id": f"mail_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
                            "title": f"Email Sent: {subject}",
                            "content": f"Sent to {recipient_email} on {datetime.datetime.now().strftime('%d %b %Y, %I:%M %p')}\n\n{text_content[:300]}...",
                            "category": "email_sent",
                            "status": "completed",
                            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        })
                        await redis.set(log_key, json.dumps(notes[:40]), ex=7 * 86400)
                except Exception as log_err:
                    LOGGER.warning("email_diary_log_error", error=str(log_err))

                return {
                    "success": True,
                    "recipient": recipient_email,
                    "subject": subject,
                    "status": "SENT",
                    "message": f"Email successfully dispatched to {recipient_email}.",
                    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
            else:
                err_msg = data.get("message") or f"HTTP {resp.status_code}: {resp.text}"
                LOGGER.warning("express_email_failed", error=err_msg)
                return {
                    "success": False,
                    "recipient": recipient_email,
                    "status": "FAILED",
                    "message": f"Express email backend returned error: {err_msg}",
                }

    except Exception as exc:
        LOGGER.error("email_dispatch_exception", error=str(exc))
        return {
            "success": False,
            "recipient": recipient_email,
            "status": "ERROR",
            "message": f"Could not connect to email service: {str(exc)}. Please verify Express backend is running on {express_base}.",
        }
