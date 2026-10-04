"""
app/agent/tools/purchases/invoice_ocr.py
=========================================
Supplier Invoice OCR & Image Scanner Tool (Vision AI).

Accepts an uploaded photo or image URL of a physical paper bill/invoice,
uses Gemini Vision AI to extract line items, prices, quantities, and supplier details,
and optionally drafts/records a Purchase Order document in the database.
"""

import json
import base64
import httpx
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import structlog

from app.lib.gemini_keys import get_next_gemini_key
from app.agent.service.products.search import search_products
from app.agent.service.purchases.write import create_purchase_order_record

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.tools.purchases.invoice_ocr")


class InvoiceOcrInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    image_input: str = Field(
        ...,
        description="The uploaded paper bill image — accepts an image URL (http/https) or base64-encoded image string."
    )
    auto_create_purchase: bool = Field(
        default=False,
        description="If True, automatically records the extracted purchase order into the store database."
    )
    user_id: Optional[str] = Field(None, description="Optional user ID context")

    class Config:
        extra = "ignore"


async def _fetch_image_base64_and_mime(image_input: str) -> tuple[str, str]:
    """Resolve image input (URL or raw base64) to (base64_data, mime_type)."""
    clean_input = image_input.strip()

    # Data URI format (e.g. data:image/png;base64,iVBORw0...)
    if clean_input.startswith("data:"):
        header, b64 = clean_input.split(",", 1)
        mime = header.split(";")[0].replace("data:", "") or "image/jpeg"
        return b64, mime

    # HTTP / HTTPS URL
    if clean_input.startswith("http://") or clean_input.startswith("https://"):
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(clean_input)
            resp.raise_for_status()
            mime = resp.headers.get("content-type", "image/jpeg").split(";")[0]
            b64 = base64.b64encode(resp.content).decode("ascii")
            return b64, mime

    # Pure base64 fallback
    return clean_input, "image/jpeg"


async def _parse_invoice_with_gemini_vision(base64_data: str, mime_type: str) -> Dict[str, Any]:
    """Call Gemini Multimodal Vision API to parse invoice line items and metadata."""
    api_key = get_next_gemini_key()
    if not api_key:
        raise ValueError("No GEMINI_API_KEY configured for Vision OCR.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    prompt_text = """You are an expert Indian retail accountant Vision OCR engine.
Carefully analyze this physical paper supplier bill / invoice image and extract all structured data.

Extract the following JSON schema strictly:
{
  "seller_name": "Supplier or Vendor company name (or 'Unknown Supplier')",
  "invoice_number": "Invoice or Bill Bill number (or '')",
  "invoice_date": "YYYY-MM-DD or date text",
  "line_items": [
    {
      "name": "Product name or description",
      "quantity": 10.0,
      "unit_price": 45.0,
      "total_price": 450.0,
      "tax_rate": 5.0
    }
  ],
  "total_amount": 450.0,
  "paid_amount": 450.0,
  "payment_status": "paid or pending or partial"
}

Return ONLY valid JSON matching this schema."""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text},
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": base64_data
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.1
        }
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError("Gemini Vision returned empty OCR response.")

    text_part = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    return json.loads(text_part)


@tool("parse_supplier_invoice_image", args_schema=InvoiceOcrInput)
async def parse_supplier_invoice_image(
    store_id: str,
    image_input: str,
    auto_create_purchase: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """Parse an uploaded paper invoice or bill photo using Gemini Vision AI.
    Extracts supplier name, invoice number, items, quantities, unit prices, and total amount.
    Optionally creates a draft Purchase Order in the database.
    """
    LOGGER.info("parse_supplier_invoice_image_start", store_id=store_id, auto_create=auto_create_purchase)

    try:
        b64_data, mime_type = await _fetch_image_base64_and_mime(image_input)
        parsed_data = await _parse_invoice_with_gemini_vision(b64_data, mime_type)
    except Exception as exc:
        LOGGER.error("invoice_ocr_failed", error=str(exc), exc_info=True)
        return {
            "status": "error",
            "message": f"Failed to parse invoice image via Vision AI: {str(exc)}"
        }

    seller_name = parsed_data.get("seller_name", "Unknown Seller")
    invoice_number = parsed_data.get("invoice_number", f"INV-{datetime.now().strftime('%Y%m%d-%H%M')}")
    raw_items = parsed_data.get("line_items", [])
    total_amount = float(parsed_data.get("total_amount", 0.0))
    payment_status = parsed_data.get("payment_status", "pending")

    # Match extracted line items with existing store catalog products
    matched_items = []
    for item in raw_items:
        item_name = item.get("name", "Product")
        qty = float(item.get("quantity", 1.0))
        unit_price = float(item.get("unit_price", 0.0))
        line_total = float(item.get("total_price", qty * unit_price))

        # Search store catalog for matching product
        catalog_matches = await search_products(store_id=store_id, query=item_name)
        matched_product_id = catalog_matches[0]["product_id"] if catalog_matches else None

        matched_items.append({
            "product_name": item_name,
            "product_id": matched_product_id,
            "quantity": qty,
            "unit_price": unit_price,
            "line_total": line_total,
            "catalog_matched": matched_product_id is not None
        })

        # Record unit price in supplier_price_history collection for price intelligence
        if unit_price > 0:
            try:
                from app.agent.service.purchases.price_history import record_supplier_price
                effective_user_id = kwargs.get("user_id") or kwargs.get("configurable", {}).get("user_id") or "system"
                await record_supplier_price(
                    user_id=effective_user_id,
                    store_id=store_id,
                    supplier_name=seller_name,
                    product_name=item_name,
                    unit_cost_price=unit_price
                )
            except Exception as pe:
                LOGGER.debug("record_supplier_price_ocr_failed", error=str(pe))

    # Auto-create Purchase Order in MongoDB if requested
    created_po = None
    if auto_create_purchase and matched_items:
        try:
            po_items = [
                {
                    "product_name": it["product_name"],
                    "quantity": it["quantity"],
                    "unit_price": it["unit_price"],
                }
                for it in matched_items
            ]
            created_po = await create_purchase_order_record(
                store_id=store_id,
                seller_name=seller_name,
                items=po_items,
                invoice_number=invoice_number,
                payment_status=payment_status,
                notes="Created automatically via Vision AI Invoice OCR"
            )
        except Exception as exc:
            LOGGER.warning("invoice_auto_po_creation_failed", error=str(exc))

    item_names_str = ", ".join(f"{it['quantity']}x {it['product_name']}" for it in matched_items[:3])
    summary_text = (
        f"Successfully scanned paper bill from '{seller_name}' (Invoice #{invoice_number}). "
        f"Extracted {len(matched_items)} items ({item_names_str}) totaling ₹{total_amount:,.2f}."
    )
    if created_po:
        summary_text += f" Purchase Order #{created_po.get('purchase_id', '')} created in database."

    return {
        "status": "success",
        "seller_name": seller_name,
        "invoice_number": invoice_number,
        "invoice_date": parsed_data.get("invoice_date", datetime.now().strftime("%Y-%m-%d")),
        "total_amount": total_amount,
        "payment_status": payment_status,
        "extracted_items_count": len(matched_items),
        "items": matched_items,
        "purchase_order_created": created_po is not None,
        "created_purchase_details": created_po,
        "executive_summary": summary_text
    }
