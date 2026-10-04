"""
app/agent/tools/forecast/smart_purchase_order.py
================================================
Generates end-to-end intelligent Purchase Orders based on real-time stock deficits,
sales velocity forecast, and supplier lead times.
Automatically registers the generated draft into the Redis Merchant Scratchpad for
multi-turn review, modification, or direct email dispatch by either Voice or Text agent.
"""

from __future__ import annotations

import datetime
import math
import uuid
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from bson import ObjectId
import structlog

from app.config.database import get_database
from app.agent.service.inventory.low_stock import fetch_low_stock_products
from app.agent.service.forecast.restock import fetch_restock_priorities
from app.agent.memory.redis_cache import get_redis

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.smart_purchase_order")


class SmartPurchaseOrderInput(BaseModel):
    supplier_name: Optional[str] = Field(
        default=None,
        description="Filter by specific supplier or distributor name (e.g. 'Ramesh Traders'). If omitted, includes all suppliers.",
    )
    days_lead_time: int = Field(
        default=3,
        ge=1,
        le=30,
        description="Supplier fulfillment lead time in days (e.g. 3 days).",
    )
    buffer_days: int = Field(
        default=7,
        ge=2,
        le=45,
        description="Safety buffer days of inventory demand to fulfill.",
    )
    auto_save_to_scratchpad: bool = Field(
        default=True,
        description="Whether to automatically store this draft purchase order in Redis Merchant Diary.",
    )
    store_id: Optional[str] = Field(default=None, description="The store ID.")
    user_id: Optional[str] = Field(default=None, description="The user/owner ID.")


class POItemDetail(BaseModel):
    product_name: str
    current_stock: int
    daily_run_rate: float
    recommended_order_qty: int
    unit_cost: float
    line_total: float
    supplier_name: str
    urgency: str


class SmartPurchaseOrderOutput(BaseModel):
    po_number: str
    total_items_count: int
    total_units_ordered: int
    total_estimated_amount: float
    items: List[POItemDetail]
    supplier_groups: Dict[str, float]
    scratchpad_note_id: Optional[str] = None
    email_ready_html: str
    voice_summary: str
    created_at: str


@tool("create_smart_purchase_order", args_schema=SmartPurchaseOrderInput)
async def create_smart_purchase_order(
    supplier_name: Optional[str] = None,
    days_lead_time: int = 3,
    buffer_days: int = 7,
    auto_save_to_scratchpad: bool = True,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Generate an end-to-end Smart Purchase Order (PO) for items running low or at stockout risk.
    Calculates exact order quantities based on sales run-rate, lead time (e.g. 3 days), and buffer days.
    Calculates itemized supplier cost breakdown, compiles HTML/Text PO tables, and saves the draft
    into the Redis Merchant Diary so it can be confirmed, modified, or emailed.

    Use this when:
    - The merchant asks "Generate a purchase order for low stock items", "Distributor ke liye PO bana do", "Order list tayyar karo".
    - Planning inventory replenishment before stock runs out.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")
    target_id = store_id or user_id

    LOGGER.info("create_smart_purchase_order_invoked", store_id=store_id, supplier=supplier_name)

    if not store_id:
        return {
            "po_number": "PO-UNKNOWN",
            "total_items_count": 0,
            "total_units_ordered": 0,
            "total_estimated_amount": 0.0,
            "items": [],
            "supplier_groups": {},
            "voice_summary": "Store identity missing from context.",
            "email_ready_html": "",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    # 1. Fetch restock velocity & low stock
    raw_priorities = []
    try:
        raw_priorities = await fetch_restock_priorities(store_id)
    except Exception as e:
        LOGGER.warning("restock_priority_fetch_err", error=str(e))

    raw_low_stock = []
    try:
        raw_low_stock = await fetch_low_stock_products(store_id, threshold=25, limit=30)
    except Exception as e:
        LOGGER.warning("low_stock_fetch_err", error=str(e))

    # Merge products
    merged_products: Dict[str, Dict[str, Any]] = {}

    for item in raw_priorities:
        p_name = item.get("name") or item.get("product_name", "Item")
        merged_products[p_name] = {
            "name": p_name,
            "stock": int(item.get("current_quantity", 0)),
            "daily_sales": float(item.get("avg_daily_sales", 1.5)),
            "priority": item.get("priority", "RED"),
            "suggested": int(item.get("suggested_restock_quantity", 15)),
            "price": float(item.get("price", 100.0)),
            "supplier": item.get("supplier_name", "Primary Distributor"),
        }

    for item in raw_low_stock:
        p_name = item.get("name") or item.get("product_name", "Item")
        if p_name not in merged_products:
            merged_products[p_name] = {
                "name": p_name,
                "stock": int(item.get("current_quantity", 0)),
                "daily_sales": 1.0,
                "priority": "YELLOW",
                "suggested": max(10, 20 - int(item.get("current_quantity", 0))),
                "price": float(item.get("price", 100.0)),
                "supplier": item.get("supplier_name", "Primary Distributor"),
            }

    # 2. Build line items
    planning_window_days = days_lead_time + buffer_days
    po_items: List[Dict[str, Any]] = []
    supplier_groups: Dict[str, float] = {}
    total_units = 0
    grand_total_cost = 0.0

    for name, p in merged_products.items():
        sup = p.get("supplier") or "Primary Distributor"
        if supplier_name and supplier_name.lower() not in sup.lower():
            continue

        stock = p["stock"]
        daily_sales = max(0.5, p["daily_sales"])
        needed_units = int(math.ceil(daily_sales * planning_window_days))
        order_qty = max(10, needed_units - stock)
        unit_cost = float(p.get("price", 100.0)) * 0.8  # Estimate 80% wholesale cost
        line_total = round(order_qty * unit_cost, 2)

        total_units += order_qty
        grand_total_cost += line_total
        supplier_groups[sup] = supplier_groups.get(sup, 0.0) + line_total

        po_items.append({
            "product_name": name,
            "current_stock": stock,
            "daily_run_rate": round(daily_sales, 1),
            "recommended_order_qty": order_qty,
            "unit_cost": round(unit_cost, 2),
            "line_total": line_total,
            "supplier_name": sup,
            "urgency": p["priority"],
        })

    po_items.sort(key=lambda x: (0 if x["urgency"] == "RED" else 1, -x["line_total"]))

    # 3. Resolve Store & Owner Details
    store_name = "Vyapar Sakha Store"
    owner_name = "Store Owner"
    owner_email = ""
    try:
        db = get_database()
        s_doc = None
        if ObjectId.is_valid(store_id):
            s_res = db["stores"].find_one({"_id": ObjectId(store_id)})
            s_doc = (await s_res) if hasattr(s_res, "__await__") else s_res
        elif store_id:
            s_res = db["stores"].find_one({"name": store_id})
            s_doc = (await s_res) if hasattr(s_res, "__await__") else s_res

        if s_doc and isinstance(s_doc, dict):
            store_name = s_doc.get("name") or store_name
            owner_ref = s_doc.get("owner")
            if owner_ref:
                u_oid = owner_ref if isinstance(owner_ref, ObjectId) else (ObjectId(str(owner_ref)) if ObjectId.is_valid(str(owner_ref)) else None)
                if u_oid:
                    u_res = db["users"].find_one({"_id": u_oid})
                    u_doc = (await u_res) if hasattr(u_res, "__await__") else u_res
                    if u_doc and isinstance(u_doc, dict):
                        owner_name = u_doc.get("name") or owner_name
                        owner_email = u_doc.get("email") or owner_email
        if not owner_email and user_id and ObjectId.is_valid(user_id):
            u_res = db["users"].find_one({"_id": ObjectId(user_id)})
            u_doc = (await u_res) if hasattr(u_res, "__await__") else u_res
            if u_doc and isinstance(u_doc, dict):
                owner_name = u_doc.get("name") or owner_name
                owner_email = u_doc.get("email") or owner_email
    except Exception as fetch_err:
        LOGGER.debug("po_owner_meta_fetch_err", error=str(fetch_err))

    # 4. Generate PO Identifier
    po_number = f"PO-{datetime.datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"

    # 5. Generate HTML Email/Invoice Template
    rows_html = "".join([
        f"<tr>"
        f"<td style='padding:8px;border:1px solid #e2e8f0;'><b>{item['product_name']}</b></td>"
        f"<td style='padding:8px;border:1px solid #e2e8f0;text-align:center;'>{item['current_stock']}</td>"
        f"<td style='padding:8px;border:1px solid #e2e8f0;text-align:center;'><b>{item['recommended_order_qty']}</b></td>"
        f"<td style='padding:8px;border:1px solid #e2e8f0;text-align:right;'>₹{item['unit_cost']:,.2f}</td>"
        f"<td style='padding:8px;border:1px solid #e2e8f0;text-align:right;'>₹{item['line_total']:,.2f}</td>"
        f"</tr>"
        for item in po_items
    ])

    owner_contact_snippet = f" | Owner: <b>{owner_name}</b>" + (f" (<a href='mailto:{owner_email}'>{owner_email}</a>)" if owner_email else "")
    email_html = f"""
    <div style='font-family: Arial, sans-serif; max-width: 650px; margin: 0 auto; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px;'>
      <div style='border-bottom: 2px solid #3b82f6; padding-bottom: 12px; margin-bottom: 16px;'>
        <h2 style='color: #1e3a8a; margin: 0;'>PURCHASE ORDER: {po_number}</h2>
        <p style='color: #64748b; margin: 4px 0 0 0;'>Date: {datetime.datetime.now().strftime('%d %B %Y')} | Lead Time: {days_lead_time} days</p>
        <p style='color: #334155; margin: 6px 0 0 0; font-size: 13px;'>
          From Store: <b>{store_name}</b>{owner_contact_snippet}
        </p>
      </div>
      <table style='width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 14px;'>
        <thead>
          <tr style='background: #f1f5f9;'>
            <th style='padding:8px;border:1px solid #e2e8f0;text-align:left;'>Item</th>
            <th style='padding:8px;border:1px solid #e2e8f0;text-align:center;'>Current Stock</th>
            <th style='padding:8px;border:1px solid #e2e8f0;text-align:center;'>Order Qty</th>
            <th style='padding:8px;border:1px solid #e2e8f0;text-align:right;'>Unit Cost</th>
            <th style='padding:8px;border:1px solid #e2e8f0;text-align:right;'>Total</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
      <div style='text-align: right; margin-top: 12px;'>
        <p style='font-size: 16px; font-weight: bold; color: #0f172a; margin: 0;'>
          Grand Total: ₹{grand_total_cost:,.2f} ({total_units} units across {len(po_items)} items)
        </p>
      </div>
      <div style='margin-top: 20px; padding: 12px; background: #f8fafc; border-top: 1px solid #e2e8f0; border-radius: 4px; font-size: 12px; color: #64748b;'>
        <p style='margin: 0;'><b>Authorized Store Owner:</b> {owner_name} {f"| <b>Email:</b> {owner_email}" if owner_email else ""}</p>
      </div>
    </div>
    """

    # 5. Spoken summary for Voice & Copilot
    top_items_text = ", ".join([f"{it['recommended_order_qty']} units of {it['product_name']}" for it in po_items[:3]])
    voice_summary = (
        f"I have generated purchase order {po_number} covering {len(po_items)} items for a total of ₹{grand_total_cost:,.2f}. "
        f"Key items include {top_items_text}. "
        f"This draft has been saved into your Merchant Diary so you can confirm or email it whenever you're ready."
        if po_items else "All inventory items are currently well-stocked. No purchase order required."
    )

    # 6. Auto-save into Redis Merchant Diary
    scratchpad_note_id = None
    if auto_save_to_scratchpad and target_id and po_items:
        try:
            redis = await get_redis()
            if redis:
                scratchpad_note_id = f"po_{uuid.uuid4().hex[:8]}"
                note_entry = {
                    "note_id": scratchpad_note_id,
                    "title": f"Draft Purchase Order: {po_number}",
                    "content": f"Purchase Order for {len(po_items)} items totaling ₹{grand_total_cost:,.2f}.\nPlanning window: {planning_window_days} days.\nTop suppliers: {', '.join(supplier_groups.keys())}.",
                    "category": "purchase_order_draft",
                    "draft_data": {
                        "po_number": po_number,
                        "total_amount": grand_total_cost,
                        "total_units": total_units,
                        "supplier_breakdown": supplier_groups,
                        "items": po_items,
                        "email_html": email_html,
                    },
                    "status": "draft",
                    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                key = f"vyapar:scratchpad:{target_id}"
                raw = await redis.get(key)
                notes = json.loads(raw) if raw else []
                notes.insert(0, note_entry)
                await redis.set(key, json.dumps(notes[:40]), ex=7 * 86400)
                LOGGER.info("po_draft_saved_to_scratchpad", note_id=scratchpad_note_id, po=po_number)
        except Exception as scratch_err:
            LOGGER.warning("po_scratchpad_save_err", error=str(scratch_err))

    return {
        "po_number": po_number,
        "total_items_count": len(po_items),
        "total_units_ordered": total_units,
        "total_estimated_amount": grand_total_cost,
        "items": po_items,
        "supplier_groups": supplier_groups,
        "scratchpad_note_id": scratchpad_note_id,
        "email_ready_html": email_html,
        "voice_summary": voice_summary,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
