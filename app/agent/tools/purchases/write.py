"""
app/agent/tools/purchases/write.py
==================================
Record and place official purchase orders from sellers and distributors directly into MongoDB.
Ensures the order is immediately visible on the store's Purchases dashboard and increments stock.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import structlog

from app.agent.service.purchases.write import (
    create_purchase_order_record,
    update_purchase_order_record,
    delete_purchase_order_record,
    receive_purchase_order_stock,
)
from app.agent.memory.redis_cache import get_redis

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.purchases.write")


class PurchaseLineItem(BaseModel):
    product_name: str = Field(..., description="Name of the product being purchased/ordered (e.g. 'Tata Salt 1kg', 'Fortune Oil').")
    quantity: float = Field(..., gt=0, description="Quantity or units to order (e.g. 20, 50, 10).")
    purchase_price: Optional[float] = Field(default=None, description="Purchase/wholesale price per unit in ₹. If omitted, uses current buying price.")

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Fallbacks for product_name
            if "product_name" not in data or not data.get("product_name"):
                for alias in ("name", "item_name", "product", "title", "itemName", "item"):
                    if alias in data and data.get(alias):
                        data["product_name"] = str(data[alias])
                        break
            # Fallbacks for quantity
            if "quantity" not in data or data.get("quantity") is None:
                for alias in ("qty", "count", "units", "quantity_ordered", "amount"):
                    if alias in data and data.get(alias) is not None:
                        data["quantity"] = data[alias]
                        break
            # Fallbacks for purchase_price
            if "purchase_price" not in data or data.get("purchase_price") is None:
                for alias in ("price", "unit_price", "cost", "unit_cost", "rate", "buying_price"):
                    if alias in data and data.get(alias) is not None:
                        data["purchase_price"] = data[alias]
                        break
        return data


class CreatePurchaseInput(BaseModel):
    seller_name: str = Field(..., description="Name of the seller, vendor, or wholesale distributor (e.g. 'Global Traders', 'Ramesh Enterprises').")
    items: List[PurchaseLineItem] = Field(..., description="List of items to order with product name, quantity, and optional purchase price.")
    paid_amount: float = Field(default=0.0, ge=0, description="Amount paid upfront to the seller in ₹ (0 for credit/unpaid order).")
    payment_status: Optional[str] = Field(default="unpaid", description="Payment status: 'unpaid', 'partial', or 'paid'.")
    invoice_number: Optional[str] = Field(default=None, description="Optional invoice or bill number (auto-generated if omitted).")
    notes: Optional[str] = Field(default=None, description="Optional notes or remarks for this purchase order.")
    received_into_stock: bool = Field(default=False, description="Set to True ONLY if the merchant explicitly confirms the order/goods have ALREADY arrived physically and should be added to stock immediately. Otherwise leave False (default).")
    store_id: Optional[str] = Field(default=None, description="The store ID.")
    user_id: Optional[str] = Field(default=None, description="The user/owner ID.")

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Fallbacks for seller_name
            if "seller_name" not in data or not data.get("seller_name"):
                for alias in ("seller", "vendor", "supplier", "supplier_name", "distributor"):
                    if alias in data and data.get(alias):
                        data["seller_name"] = str(data[alias])
                        break
        return data


@tool("tool_create_purchase", args_schema=CreatePurchaseInput)
async def tool_create_purchase(
    seller_name: str,
    items: List[PurchaseLineItem],
    paid_amount: float = 0.0,
    payment_status: Optional[str] = "unpaid",
    invoice_number: Optional[str] = None,
    notes: Optional[str] = None,
    received_into_stock: bool = False,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Place and record an official Purchase Order or supplier bill from a seller/distributor.
    This writes directly to the store database so that:
    1. The purchase order appears immediately in your Purchases section & dashboard.
    2. The seller's outstanding balance & purchase ledger are updated on the Sellers page.
    3. Product inventory stock levels are updated ONLY IF received_into_stock is True (e.g. goods have arrived).

    Use this when:
    - The merchant asks "Order 20 packets of Tata Salt from Global Traders", "Place order with seller...", "Create purchase entry for 50 boxes of biscuits".
    - Placing a new purchase order to be sent to a seller.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")
    target_id = store_id or user_id

    LOGGER.info("tool_create_purchase_invoked", store_id=store_id, seller=seller_name, items_count=len(items))

    if not store_id:
        return {
            "success": False,
            "message": "Store ID missing from runtime context. Cannot record purchase.",
        }

    try:
        raw_items = [
            {
                "product_name": it.product_name if isinstance(it, PurchaseLineItem) else it["product_name"],
                "quantity": float(it.quantity if isinstance(it, PurchaseLineItem) else it["quantity"]),
                "purchase_price": it.purchase_price if isinstance(it, PurchaseLineItem) else it.get("purchase_price"),
            }
            for it in items
        ]

        res = await create_purchase_order_record(
            store_id=store_id,
            seller_name=seller_name,
            items=raw_items,
            invoice_number=invoice_number or "",
            paid_amount=paid_amount,
            payment_status=payment_status or "",
            notes=notes or "",
            received_into_stock=received_into_stock,
        )

        # Log into Redis Merchant Diary as completed order
        if target_id:
            try:
                redis = await get_redis()
                if redis:
                    key = f"vyapar:scratchpad:{target_id}"
                    existing = await redis.get(key)
                    diary = json.loads(existing) if existing else []
                    diary.insert(0, {
                        "note_id": f"po_rec_{res['purchase_id'][:8]}",
                        "title": f"Purchased from {res['seller_name']}: {res['invoice_number']}",
                        "content": f"Official Purchase Order recorded for {res['items_count']} items totaling ₹{res['grand_total']:,.2f}.\nStatus: {res['payment_status']}. Visible on Purchases dashboard.",
                        "category": "purchase_order",
                        "status": "completed",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    })
                    await redis.set(key, json.dumps(diary[:40]), ex=7 * 86400)
            except Exception as d_err:
                LOGGER.warning("purchase_diary_log_error", error=str(d_err))

        items_desc = ", ".join([f"{it['quantity']}x {it['product_name']}" for it in res["items"]])
        if received_into_stock:
            stock_msg = "This order is recorded on your Purchases page and stock has been added to your inventory."
        else:
            stock_msg = "This order is recorded on your Purchases & Sellers pages (order status: 'ordered'). Stock will be added to your inventory once you confirm receipt of the physical shipment."

        summary = (
            f"Successfully recorded purchase order {res['invoice_number']} from {res['seller_name']} "
            f"({items_desc}) totaling ₹{res['grand_total']:,.2f}. {stock_msg}"
        )

        return {
            "success": True,
            "purchase_id": res["purchase_id"],
            "invoice_number": res["invoice_number"],
            "seller_name": res["seller_name"],
            "grand_total": res["grand_total"],
            "paid_amount": res["paid_amount"],
            "due_amount": res["due_amount"],
            "payment_status": res["payment_status"],
            "stock_updated": received_into_stock,
            "items": res["items"],
            "summary": summary,
        }

    except Exception as exc:
        LOGGER.error("tool_create_purchase_error", error=str(exc))
        return {
            "success": False,
            "message": f"Failed to record purchase order: {str(exc)}",
        }


class ReceivePurchaseInput(BaseModel):
    purchase_identifier: str = Field(..., description="The purchase invoice number (e.g. 'PO-20261003-ADF4') or purchase ID ObjectId string.")
    store_id: Optional[str] = Field(default=None, description="The store ID.")
    user_id: Optional[str] = Field(default=None, description="The user/owner ID.")


@tool("tool_receive_purchase", args_schema=ReceivePurchaseInput)
async def tool_receive_purchase(
    purchase_identifier: str,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Mark a previously placed purchase order (e.g. 'PO-20261003-ADF4') as physically received/delivered,
    and increment product & inventory stock levels.
    
    Use this when the merchant explicitly states:
    - "I have received the order PO-...", "Goods have arrived for PO-...", "Maal receive ho gaya hai, stock me add kar do", "Add items from PO-XXX into stock".
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    if not store_id:
        return {
            "success": False,
            "message": "Store ID missing from runtime context. Cannot mark purchase received.",
        }

    try:
        res = await receive_purchase_order_stock(
            store_id=store_id,
            purchase_identifier=purchase_identifier,
        )
        return {
            "success": True,
            "purchase_id": res.get("purchase_id"),
            "invoice_number": res.get("invoice_number"),
            "already_received": res.get("already_received", False),
            "items": res.get("items", []),
            "summary": res.get("message", f"Purchase order {purchase_identifier} marked as received and stock updated!"),
        }
    except Exception as exc:
        LOGGER.error("tool_receive_purchase_error", error=str(exc))
        return {
            "success": False,
            "message": f"Failed to mark purchase as received: {str(exc)}",
        }


class UpdatePurchaseInput(BaseModel):
    purchase_identifier: str = Field(..., description="The purchase invoice number (e.g. 'PO-20261003-ADF4') or purchase ID ObjectId string.")
    paid_amount: Optional[float] = Field(default=None, description="Updated amount paid in ₹ to supplier.")
    payment_status: Optional[str] = Field(default=None, description="Updated payment status: 'paid', 'partial', or 'unpaid'.")
    notes: Optional[str] = Field(default=None, description="Updated notes or remarks.")
    invoice_number: Optional[str] = Field(default=None, description="Updated invoice number.")
    items: Optional[List[PurchaseLineItem]] = Field(default=None, description="Updated items if replacing line items.")
    store_id: Optional[str] = Field(default=None, description="The store ID.")
    user_id: Optional[str] = Field(default=None, description="The user/owner ID.")


@tool("tool_update_purchase", args_schema=UpdatePurchaseInput)
async def tool_update_purchase(
    purchase_identifier: str,
    paid_amount: Optional[float] = None,
    payment_status: Optional[str] = None,
    notes: Optional[str] = None,
    invoice_number: Optional[str] = None,
    items: Optional[List[PurchaseLineItem]] = None,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Update an existing Purchase Order or supplier bill by invoice number (e.g. 'PO-20261003-ADF4') or purchase ID.
    Use this to update payment status ('paid', 'partial', 'unpaid'), paid amounts, notes, or items.
    Automatically synchronizes seller ledger dues and inventory.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    if not store_id:
        return {
            "success": False,
            "message": "Store ID missing from runtime context. Cannot update purchase.",
        }

    try:
        raw_items = None
        if items is not None:
            raw_items = [
                {
                    "product_name": it.product_name if isinstance(it, PurchaseLineItem) else it["product_name"],
                    "quantity": float(it.quantity if isinstance(it, PurchaseLineItem) else it["quantity"]),
                    "purchase_price": it.purchase_price if isinstance(it, PurchaseLineItem) else it.get("purchase_price"),
                }
                for it in items
            ]

        res = await update_purchase_order_record(
            store_id=store_id,
            purchase_identifier=purchase_identifier,
            paid_amount=paid_amount,
            payment_status=payment_status,
            notes=notes,
            invoice_number=invoice_number,
            items=raw_items,
        )

        summary = (
            f"Successfully updated purchase order {res['invoice_number']} for {res['seller_name']}. "
            f"Grand Total: ₹{res['grand_total']:,.2f}, Paid: ₹{res['paid_amount']:,.2f}, "
            f"Due: ₹{res['due_amount']:,.2f}, Status: {res['payment_status']}."
        )

        return {
            "success": True,
            "purchase_id": res["purchase_id"],
            "invoice_number": res["invoice_number"],
            "seller_name": res["seller_name"],
            "grand_total": res["grand_total"],
            "paid_amount": res["paid_amount"],
            "due_amount": res["due_amount"],
            "payment_status": res["payment_status"],
            "summary": summary,
        }

    except Exception as exc:
        LOGGER.error("tool_update_purchase_error", error=str(exc))
        return {
            "success": False,
            "message": f"Failed to update purchase order: {str(exc)}",
        }


class DeletePurchaseInput(BaseModel):
    purchase_identifier: str = Field(..., description="The purchase invoice number (e.g. 'PO-20261003-ADF4') or purchase ID ObjectId string to delete.")
    store_id: Optional[str] = Field(default=None, description="The store ID.")
    user_id: Optional[str] = Field(default=None, description="The user/owner ID.")


@tool("tool_delete_purchase", args_schema=DeletePurchaseInput)
async def tool_delete_purchase(
    purchase_identifier: str,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Delete / cancel an existing purchase order by invoice number (e.g. 'PO-20261003-ADF4') or purchase ID.
    Automatically restores product stock and adjusts seller dues.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    if not store_id:
        return {
            "success": False,
            "message": "Store ID missing from runtime context. Cannot delete purchase.",
        }

    try:
        res = await delete_purchase_order_record(
            store_id=store_id,
            purchase_identifier=purchase_identifier,
        )
        return {
            "success": True,
            "purchase_id": res["purchase_id"],
            "invoice_number": res["invoice_number"],
            "summary": res["message"],
        }
    except Exception as exc:
        LOGGER.error("tool_delete_purchase_error", error=str(exc))
        return {
            "success": False,
            "message": f"Failed to delete purchase order: {str(exc)}",
        }
