"""
app/agent/service/purchases/write.py
====================================
Create and record official purchase orders directly into MongoDB `purchases` collection.
Synchronizes product stock levels, inventory records, and seller account totals.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from bson import ObjectId
from bson.errors import InvalidId
import structlog

from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.purchases.write")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    return None


async def create_purchase_order_record(
    store_id: str,
    seller_name: str,
    items: list[dict],
    invoice_number: str = "",
    paid_amount: float = 0.0,
    payment_status: str = "",
    notes: str = "",
) -> dict:
    """
    Creates a persistent purchase order document in MongoDB `purchases`,
    increments stock in `products` & `inventories`, and updates `sellers` totalPurchase/totalDue.
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        raise ValueError(f"Store '{store_id}' not found.")

    if not items:
        raise ValueError("At least one product item is required to place a purchase order.")

    # 1. Resolve or Match Existing Seller
    seller_oid = None
    seller_doc = None
    clean_seller_str = (seller_name or "").strip()

    try:
        if ObjectId.is_valid(clean_seller_str):
            seller_doc = await db["sellers"].find_one({"_id": ObjectId(clean_seller_str), "store": store_oid})
    except Exception:
        pass

    # Step 1: Exact case-insensitive match on name, businessName, email, or phone
    if not seller_doc and clean_seller_str:
        seller_doc = await db["sellers"].find_one({
            "store": store_oid,
            "$or": [
                {"name": {"$regex": f"^{re.escape(clean_seller_str)}$", "$options": "i"}},
                {"businessName": {"$regex": f"^{re.escape(clean_seller_str)}$", "$options": "i"}},
                {"email": {"$regex": f"^{re.escape(clean_seller_str)}$", "$options": "i"}},
                {"phone": {"$regex": f"^{re.escape(clean_seller_str)}$", "$options": "i"}},
            ]
        })

    # Step 2: Partial / substring match on name or businessName
    if not seller_doc and clean_seller_str:
        seller_doc = await db["sellers"].find_one({
            "store": store_oid,
            "$or": [
                {"name": {"$regex": re.escape(clean_seller_str), "$options": "i"}},
                {"businessName": {"$regex": re.escape(clean_seller_str), "$options": "i"}},
            ]
        })

    # Step 3: Multi-word / token match (e.g. "Ramesh" in "Ramesh Traders")
    if not seller_doc and clean_seller_str:
        words = [w for w in re.split(r"[\s\-_]+", clean_seller_str) if len(w) >= 3]
        for w in words:
            seller_doc = await db["sellers"].find_one({
                "store": store_oid,
                "$or": [
                    {"name": {"$regex": re.escape(w), "$options": "i"}},
                    {"businessName": {"$regex": re.escape(w), "$options": "i"}},
                ]
            })
            if seller_doc:
                break

    # Step 4: Fallback to existing single seller if only one exists in store
    if not seller_doc and clean_seller_str:
        all_store_sellers = await db["sellers"].find({"store": store_oid}).to_list(length=5)
        if len(all_store_sellers) == 1:
            seller_doc = all_store_sellers[0]

    if not seller_doc:
        # Auto-create the seller only if no existing seller matches
        new_seller = {
            "store": store_oid,
            "name": clean_seller_str or "General Supplier",
            "businessName": clean_seller_str or "General Supplier",
            "phone": "0000000000",
            "totalPurchase": 0.0,
            "totalPaid": 0.0,
            "totalDue": 0.0,
            "status": "active",
            "createdAt": datetime.now(timezone.utc),
            "updatedAt": datetime.now(timezone.utc),
        }
        res_sel = await db["sellers"].insert_one(new_seller)
        seller_oid = res_sel.inserted_id
        resolved_seller_name = clean_seller_str or "General Supplier"
        LOGGER.info("auto_created_seller_for_purchase", seller_name=resolved_seller_name)
    else:
        seller_oid = seller_doc["_id"]
        resolved_seller_name = seller_doc.get("businessName") or seller_doc.get("name") or clean_seller_str

    # 2. Process and resolve line items
    purchase_items = []
    subtotal = 0.0

    for it in items:
        p_name = it.get("product_name") or it.get("name", "Product")
        qty = float(it.get("quantity") or it.get("recommended_order_qty", 1))

        # Look up product in catalog
        p_doc = await db["products"].find_one({
            "store": store_oid,
            "name": {"$regex": f"^{re.escape(p_name.strip())}$", "$options": "i"}
        })

        if p_doc:
            p_oid = p_doc["_id"]
            cost_price = float(it.get("purchase_price") or it.get("unit_cost") or p_doc.get("buyingPrice") or (p_doc.get("sellingPrice", 100.0) * 0.8))
        else:
            # Fallback product ID or create basic SKU
            p_oid = ObjectId()
            cost_price = float(it.get("purchase_price") or it.get("unit_cost", 100.0))

        line_subtotal = round(qty * cost_price, 2)
        subtotal += line_subtotal

        purchase_items.append({
            "product": p_oid,
            "productName": p_name,
            "quantity": qty,
            "purchasePrice": cost_price,
            "discount": 0.0,
            "tax": 0.0,
            "subtotal": line_subtotal,
        })

    # 3. Calculate grand total & payment status
    grand_total = round(subtotal, 2)
    paid = float(paid_amount)
    due = max(0.0, round(grand_total - paid, 2))

    if payment_status in ("paid", "partial", "unpaid"):
        status = payment_status
    else:
        if paid >= grand_total:
            status = "paid"
        elif paid > 0:
            status = "partial"
        else:
            status = "unpaid"

    inv_num = invoice_number.strip() if invoice_number else f"PO-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:4].upper()}"

    # 4. Insert purchase document
    purchase_doc = {
        "store": store_oid,
        "seller": seller_oid,
        "invoiceNumber": inv_num,
        "purchaseDate": datetime.now(timezone.utc),
        "items": [
            {
                "product": pi["product"],
                "productName": pi.get("productName", ""),
                "quantity": pi["quantity"],
                "purchasePrice": pi["purchasePrice"],
                "discount": pi["discount"],
                "tax": pi["tax"],
                "subtotal": pi["subtotal"],
            }
            for pi in purchase_items
        ],
        "subtotal": subtotal,
        "discount": 0.0,
        "tax": 0.0,
        "grandTotal": grand_total,
        "paidAmount": paid,
        "dueAmount": due,
        "paymentStatus": status,
        "notes": notes or f"Ordered via Vyapar Sathi AI on {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
        "createdAt": datetime.now(timezone.utc),
        "updatedAt": datetime.now(timezone.utc),
    }

    result = await db["purchases"].insert_one(purchase_doc)
    purchase_id = str(result.inserted_id)
    LOGGER.info("purchase_order_created", purchase_id=purchase_id, invoice=inv_num, store_id=store_id)

    # 5. Increment product stock & sync inventory
    for pi in purchase_items:
        p_oid = pi["product"]
        qty = pi["quantity"]
        # Increment quantity on product
        await db["products"].update_one(
            {"_id": p_oid, "store": store_oid},
            {"$inc": {"quantity": qty}, "$set": {"updatedAt": datetime.now(timezone.utc)}}
        )
        # Update inventory record
        inv = await db["inventories"].find_one({"store": store_oid, "product": p_oid})
        if inv:
            new_qty = inv.get("quantity", 0) + qty
            min_stock = inv.get("minStockLevel", 10)
            await db["inventories"].update_one(
                {"_id": inv["_id"]},
                {
                    "$inc": {"quantity": qty},
                    "$set": {
                        "isOutOfStock": new_qty <= 0,
                        "isLowStock": 0 < new_qty <= min_stock,
                        "updatedAt": datetime.now(timezone.utc),
                    }
                }
            )

    # 6. Update seller financial totals
    await db["sellers"].update_one(
        {"_id": seller_oid, "store": store_oid},
        {
            "$inc": {
                "totalPurchase": grand_total,
                "totalPaid": paid,
                "totalDue": due,
            },
            "$set": {"updatedAt": datetime.now(timezone.utc)}
        }
    )

    return {
        "success": True,
        "purchase_id": purchase_id,
        "invoice_number": inv_num,
        "seller_name": resolved_seller_name,
        "seller_id": str(seller_oid),
        "items_count": len(purchase_items),
        "items": [
            {
                "product_name": pi["productName"],
                "quantity": pi["quantity"],
                "purchase_price": pi["purchasePrice"],
                "subtotal": pi["subtotal"],
            }
            for pi in purchase_items
        ],
        "grand_total": grand_total,
        "paid_amount": paid,
        "due_amount": due,
        "payment_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }



async def _find_purchase_by_identifier(db, store_oid: ObjectId, identifier: str) -> dict | None:
    """Finds a purchase document by ObjectId or by invoice number (exact/partial regex)."""
    clean_id = (identifier or "").strip()
    if not clean_id:
        return None

    # 1. Try ObjectId match
    try:
        if ObjectId.is_valid(clean_id):
            doc = await db["purchases"].find_one({"_id": ObjectId(clean_id), "store": store_oid})
            if doc:
                return doc
    except Exception:
        pass

    # 2. Try exact invoice number match (case-insensitive)
    doc = await db["purchases"].find_one({
        "store": store_oid,
        "invoiceNumber": {"$regex": f"^{re.escape(clean_id)}$", "$options": "i"}
    })
    if doc:
        return doc

    # 3. Try partial invoice number match
    doc = await db["purchases"].find_one({
        "store": store_oid,
        "invoiceNumber": {"$regex": re.escape(clean_id), "$options": "i"}
    })
    return doc


async def update_purchase_order_record(
    store_id: str,
    purchase_identifier: str,
    paid_amount: Optional[float] = None,
    payment_status: Optional[str] = None,
    notes: Optional[str] = None,
    invoice_number: Optional[str] = None,
    items: Optional[list[dict]] = None,
) -> dict:
    """
    Updates an existing purchase order by ID or invoice number (e.g. PO-20261003-ADF4).
    Re-balances seller ledger and inventory stock when paid amounts or items are modified.
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        raise ValueError(f"Store '{store_id}' not found.")

    purchase_doc = await _find_purchase_by_identifier(db, store_oid, purchase_identifier)
    if not purchase_doc:
        raise ValueError(f"Purchase order '{purchase_identifier}' not found in store records.")

    p_oid = purchase_doc["_id"]
    seller_oid = purchase_doc.get("seller")
    old_grand_total = float(purchase_doc.get("grandTotal", 0.0))
    old_paid = float(purchase_doc.get("paidAmount", 0.0))
    old_due = float(purchase_doc.get("dueAmount", 0.0))
    inv_num = purchase_doc.get("invoiceNumber", "")

    update_fields: dict[str, Any] = {"updatedAt": datetime.now(timezone.utc)}
    new_grand_total = old_grand_total

    # 1. Update items if provided (revert previous stock, apply new stock)
    if items is not None and len(items) > 0:
        # Revert old stock
        for old_it in purchase_doc.get("items", []):
            prod_id = old_it.get("product")
            old_qty = float(old_it.get("quantity", 0))
            if prod_id:
                await db["products"].update_one(
                    {"_id": prod_id, "store": store_oid},
                    {"$inc": {"quantity": -old_qty}, "$set": {"updatedAt": datetime.now(timezone.utc)}}
                )
                await db["inventories"].update_one(
                    {"product": prod_id, "store": store_oid},
                    {"$inc": {"quantity": -old_qty}, "$set": {"updatedAt": datetime.now(timezone.utc)}}
                )

        # Build and apply new items
        new_purchase_items = []
        new_subtotal = 0.0
        for it in items:
            p_name = it.get("product_name") or it.get("name", "Product")
            qty = float(it.get("quantity", 1))
            p_doc = await db["products"].find_one({
                "store": store_oid,
                "name": {"$regex": f"^{p_name.strip()}$", "$options": "i"}
            })
            prod_id = p_doc["_id"] if p_doc else ObjectId()
            cost_price = float(it.get("purchase_price") or (p_doc.get("buyingPrice") if p_doc else 100.0))
            line_tot = round(qty * cost_price, 2)
            new_subtotal += line_tot

            new_purchase_items.append({
                "product": prod_id,
                "productName": p_name,
                "quantity": qty,
                "purchasePrice": cost_price,
                "discount": 0.0,
                "tax": 0.0,
                "subtotal": line_tot,
            })

            # Increment new stock
            if p_doc:
                await db["products"].update_one(
                    {"_id": prod_id, "store": store_oid},
                    {"$inc": {"quantity": qty}, "$set": {"updatedAt": datetime.now(timezone.utc)}}
                )
                await db["inventories"].update_one(
                    {"product": prod_id, "store": store_oid},
                    {"$inc": {"quantity": qty}, "$set": {"updatedAt": datetime.now(timezone.utc)}}
                )

        new_grand_total = round(new_subtotal, 2)
        update_fields["items"] = new_purchase_items
        update_fields["subtotal"] = new_subtotal
        update_fields["grandTotal"] = new_grand_total

    # 2. Update paid amount & payment status
    if paid_amount is not None:
        new_paid = float(paid_amount)
    else:
        new_paid = old_paid

    new_due = max(0.0, round(new_grand_total - new_paid, 2))
    update_fields["paidAmount"] = new_paid
    update_fields["dueAmount"] = new_due

    if payment_status in ("paid", "partial", "unpaid"):
        update_fields["paymentStatus"] = payment_status
    else:
        if new_paid >= new_grand_total:
            update_fields["paymentStatus"] = "paid"
        elif new_paid > 0:
            update_fields["paymentStatus"] = "partial"
        else:
            update_fields["paymentStatus"] = "unpaid"

    if notes is not None:
        update_fields["notes"] = notes
    if invoice_number is not None and invoice_number.strip():
        update_fields["invoiceNumber"] = invoice_number.strip()
        inv_num = invoice_number.strip()

    await db["purchases"].update_one({"_id": p_oid}, {"$set": update_fields})

    # 3. Update seller totals with differences
    diff_total = new_grand_total - old_grand_total
    diff_paid = new_paid - old_paid
    diff_due = new_due - old_due

    if seller_oid and (diff_total != 0 or diff_paid != 0 or diff_due != 0):
        await db["sellers"].update_one(
            {"_id": seller_oid, "store": store_oid},
            {
                "$inc": {
                    "totalPurchase": diff_total,
                    "totalPaid": diff_paid,
                    "totalDue": diff_due,
                },
                "$set": {"updatedAt": datetime.now(timezone.utc)}
            }
        )

    # Fetch seller name
    seller_name = "Seller"
    if seller_oid:
        seller_doc = await db["sellers"].find_one({"_id": seller_oid})
        if seller_doc:
            seller_name = seller_doc.get("businessName") or seller_doc.get("name") or "Seller"

    LOGGER.info("purchase_order_updated", purchase_id=str(p_oid), invoice=inv_num, store_id=store_id)
    return {
        "success": True,
        "purchase_id": str(p_oid),
        "invoice_number": inv_num,
        "seller_name": seller_name,
        "grand_total": new_grand_total,
        "paid_amount": new_paid,
        "due_amount": new_due,
        "payment_status": update_fields.get("paymentStatus"),
        "notes": update_fields.get("notes", purchase_doc.get("notes")),
        "updated_at": update_fields["updatedAt"].isoformat(),
    }


async def delete_purchase_order_record(store_id: str, purchase_identifier: str) -> dict:
    """
    Deletes a purchase order by ID or invoice number, reverting product stock and seller balances.
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        raise ValueError(f"Store '{store_id}' not found.")

    purchase_doc = await _find_purchase_by_identifier(db, store_oid, purchase_identifier)
    if not purchase_doc:
        raise ValueError(f"Purchase order '{purchase_identifier}' not found in store records.")

    p_oid = purchase_doc["_id"]
    seller_oid = purchase_doc.get("seller")
    grand_total = float(purchase_doc.get("grandTotal", 0.0))
    paid_amount = float(purchase_doc.get("paidAmount", 0.0))
    due_amount = float(purchase_doc.get("dueAmount", 0.0))
    inv_num = purchase_doc.get("invoiceNumber", str(p_oid))

    # 1. Revert product stock
    for item in purchase_doc.get("items", []):
        prod_id = item.get("product")
        qty = float(item.get("quantity", 0))
        if prod_id and qty > 0:
            await db["products"].update_one(
                {"_id": prod_id, "store": store_oid},
                {"$inc": {"quantity": -qty}, "$set": {"updatedAt": datetime.now(timezone.utc)}}
            )
            await db["inventories"].update_one(
                {"product": prod_id, "store": store_oid},
                {"$inc": {"quantity": -qty}, "$set": {"updatedAt": datetime.now(timezone.utc)}}
            )

    # 2. Revert seller totals
    if seller_oid:
        await db["sellers"].update_one(
            {"_id": seller_oid, "store": store_oid},
            {
                "$inc": {
                    "totalPurchase": -grand_total,
                    "totalPaid": -paid_amount,
                    "totalDue": -due_amount,
                },
                "$set": {"updatedAt": datetime.now(timezone.utc)}
            }
        )

    # 3. Delete purchase document
    await db["purchases"].delete_one({"_id": p_oid, "store": store_oid})
    LOGGER.info("purchase_order_deleted", purchase_id=str(p_oid), invoice=inv_num)

    return {
        "success": True,
        "purchase_id": str(p_oid),
        "invoice_number": inv_num,
        "message": f"Successfully deleted purchase order {inv_num} and restored product stock.",
    }
