"""
app/agent/service/purchases/search.py
=====================================
Search and filter purchase orders and supplier invoices from MongoDB.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.purchases.search")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    LOGGER.warning("purchases_store_not_found", store_id=store_id)
    return None


async def fetch_purchases(
    store_id: str,
    query: str = "",
    seller_name: str = "",
    payment_status: str = "",
    days_lookback: int = 60,
    limit: int = 20,
) -> list[dict]:
    """
    Search purchase invoices / orders for a store.
    Supports filtering by invoice number, seller, payment status, and date range.
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return []

    match: dict = {"store": store_oid}

    if days_lookback > 0:
        since = datetime.now(timezone.utc) - timedelta(days=days_lookback)
        match["purchaseDate"] = {"$gte": since}

    if payment_status in ("paid", "unpaid", "partial"):
        match["paymentStatus"] = payment_status

    if query:
        match["$or"] = [
            {"invoiceNumber": {"$regex": query, "$options": "i"}},
            {"billNumber": {"$regex": query, "$options": "i"}},
            {"notes": {"$regex": query, "$options": "i"}},
        ]

    # Pipeline to lookup seller details
    pipeline = [
        {"$match": match},
        {"$sort": {"purchaseDate": -1}},
        {"$limit": limit * 2 if seller_name else limit},
        {
            "$lookup": {
                "from": "sellers",
                "localField": "seller",
                "foreignField": "_id",
                "as": "seller_doc",
            }
        },
        {"$unwind": {"path": "$seller_doc", "preserveNullAndEmptyArrays": True}},
    ]

    cursor = db["purchases"].aggregate(pipeline)
    raw_results = await cursor.to_list(length=limit * 2 if seller_name else limit)

    output = []
    for p in raw_results:
        seller_info = p.get("seller_doc") or {}
        s_name = seller_info.get("name") or seller_info.get("businessName") or p.get("sellerName", "Unknown Seller")

        if seller_name and seller_name.lower() not in s_name.lower():
            continue

        p_date = p.get("purchaseDate")
        date_str = p_date.strftime("%Y-%m-%d") if isinstance(p_date, datetime) else str(p_date or "")

        items_summary = []
        for it in p.get("items", []):
            items_summary.append({
                "product_name": it.get("productName") or it.get("name", "Item"),
                "quantity": it.get("quantity", 0),
                "purchase_price": it.get("purchasePrice") or it.get("price", 0.0),
                "total_cost": it.get("totalCost") or it.get("total", 0.0),
            })

        output.append({
            "purchase_id": str(p["_id"]),
            "invoice_number": p.get("invoiceNumber") or p.get("billNumber") or f"PO-{str(p['_id'])[-6:].upper()}",
            "seller_name": s_name,
            "seller_id": str(p.get("seller", "")),
            "purchase_date": date_str,
            "total_amount": round(p.get("grandTotal") or p.get("totalAmount", 0.0), 2),
            "paid_amount": round(p.get("paidAmount", 0.0), 2),
            "due_amount": round(p.get("dueAmount", 0.0), 2),
            "payment_status": p.get("paymentStatus", "unpaid"),
            "items_count": len(items_summary),
            "items": items_summary[:5],
        })

        if len(output) >= limit:
            break

    LOGGER.debug("fetch_purchases", store_id=store_id, count=len(output))
    return output
