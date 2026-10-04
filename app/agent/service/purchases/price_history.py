"""
Supplier Price History & Intelligence Service.
Tracks item purchase prices across suppliers, logs price changes from invoice scans, and detects cost hikes.
Collection: supplier_price_history
"""
from __future__ import annotations

from datetime import datetime, timezone
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.purchases.price_history")


async def record_supplier_price(
    user_id: str,
    store_id: str | None,
    supplier_name: str,
    product_name: str,
    unit_cost_price: float,
) -> dict:
    """
    Log purchase price for an item and track price inflation/discount compared to prior records.
    """
    if not user_id or not product_name or unit_cost_price <= 0:
        return {"error": "Invalid inputs for recording supplier price."}

    db = get_database()
    now = datetime.now(timezone.utc)

    # Check last recorded price for this product & user
    last_record = await db["supplier_price_history"].find_one(
        {"user_id": str(user_id), "product_name": product_name.strip()},
        sort=[("created_at", -1)]
    )

    prev_price = float(last_record.get("unit_cost_price", unit_cost_price)) if last_record else unit_cost_price
    price_diff = unit_cost_price - prev_price
    price_change_pct = round((price_diff / prev_price) * 100.0, 2) if prev_price > 0 else 0.0

    doc = {
        "user_id": str(user_id),
        "store_id": str(store_id) if store_id else None,
        "supplier_name": supplier_name.strip(),
        "product_name": product_name.strip(),
        "unit_cost_price": float(unit_cost_price),
        "previous_cost_price": prev_price,
        "price_change_pct": price_change_pct,
        "created_at": now
    }

    res = await db["supplier_price_history"].insert_one(doc)
    doc["_id"] = str(res.inserted_id)

    LOGGER.info(
        "supplier_price_recorded",
        user_id=user_id,
        supplier=supplier_name,
        product=product_name,
        price=unit_cost_price,
        pct_change=price_change_pct
    )
    return doc


async def get_supplier_price_trends(
    user_id: str,
    product_name: str | None = None,
    supplier_name: str | None = None,
    limit: int = 10
) -> list[dict]:
    """
    Query historical price movements for products across suppliers.
    """
    if not user_id:
        return []

    db = get_database()
    query: dict = {"user_id": str(user_id)}

    if product_name:
        query["product_name"] = {"$regex": product_name.strip(), "$options": "i"}
    if supplier_name:
        query["supplier_name"] = {"$regex": supplier_name.strip(), "$options": "i"}

    cursor = db["supplier_price_history"].find(query).sort("created_at", -1).limit(limit)
    records = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        records.append(doc)

    return records
