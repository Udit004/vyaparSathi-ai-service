"""
app/agent/service/products/write.py
=====================================
Create, update, and delete products via MongoDB, including inventory synchronization.
"""
from __future__ import annotations

from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.products.write")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    return None


async def create_product(store_id: str, data: dict, created_by: str) -> str:
    """
    Creates a Product and its matching Inventory record.
    Requires: name, category, selling_price, buying_price, quantity.
    """
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        raise ValueError(f"Store {store_id} not found.")

    try:
        user_oid = ObjectId(created_by)
    except InvalidId:
        user_oid = None  # Depending on requirements, this might need validation

    # 1. Create Product
    product_doc = {
        "store": store_oid,
        "name": data["name"],
        "brand": data.get("brand"),
        "barcode": data.get("barcode"),
        "sku": data.get("sku"),
        "category": data["category"],
        "sellingPrice": float(data["selling_price"]),
        "buyingPrice": float(data["buying_price"]),
        "quantity": float(data.get("quantity", 0)),
        "unit": data.get("unit", "Pieces"),
        "source": "copilot",
        "isActive": True,
        "createdBy": user_oid,
        "createdAt": datetime.now(timezone.utc),
        "updatedAt": datetime.now(timezone.utc),
    }

    if data.get("exp_date"):
        product_doc["expDate"] = datetime.fromisoformat(data["exp_date"].replace("Z", "+00:00"))

    prod_result = await db["products"].insert_one(product_doc)
    product_id = prod_result.inserted_id

    # 2. Create Inventory
    qty = float(data.get("quantity", 0))
    min_stock = float(data.get("min_stock_level", 10))
    is_out_of_stock = qty == 0
    is_low_stock = (not is_out_of_stock) and (qty <= min_stock)

    inv_doc = {
        "store": store_oid,
        "product": product_id,
        "quantity": qty,
        "minStockLevel": min_stock,
        "sellingPrice": float(data["selling_price"]),
        "isActive": True,
        "isLowStock": is_low_stock,
        "isOutOfStock": is_out_of_stock,
        "createdAt": datetime.now(timezone.utc),
        "updatedAt": datetime.now(timezone.utc),
    }
    await db["inventories"].insert_one(inv_doc)

    LOGGER.info("product_created", product_id=str(product_id), store_id=store_id)
    return str(product_id)


async def update_product(store_id: str, product_id: str, data: dict) -> bool:
    """Updates Product and syncs fields to Inventory if applicable."""
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        raise ValueError(f"Store {store_id} not found.")

    try:
        prod_oid = ObjectId(product_id)
    except InvalidId:
        raise ValueError("Invalid product ID.")

    # 1. Update Product fields
    prod_fields = {}
    if "name" in data: prod_fields["name"] = data["name"]
    if "brand" in data: prod_fields["brand"] = data["brand"]
    if "category" in data: prod_fields["category"] = data["category"]
    if "selling_price" in data: prod_fields["sellingPrice"] = float(data["selling_price"])
    if "buying_price" in data: prod_fields["buyingPrice"] = float(data["buying_price"])
    if "quantity" in data: prod_fields["quantity"] = float(data["quantity"])
    if "unit" in data: prod_fields["unit"] = data["unit"]
    if "barcode" in data: prod_fields["barcode"] = data["barcode"]
    if "sku" in data: prod_fields["sku"] = data["sku"]
    if "is_active" in data: prod_fields["isActive"] = data["is_active"]
    if "exp_date" in data:
        prod_fields["expDate"] = datetime.fromisoformat(data["exp_date"].replace("Z", "+00:00")) if data["exp_date"] else None

    if prod_fields:
        prod_fields["updatedAt"] = datetime.now(timezone.utc)
        await db["products"].update_one(
            {"_id": prod_oid, "store": store_oid},
            {"$set": prod_fields}
        )

    # 2. Sync to Inventory
    inv_fields = {}
    if "quantity" in data: inv_fields["quantity"] = float(data["quantity"])
    if "selling_price" in data: inv_fields["sellingPrice"] = float(data["selling_price"])
    if "min_stock_level" in data: inv_fields["minStockLevel"] = float(data["min_stock_level"])
    if "is_active" in data: inv_fields["isActive"] = data["is_active"]

    if inv_fields:
        # We must recalculate low/out of stock if quantity or min_stock_level changed
        inv = await db["inventories"].find_one({"store": store_oid, "product": prod_oid})
        if inv:
            new_qty = inv_fields.get("quantity", inv.get("quantity", 0))
            new_min = inv_fields.get("minStockLevel", inv.get("minStockLevel", 10))
            
            is_out_of_stock = new_qty == 0
            is_low_stock = (not is_out_of_stock) and (new_qty <= new_min)
            
            inv_fields["isOutOfStock"] = is_out_of_stock
            inv_fields["isLowStock"] = is_low_stock
            inv_fields["updatedAt"] = datetime.now(timezone.utc)

            await db["inventories"].update_one(
                {"_id": inv["_id"]},
                {"$set": inv_fields}
            )

    LOGGER.info("product_updated", product_id=product_id)
    return True


async def delete_product(store_id: str, product_id: str) -> bool:
    """Soft deletes the product and inventory."""
    return await update_product(store_id, product_id, {"is_active": False})
