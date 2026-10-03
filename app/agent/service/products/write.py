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


import re


async def update_product(store_id: str, product_id: str = None, data: dict = None) -> dict:
    """
    Updates Product and syncs fields to Inventory.
    Supports setting absolute quantity or adding relative quantity (add_quantity).
    Automatically resolves product by ID or by name.
    """
    if data is None:
        data = {}

    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        raise ValueError(f"Store {store_id} not found.")

    products_col = db["products"]
    inventories_col = db["inventories"]

    prod_doc = None
    prod_oid = None

    # 1. Try finding product by ObjectId
    if product_id:
        try:
            prod_oid = ObjectId(product_id)
            prod_doc = await products_col.find_one({"_id": prod_oid, "store": store_oid})
        except (InvalidId, TypeError):
            prod_oid = None

    # 2. Try finding product by name or alias if not found by ObjectId
    if not prod_doc:
        search_name = (data.get("product_name") or product_id or data.get("name") or "").strip()
        if search_name:
            # Exact regex match (case-insensitive)
            prod_doc = await products_col.find_one({
                "store": store_oid,
                "name": {"$regex": f"^{re.escape(search_name)}$", "$options": "i"},
                "isActive": {"$ne": False}
            })
            if not prod_doc:
                # Partial regex match
                prod_doc = await products_col.find_one({
                    "store": store_oid,
                    "name": {"$regex": re.escape(search_name), "$options": "i"},
                    "isActive": {"$ne": False}
                })

    if not prod_doc:
        identifier = product_id or data.get("product_name") or data.get("name") or "unknown"
        raise ValueError(f"Product '{identifier}' not found in store.")

    prod_oid = prod_doc["_id"]
    prod_name = prod_doc.get("name", "Product")
    previous_qty = float(prod_doc.get("quantity") or 0.0)

    # 3. Handle Product fields
    prod_fields = {}
    if "name" in data and data["name"]:
        prod_fields["name"] = data["name"]
    if "brand" in data:
        prod_fields["brand"] = data["brand"]
    if "category" in data:
        prod_fields["category"] = data["category"]
    if "selling_price" in data and data["selling_price"] is not None:
        prod_fields["sellingPrice"] = float(data["selling_price"])
    if "buying_price" in data and data["buying_price"] is not None:
        prod_fields["buyingPrice"] = float(data["buying_price"])

    # Check for relative stock addition vs absolute override
    added_delta = None
    new_calculated_qty = None

    if "add_quantity" in data and data["add_quantity"] is not None:
        added_delta = float(data["add_quantity"])
        new_calculated_qty = max(0.0, previous_qty + added_delta)
        prod_fields["quantity"] = new_calculated_qty
    elif "quantity_to_add" in data and data["quantity_to_add"] is not None:
        added_delta = float(data["quantity_to_add"])
        new_calculated_qty = max(0.0, previous_qty + added_delta)
        prod_fields["quantity"] = new_calculated_qty
    elif "quantity" in data and data["quantity"] is not None:
        new_calculated_qty = max(0.0, float(data["quantity"]))
        prod_fields["quantity"] = new_calculated_qty

    if "unit" in data:
        prod_fields["unit"] = data["unit"]
    if "barcode" in data:
        prod_fields["barcode"] = data["barcode"]
    if "sku" in data:
        prod_fields["sku"] = data["sku"]
    if "is_active" in data and data["is_active"] is not None:
        prod_fields["isActive"] = data["is_active"]
    if "exp_date" in data:
        prod_fields["expDate"] = datetime.fromisoformat(data["exp_date"].replace("Z", "+00:00")) if data["exp_date"] else None

    if prod_fields:
        prod_fields["updatedAt"] = datetime.now(timezone.utc)
        await products_col.update_one(
            {"_id": prod_oid, "store": store_oid},
            {"$set": prod_fields}
        )

    # 4. Sync to Inventory
    inv_fields = {}
    if new_calculated_qty is not None:
        inv_fields["quantity"] = new_calculated_qty
    if "selling_price" in data and data["selling_price"] is not None:
        inv_fields["sellingPrice"] = float(data["selling_price"])
    if "min_stock_level" in data and data["min_stock_level"] is not None:
        inv_fields["minStockLevel"] = float(data["min_stock_level"])
    if "is_active" in data and data["is_active"] is not None:
        inv_fields["isActive"] = data["is_active"]

    if inv_fields:
        inv = await inventories_col.find_one({"store": store_oid, "product": prod_oid})
        if inv:
            new_qty = inv_fields.get("quantity", inv.get("quantity", 0.0))
            new_min = inv_fields.get("minStockLevel", inv.get("minStockLevel", 10.0))

            is_out_of_stock = new_qty == 0
            is_low_stock = (not is_out_of_stock) and (new_qty <= new_min)

            inv_fields["isOutOfStock"] = is_out_of_stock
            inv_fields["isLowStock"] = is_low_stock
            inv_fields["updatedAt"] = datetime.now(timezone.utc)

            await inventories_col.update_one(
                {"_id": inv["_id"]},
                {"$set": inv_fields}
            )

    LOGGER.info(
        "product_updated",
        product_id=str(prod_oid),
        product_name=prod_name,
        previous_qty=previous_qty,
        new_qty=new_calculated_qty,
        added_delta=added_delta,
    )
    return {
        "success": True,
        "product_id": str(prod_oid),
        "product_name": prod_name,
        "previous_quantity": previous_qty,
        "new_quantity": new_calculated_qty if new_calculated_qty is not None else previous_qty,
        "added_quantity": added_delta,
    }


async def delete_product(store_id: str, product_id: str) -> bool:
    """Soft deletes the product and inventory."""
    res = await update_product(store_id, product_id, {"is_active": False})
    return bool(res)
