import re
from typing import Optional, Any, Dict
from bson import ObjectId
from bson.errors import InvalidId
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.billing.add_item")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    if not store_id:
        return None
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass
    doc = await db["stores"].find_one({"name": store_id}, {"_id": 1})
    if doc:
        return doc["_id"]
    return None


class AddBillingItemInput(BaseModel):
    product_name: str = Field(description="The name of the product to add to the bill.")
    quantity: int = Field(default=1, description="The quantity of the product to add.")
    store_id: str = Field(description="The store ID")


@tool("tool_add_billing_item", args_schema=AddBillingItemInput)
async def tool_add_billing_item(product_name: str, quantity: int = 1, store_id: str = "") -> dict:
    """Adds a product to the active billing cart (POS) by its name. This tool will search for the product and trigger the UI to add it to the bill. Note: the user MUST be on the billing page for this to work visually."""
    db = get_database()
    cleaned_name = (product_name or "").strip()
    LOGGER.info("tool_add_billing_item_start", product_name=cleaned_name, quantity=quantity, store_id=store_id)

    if not cleaned_name:
        return {"error": "Product name is required to add an item to the bill."}

    store_oid = await _resolve_store_id(db, store_id)

    # Base filter for active products
    store_filter = []
    if store_oid:
        store_filter.append({"store": store_oid})
    if store_id:
        store_filter.append({"store": store_id})

    base_match: Dict[str, Any] = {"isActive": {"$ne": False}}
    if store_filter:
        base_match["$or"] = store_filter

    # 1. Exact / Substring Regex Search
    escaped = re.escape(cleaned_name)
    query = {
        **base_match,
        "name": {"$regex": escaped, "$options": "i"}
    }
    product = await db["products"].find_one(query)

    # 2. Word Token Matching Search (e.g. 'Amul Cow Milk' -> words: 'Amul', 'Cow', 'Milk')
    if not product:
        words = [re.escape(w) for w in cleaned_name.split() if len(w) > 1]
        if words:
            word_query = {
                **base_match,
                "$and": [{"name": {"$regex": w, "$options": "i"}} for w in words]
            }
            product = await db["products"].find_one(word_query)

    # 3. Flexible OR Search across Name, Category, SKU, or Barcode
    if not product and len(cleaned_name) >= 3:
        flexible_query = {
            **base_match,
            "$or": [
                {"name": {"$regex": escaped, "$options": "i"}},
                {"category": {"$regex": escaped, "$options": "i"}},
                {"sku": {"$regex": escaped, "$options": "i"}},
                {"barcode": {"$regex": escaped, "$options": "i"}},
            ]
        }
        product = await db["products"].find_one(flexible_query)

    if not product:
        LOGGER.warning("tool_add_billing_item_not_found", product_name=cleaned_name, store_id=store_id)
        return {"error": f"Product '{cleaned_name}' not found in inventory."}

    barcode = product.get("barcode") or product.get("sku") or str(product.get("_id"))
    if not barcode:
        return {"error": f"Product '{product.get('name', cleaned_name)}' does not have a barcode or identifier. Cannot add to bill."}

    LOGGER.info(
        "tool_add_billing_item_success",
        product_name=product.get("name"),
        barcode=barcode,
        quantity=quantity,
    )

    return {
        "success": True,
        "message": f"Added {quantity} x {product.get('name', cleaned_name)} to the bill.",
        "barcode": barcode,
        "quantity": quantity,
        "product_id": str(product.get("_id", "")),
        "price": float(product.get("price") or product.get("sellingPrice") or 0.0),
    }

