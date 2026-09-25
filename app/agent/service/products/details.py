from __future__ import annotations
from typing import Dict, Any
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.products.details")


async def fetch_product_details(store_id: str, product_id: str) -> Dict[str, Any]:
    """
    Fetch product details from MongoDB by ID, exact name, or SKU.
    """
    db = get_database()
    LOGGER.info("fetch_product_details_start", store_id=store_id, product_id=product_id)

    try:
        query_target = (product_id or "").strip()
        or_clauses: list[dict] = [
            {"name": {"$regex": f"^{query_target}$", "$options": "i"}},
            {"sku": query_target},
        ]
        try:
            or_clauses.append({"_id": ObjectId(query_target)})
        except (InvalidId, TypeError):
            pass

        doc = await db["products"].find_one({"$or": or_clauses})
        if doc:
            return {
                "product_id": str(doc["_id"]),
                "name": doc.get("name", "Unknown Product"),
                "category": doc.get("category", "General"),
                "description": doc.get("description", f"{doc.get('name')} in stock."),
                "price": float(doc.get("price", 0.0)),
                "quantity": float(doc.get("quantity", 0.0)),
                "sku": doc.get("sku", "N/A"),
            }
    except Exception as exc:
        LOGGER.error("fetch_product_details_db_error", store_id=store_id, product_id=product_id, error=str(exc))

    return {
        "product_id": product_id,
        "name": product_id,
        "category": "General",
        "description": f"Product '{product_id}' details",
        "price": 0.0,
        "quantity": 0.0,
        "sku": "N/A",
    }
