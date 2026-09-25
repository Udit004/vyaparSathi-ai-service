from __future__ import annotations
import re
from typing import List, Dict, Any
from bson import ObjectId
from bson.errors import InvalidId
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.service.products.search")


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    """
    Resolve a store identifier to an ObjectId.
    Accepts either a valid 24-char hex ObjectId string or a store name.
    """
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


async def search_products(store_id: str, query: str) -> List[Dict[str, Any]]:
    """
    Search for products in MongoDB by name, category, or SKU using case-insensitive regex matching.
    """
    db = get_database()
    query_str = (query or "").strip()
    LOGGER.info("search_products_start", store_id=store_id, query=query_str)

    results: List[Dict[str, Any]] = []

    try:
        store_oid = await _resolve_store_id(db, store_id)

        # Base filter
        base_match: Dict[str, Any] = {"isActive": {"$ne": False}}
        if store_oid:
            base_match["store"] = store_oid

        # Build regex criteria
        if query_str:
            regex_escaped = re.escape(query_str)
            words = [re.escape(w) for w in query_str.split() if w]
            word_patterns = [{"name": {"$regex": w, "$options": "i"}} for w in words]

            or_conditions: List[Dict[str, Any]] = [
                {"name": {"$regex": regex_escaped, "$options": "i"}},
                {"category": {"$regex": regex_escaped, "$options": "i"}},
                {"sku": {"$regex": regex_escaped, "$options": "i"}},
            ]
            if len(word_patterns) > 1:
                or_conditions.append({"$and": word_patterns})

            search_filter = {"$and": [base_match, {"$or": or_conditions}]}
        else:
            search_filter = base_match

        cursor = db["products"].find(
            search_filter,
            {"_id": 1, "name": 1, "category": 1, "price": 1, "quantity": 1, "sku": 1},
        ).limit(20)

        docs = await cursor.to_list(length=20)

        # Fallback: If store filter produced 0 results, try matching without store filter
        if not docs and query_str and store_oid:
            fallback_filter = {"isActive": {"$ne": False}, "$or": or_conditions}
            cursor = db["products"].find(
                fallback_filter,
                {"_id": 1, "name": 1, "category": 1, "price": 1, "quantity": 1, "sku": 1},
            ).limit(20)
            docs = await cursor.to_list(length=20)

        for doc in docs:
            results.append({
                "product_id": str(doc["_id"]),
                "name": doc.get("name", "Unknown Product"),
                "category": doc.get("category", "General"),
                "price": float(doc.get("price", 0.0)),
            })

    except Exception as exc:
        LOGGER.error("search_products_db_error", store_id=store_id, query=query_str, error=str(exc))

    LOGGER.info("search_products_result", store_id=store_id, query=query_str, count=len(results))
    return results
