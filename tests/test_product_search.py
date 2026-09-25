import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.agent.service.products.search import search_products
from app.agent.service.products.details import fetch_product_details
from bson import ObjectId

@pytest.mark.asyncio
async def test_search_products_regex_match():
    mock_doc = {
        "_id": ObjectId("650000000000000000000001"),
        "name": "Aloe Vera Face Gel",
        "category": "Skincare",
        "price": 250.0,
        "quantity": 15.0,
        "sku": "AVG-001"
    }

    mock_cursor = MagicMock()
    mock_cursor.limit.return_value = mock_cursor
    mock_cursor.to_list = AsyncMock(return_value=[mock_doc])

    mock_db = MagicMock()
    mock_db["stores"].find_one = AsyncMock(return_value=None)
    mock_db["products"].find.return_value = mock_cursor

    with patch("app.agent.service.products.search.get_database", return_value=mock_db):
        results = await search_products("store123", "Aloe Vera")
        assert len(results) == 1
        assert results[0]["name"] == "Aloe Vera Face Gel"
        assert results[0]["category"] == "Skincare"
        assert results[0]["price"] == 250.0

@pytest.mark.asyncio
async def test_fetch_product_details_success():
    mock_doc = {
        "_id": ObjectId("650000000000000000000001"),
        "name": "Aloe Vera Face Gel",
        "category": "Skincare",
        "price": 250.0,
        "quantity": 15.0,
        "sku": "AVG-001"
    }

    mock_db = MagicMock()
    mock_db["products"].find_one = AsyncMock(return_value=mock_doc)

    with patch("app.agent.service.products.details.get_database", return_value=mock_db):
        details = await fetch_product_details("store123", "Aloe Vera Face Gel")
        assert details["name"] == "Aloe Vera Face Gel"
        assert details["price"] == 250.0
        assert details["sku"] == "AVG-001"
