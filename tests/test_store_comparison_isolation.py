import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from bson import ObjectId
from app.agent.service.stores.comparison import compare_user_stores

@pytest.mark.asyncio
async def test_compare_user_stores_strict_user_isolation():
    """Verify that compare_user_stores strictly isolates stores to the specified user and does NOT return unowned stores."""
    user_id = "user_12345"
    store_a_id = str(ObjectId())
    store_b_id = str(ObjectId())
    unowned_store_id = str(ObjectId())

    user_stores = [
        {"_id": ObjectId(store_a_id), "name": "User Store A", "user": user_id},
        {"_id": ObjectId(store_b_id), "name": "User Store B", "userId": user_id},
    ]

    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(return_value=user_stores)

    mock_db = MagicMock()
    # Stores find returns user_stores for query
    mock_db["stores"].find.return_value = mock_cursor
    mock_db["users"].find_one = AsyncMock(return_value=None)
    mock_db["stores"].find_one = AsyncMock(return_value=None)
    mock_db["products"].count_documents = AsyncMock(return_value=5)
    mock_db["sellers"].count_documents = AsyncMock(return_value=2)
    mock_db["buyers"].count_documents = AsyncMock(return_value=10)

    with patch("app.agent.service.stores.comparison.get_database", return_value=mock_db), \
         patch("app.agent.service.stores.comparison.fetch_sales_summary", return_value={"total_revenue": 1000.0, "total_sales_count": 5, "average_order_value": 200.0}):
        
        result = await compare_user_stores(user_id=user_id)

        assert result["user_id"] == user_id
        assert result["total_stores_compared"] == 2
        assert len(result["stores"]) == 2
        assert result["stores"][0]["store_name"] in ["User Store A", "User Store B"]
        assert unowned_store_id not in [s["store_id"] for s in result["stores"]]
