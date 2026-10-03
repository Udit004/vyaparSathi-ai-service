import pytest
from unittest.mock import AsyncMock, patch
from app.agent.tools.sellers.search import search_sellers
from app.agent.tools.purchases.search import search_purchases


@pytest.mark.asyncio
async def test_search_sellers_tool():
    fake_sellers = [
        {
            "seller_id": "sel_1",
            "name": "Ramesh Tiwari",
            "business_name": "Global Traders",
            "phone": "9876543210",
            "email": "global@example.com",
            "address": "Delhi Market",
            "gstin": "07AAAAA0000A1Z5",
            "total_purchase": 50000.0,
            "total_paid": 40000.0,
            "total_due": 10000.0,
            "status": "active",
        }
    ]

    with patch("app.agent.tools.sellers.search.fetch_sellers", new_callable=AsyncMock, return_value=fake_sellers):
        res = await search_sellers.ainvoke({
            "query": "Global",
            "store_id": "store_123",
            "user_id": "user_456",
        })

        assert res["found"] is True
        assert res["count"] == 1
        assert res["sellers"][0]["business_name"] == "Global Traders"
        assert "Global Traders" in res["summary"]


@pytest.mark.asyncio
async def test_search_purchases_tool():
    fake_purchases = [
        {
            "purchase_id": "pur_1",
            "invoice_number": "INV-2026-001",
            "seller_name": "Global Traders",
            "seller_id": "sel_1",
            "purchase_date": "2026-10-01",
            "total_amount": 15000.0,
            "paid_amount": 10000.0,
            "due_amount": 5000.0,
            "payment_status": "partial",
            "items_count": 2,
            "items": [
                {
                    "product_name": "Wheat Flour 10kg",
                    "quantity": 20.0,
                    "purchase_price": 400.0,
                    "total_cost": 8000.0,
                }
            ],
        }
    ]

    with patch("app.agent.tools.purchases.search.fetch_purchases", new_callable=AsyncMock, return_value=fake_purchases):
        res = await search_purchases.ainvoke({
            "seller_name": "Global Traders",
            "store_id": "store_123",
            "user_id": "user_456",
        })

        assert res["found"] is True
        assert res["count"] == 1
        assert res["total_spend"] == 15000.0
        assert res["total_due"] == 5000.0
        assert res["purchases"][0]["invoice_number"] == "INV-2026-001"
