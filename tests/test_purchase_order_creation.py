import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.agent.tools.purchases.write import tool_create_purchase


@pytest.mark.asyncio
async def test_tool_create_purchase():
    fake_record = {
        "success": True,
        "purchase_id": "60d5ec49f1b2c8b1f8e4e1a1",
        "invoice_number": "PO-20261003-ABCD",
        "seller_name": "Global Traders",
        "seller_id": "60d5ec49f1b2c8b1f8e4e1a2",
        "items_count": 1,
        "items": [
            {
                "product_name": "Tata Salt 1kg",
                "quantity": 20.0,
                "purchase_price": 28.0,
                "subtotal": 560.0,
            }
        ],
        "grand_total": 560.0,
        "paid_amount": 0.0,
        "due_amount": 560.0,
        "payment_status": "unpaid",
        "created_at": "2026-10-03T10:00:00Z",
    }

    with patch("app.agent.tools.purchases.write.create_purchase_order_record", new_callable=AsyncMock, return_value=fake_record), \
         patch("app.agent.tools.purchases.write.get_redis", return_value=None):

        res = await tool_create_purchase.ainvoke({
            "seller_name": "Global Traders",
            "items": [
                {
                    "product_name": "Tata Salt 1kg",
                    "quantity": 20,
                    "purchase_price": 28.0,
                }
            ],
            "paid_amount": 0.0,
            "payment_status": "unpaid",
            "store_id": "store_123",
            "user_id": "user_456",
        })

        assert res["success"] is True
        assert res["invoice_number"] == "PO-20261003-ABCD"
        assert res["grand_total"] == 560.0
        assert "Global Traders" in res["summary"]
        assert "Tata Salt 1kg" in res["summary"]
        assert res["stock_updated"] is False


@pytest.mark.asyncio
async def test_tool_receive_purchase():
    from app.agent.tools.purchases.write import tool_receive_purchase

    fake_receive_res = {
        "success": True,
        "purchase_id": "60d5ec49f1b2c8b1f8e4e1a1",
        "invoice_number": "PO-20261003-ABCD",
        "items_count": 1,
        "items": [{"product_name": "Tata Salt 1kg", "quantity_added": 20.0}],
        "message": "Successfully received 1 items from purchase order PO-20261003-ABCD and added them to store inventory!",
    }

    with patch("app.agent.tools.purchases.write.receive_purchase_order_stock", new_callable=AsyncMock, return_value=fake_receive_res):
        res = await tool_receive_purchase.ainvoke({
            "purchase_identifier": "PO-20261003-ABCD",
            "store_id": "store_123",
            "user_id": "user_456",
        })

        assert res["success"] is True
        assert res["invoice_number"] == "PO-20261003-ABCD"
        assert "Successfully received" in res["summary"]

