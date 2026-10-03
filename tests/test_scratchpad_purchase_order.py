import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.agent.tools.memory.merchant_scratchpad import (
    write_scratchpad_note,
    read_scratchpad_notes,
    update_scratchpad_note,
    delete_scratchpad_note,
)
from app.agent.tools.forecast.smart_purchase_order import create_smart_purchase_order
from app.agent.tools.communication.send_email import send_store_email


@pytest.mark.asyncio
async def test_scratchpad_lifecycle():
    fake_store = {}

    class FakeRedis:
        async def get(self, key):
            return fake_store.get(key)

        async def set(self, key, value, ex=None):
            fake_store[key] = value

    with patch("app.agent.tools.memory.merchant_scratchpad.get_redis", return_value=FakeRedis()):
        # 1. Write Note
        w_res = await write_scratchpad_note.ainvoke({
            "title": "Test Restock Note",
            "content": "Need 50 packets of Parle-G",
            "category": "todo",
            "store_id": "store_123",
            "user_id": "user_456",
        })
        assert w_res["success"] is True
        note_id = w_res["note"]["note_id"]

        # 2. Read Note
        r_res = await read_scratchpad_notes.ainvoke({
            "store_id": "store_123",
            "user_id": "user_456",
        })
        assert r_res["found"] is True
        assert len(r_res["notes"]) == 1
        assert r_res["notes"][0]["note_id"] == note_id

        # 3. Update Note
        u_res = await update_scratchpad_note.ainvoke({
            "note_id": note_id,
            "updated_content": "Need 100 packets of Parle-G",
            "status": "completed",
            "store_id": "store_123",
            "user_id": "user_456",
        })
        assert u_res["success"] is True
        assert u_res["note"]["content"] == "Need 100 packets of Parle-G"
        assert u_res["note"]["status"] == "completed"

        # 4. Delete Note
        d_res = await delete_scratchpad_note.ainvoke({
            "note_id": note_id,
            "store_id": "store_123",
            "user_id": "user_456",
        })
        assert d_res["success"] is True


@pytest.mark.asyncio
async def test_create_smart_purchase_order():
    fake_priorities = [
        {
            "name": "Tata Salt 1kg",
            "current_quantity": 4,
            "avg_daily_sales": 3.0,
            "priority": "RED",
            "suggested_restock_quantity": 30,
            "price": 28.0,
            "supplier_name": "Laxmi Distributors",
        }
    ]

    with patch("app.agent.tools.forecast.smart_purchase_order.fetch_restock_priorities", new_callable=AsyncMock) as mock_prio, \
         patch("app.agent.tools.forecast.smart_purchase_order.fetch_low_stock_products", new_callable=AsyncMock) as mock_low, \
         patch("app.agent.tools.forecast.smart_purchase_order.get_redis", return_value=None):

        mock_prio.return_value = fake_priorities
        mock_low.return_value = []

        po = await create_smart_purchase_order.ainvoke({
            "days_lead_time": 3,
            "buffer_days": 7,
            "auto_save_to_scratchpad": False,
            "store_id": "store_123",
            "user_id": "user_456",
        })

        assert po["total_items_count"] == 1
        assert "PO-" in po["po_number"]
        assert po["total_units_ordered"] >= 10
        assert "Tata Salt 1kg" in po["voice_summary"]
        assert "PURCHASE ORDER" in po["email_ready_html"]


@pytest.mark.asyncio
async def test_send_store_email():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"success": True, "message": "Email sent successfully"}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post, \
         patch("app.agent.tools.communication.send_email.get_redis", return_value=None):

        mock_post.return_value = mock_response

        res = await send_store_email.ainvoke({
            "recipient_email": "supplier@example.com",
            "subject": "Purchase Order PO-2026-001",
            "body_html": "<p>Please find the purchase order attached.</p>",
            "store_id": "store_123",
        })

        assert res["success"] is True
        assert res["recipient"] == "supplier@example.com"
        assert res["status"] == "SENT"
