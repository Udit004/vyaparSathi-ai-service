import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId
from app.agent.service.purchases.write import (
    update_purchase_order_record,
    delete_purchase_order_record,
)
from app.agent.tools.purchases.write import tool_update_purchase, tool_delete_purchase


def _create_mock_purchase_db(purchase_doc, seller_doc=None, prod_doc=None):
    mock_purchases = MagicMock()
    mock_purchases.find_one = AsyncMock(return_value=purchase_doc)
    mock_purchases.update_one = AsyncMock()
    mock_purchases.delete_one = AsyncMock()

    mock_sellers = MagicMock()
    mock_sellers.find_one = AsyncMock(return_value=seller_doc)
    mock_sellers.update_one = AsyncMock()

    mock_products = MagicMock()
    mock_products.find_one = AsyncMock(return_value=prod_doc)
    mock_products.update_one = AsyncMock()

    mock_inventories = MagicMock()
    mock_inventories.find_one = AsyncMock(return_value=None)
    mock_inventories.update_one = AsyncMock()

    mock_stores = MagicMock()
    mock_stores.find_one = AsyncMock(return_value=None)

    mock_db = MagicMock()
    mock_db.__getitem__.side_effect = lambda k: {
        "purchases": mock_purchases,
        "sellers": mock_sellers,
        "products": mock_products,
        "inventories": mock_inventories,
        "stores": mock_stores,
    }.get(k, MagicMock())

    return mock_db, mock_purchases, mock_sellers, mock_products


@pytest.mark.asyncio
async def test_update_purchase_order_by_invoice_number():
    store_oid = ObjectId("650000000000000000000099")
    purchase_oid = ObjectId("650000000000000000000001")
    seller_oid = ObjectId("650000000000000000000002")

    mock_purchase = {
        "_id": purchase_oid,
        "store": store_oid,
        "seller": seller_oid,
        "invoiceNumber": "PO-20261003-ADF4",
        "grandTotal": 10000.0,
        "paidAmount": 2000.0,
        "dueAmount": 8000.0,
        "paymentStatus": "partial",
        "items": [],
    }

    mock_seller = {
        "_id": seller_oid,
        "name": "Tiwari Traders",
        "businessName": "Tiwari Traders",
    }

    mock_db, mock_purchases, mock_sellers, _ = _create_mock_purchase_db(mock_purchase, mock_seller)

    with patch("app.agent.service.purchases.write.get_database", return_value=mock_db):
        res = await update_purchase_order_record(
            store_id=str(store_oid),
            purchase_identifier="PO-20261003-ADF4",
            paid_amount=10000.0,
        )

        assert res["success"] is True
        assert res["invoice_number"] == "PO-20261003-ADF4"
        assert res["paid_amount"] == 10000.0
        assert res["due_amount"] == 0.0
        assert res["payment_status"] == "paid"

        # Verify update_one on purchases
        call_args = mock_purchases.update_one.call_args[0]
        assert call_args[0] == {"_id": purchase_oid}
        assert call_args[1]["$set"]["paidAmount"] == 10000.0
        assert call_args[1]["$set"]["dueAmount"] == 0.0
        assert call_args[1]["$set"]["paymentStatus"] == "paid"

        # Verify seller balance update
        seller_call = mock_sellers.update_one.call_args[0][1]
        assert seller_call["$inc"]["totalPaid"] == 8000.0  # 10000 - 2000
        assert seller_call["$inc"]["totalDue"] == -8000.0  # 0 - 8000


@pytest.mark.asyncio
async def test_tool_update_purchase_invocation():
    store_oid = ObjectId("650000000000000000000099")
    purchase_oid = ObjectId("650000000000000000000001")
    seller_oid = ObjectId("650000000000000000000002")

    mock_purchase = {
        "_id": purchase_oid,
        "store": store_oid,
        "seller": seller_oid,
        "invoiceNumber": "PO-20261003-ADF4",
        "grandTotal": 5000.0,
        "paidAmount": 0.0,
        "dueAmount": 5000.0,
        "paymentStatus": "unpaid",
        "items": [],
    }
    mock_seller = {
        "_id": seller_oid,
        "name": "Tiwari Traders",
    }

    mock_db, _, _, _ = _create_mock_purchase_db(mock_purchase, mock_seller)

    with patch("app.agent.service.purchases.write.get_database", return_value=mock_db):
        res = await tool_update_purchase.ainvoke({
            "store_id": str(store_oid),
            "purchase_identifier": "PO-20261003-ADF4",
            "paid_amount": 5000.0,
            "notes": "Full payment made via UPI",
        })

        assert res["success"] is True
        assert res["invoice_number"] == "PO-20261003-ADF4"
        assert res["payment_status"] == "paid"
        assert "Successfully updated purchase order PO-20261003-ADF4" in res["summary"]


@pytest.mark.asyncio
async def test_tool_delete_purchase_invocation():
    store_oid = ObjectId("650000000000000000000099")
    purchase_oid = ObjectId("650000000000000000000001")
    seller_oid = ObjectId("650000000000000000000002")
    prod_oid = ObjectId("650000000000000000000003")

    mock_purchase = {
        "_id": purchase_oid,
        "store": store_oid,
        "seller": seller_oid,
        "invoiceNumber": "PO-20261003-ADF4",
        "grandTotal": 4000.0,
        "paidAmount": 4000.0,
        "dueAmount": 0.0,
        "items": [{"product": prod_oid, "quantity": 20}],
    }

    mock_db, mock_purchases, mock_sellers, mock_products = _create_mock_purchase_db(mock_purchase)

    with patch("app.agent.service.purchases.write.get_database", return_value=mock_db):
        res = await tool_delete_purchase.ainvoke({
            "store_id": str(store_oid),
            "purchase_identifier": "PO-20261003-ADF4",
        })

        assert res["success"] is True
        assert res["invoice_number"] == "PO-20261003-ADF4"
        assert "Successfully deleted" in res["summary"]

        # Check purchase deleted
        mock_purchases.delete_one.assert_called_once_with({"_id": purchase_oid, "store": store_oid})

        # Check stock reverted
        prod_call = mock_products.update_one.call_args[0][1]
        assert prod_call["$inc"]["quantity"] == -20

        # Check seller balance reverted
        seller_call = mock_sellers.update_one.call_args[0][1]
        assert seller_call["$inc"]["totalPurchase"] == -4000.0
        assert seller_call["$inc"]["totalPaid"] == -4000.0
