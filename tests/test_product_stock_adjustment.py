import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from bson import ObjectId
from app.agent.service.products.write import update_product
from app.agent.tools.products.write import tool_adjust_stock, tool_update_product


def _create_mock_db(prod_doc, inv_doc=None):
    mock_products = MagicMock()
    mock_products.find_one = AsyncMock(return_value=prod_doc)
    mock_products.update_one = AsyncMock()

    mock_inventories = MagicMock()
    mock_inventories.find_one = AsyncMock(return_value=inv_doc)
    mock_inventories.update_one = AsyncMock()

    mock_stores = MagicMock()
    mock_stores.find_one = AsyncMock(return_value=None)

    mock_db = MagicMock()
    mock_db.__getitem__.side_effect = lambda k: {
        "products": mock_products,
        "inventories": mock_inventories,
        "stores": mock_stores,
    }.get(k, MagicMock())
    return mock_db, mock_products, mock_inventories


@pytest.mark.asyncio
async def test_update_product_relative_add_quantity():
    prod_oid = ObjectId("650000000000000000000001")
    store_oid = ObjectId("650000000000000000000099")

    mock_prod = {
        "_id": prod_oid,
        "store": store_oid,
        "name": "Coca-Cola 750ml",
        "quantity": 10.0,
        "sellingPrice": 45.0,
    }
    mock_inv = {
        "_id": ObjectId("650000000000000000000002"),
        "store": store_oid,
        "product": prod_oid,
        "quantity": 10.0,
        "minStockLevel": 5.0,
    }

    mock_db, mock_products, _ = _create_mock_db(mock_prod, mock_inv)

    with patch("app.agent.service.products.write.get_database", return_value=mock_db):
        res = await update_product(str(store_oid), str(prod_oid), {"add_quantity": 20})
        assert res["success"] is True
        assert res["previous_quantity"] == 10.0
        assert res["new_quantity"] == 30.0
        assert res["added_quantity"] == 20.0

        prod_update_call = mock_products.update_one.call_args[0][1]
        assert prod_update_call["$set"]["quantity"] == 30.0


@pytest.mark.asyncio
async def test_tool_adjust_stock_by_name():
    prod_oid = ObjectId("650000000000000000000001")
    store_oid = ObjectId("650000000000000000000099")

    mock_prod = {
        "_id": prod_oid,
        "store": store_oid,
        "name": "Tata Salt",
        "quantity": 5.0,
    }

    mock_db, _, _ = _create_mock_db(mock_prod)

    with patch("app.agent.service.products.write.get_database", return_value=mock_db):
        out = await tool_adjust_stock.ainvoke({
            "store_id": str(store_oid),
            "product_name": "Tata Salt",
            "quantity_to_add": 15,
        })
        assert "Successfully added to Tata Salt" in out
        assert "Previous stock: 5.0" in out
        assert "New total stock: 20.0" in out


@pytest.mark.asyncio
async def test_tool_update_product_override_vs_add():
    prod_oid = ObjectId("650000000000000000000001")
    store_oid = ObjectId("650000000000000000000099")

    mock_prod = {
        "_id": prod_oid,
        "store": store_oid,
        "name": "Parle-G 100g",
        "quantity": 12.0,
    }

    mock_db, _, _ = _create_mock_db(mock_prod)

    with patch("app.agent.service.products.write.get_database", return_value=mock_db):
        # Adding stock
        out_add = await tool_update_product.ainvoke({
            "store_id": str(store_oid),
            "product_name": "Parle-G 100g",
            "add_quantity": 25,
        })
        assert "Previous stock: 12.0" in out_add
        assert "New total stock: 37.0" in out_add

        # Setting explicit absolute count
        out_set = await tool_update_product.ainvoke({
            "store_id": str(store_oid),
            "product_name": "Parle-G 100g",
            "quantity": 50,
        })
        assert "Stock count set to 50.0" in out_set
