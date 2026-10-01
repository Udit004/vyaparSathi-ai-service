"""
app/agent/tools/products/write.py
===================================
LangChain tools for writing products.
"""
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app.agent.service.products.write import create_product, update_product, delete_product

class CreateProductSchema(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    user_id: str = Field(..., description="The ID of the user performing the action.")
    name: str = Field(..., description="Product name")
    category: str = Field(..., description="Category name (e.g. 'Electronics', 'General')")
    selling_price: float = Field(..., description="Selling price (Rate)")
    buying_price: float = Field(..., description="Buying/Cost price")
    quantity: float = Field(0, description="Initial stock quantity")
    brand: Optional[str] = Field(None, description="Brand name")
    barcode: Optional[str] = Field(None, description="Barcode/SKU")
    sku: Optional[str] = Field(None, description="SKU if different from barcode")
    unit: Optional[str] = Field("Pieces", description="Unit of measurement")
    min_stock_level: Optional[float] = Field(10, description="Alert threshold for low stock")
    exp_date: Optional[str] = Field(None, description="Expiry date in ISO format (YYYY-MM-DD)")

@tool(args_schema=CreateProductSchema)
async def tool_create_product(store_id: str, user_id: str, name: str, category: str, selling_price: float, buying_price: float,
                              quantity: float = 0, brand: str = None, barcode: str = None, sku: str = None,
                              unit: str = "Pieces", min_stock_level: float = 10, exp_date: str = None) -> str:
    """Create a new product in the store's inventory catalog."""
    if not store_id:
        return "Error: store_id is required"
        
    data = {
        "name": name, "category": category, "selling_price": selling_price,
        "buying_price": buying_price, "quantity": quantity, "brand": brand,
        "barcode": barcode, "sku": sku, "unit": unit, "min_stock_level": min_stock_level,
        "exp_date": exp_date
    }
    
    try:
        product_id = await create_product(store_id, data, user_id)
        return f"Successfully created product {name}. ID: {product_id}"
    except Exception as e:
        return f"Failed to create product: {str(e)}"


class UpdateProductSchema(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    product_id: str = Field(..., description="The ID of the product to update")
    name: Optional[str] = Field(None, description="Product name")
    category: Optional[str] = Field(None, description="Category name")
    selling_price: Optional[float] = Field(None, description="Selling price")
    buying_price: Optional[float] = Field(None, description="Buying/Cost price")
    quantity: Optional[float] = Field(None, description="Override stock quantity")
    min_stock_level: Optional[float] = Field(None, description="Alert threshold")
    is_active: Optional[bool] = Field(None, description="Set to false to soft delete")

@tool(args_schema=UpdateProductSchema)
async def tool_update_product(store_id: str, product_id: str, name: str = None, category: str = None,
                              selling_price: float = None, buying_price: float = None,
                              quantity: float = None, min_stock_level: float = None,
                              is_active: bool = None) -> str:
    """Update fields of an existing product."""
    if not store_id:
        return "Error: store_id is required"
        
    data = {}
    if name is not None: data["name"] = name
    if category is not None: data["category"] = category
    if selling_price is not None: data["selling_price"] = selling_price
    if buying_price is not None: data["buying_price"] = buying_price
    if quantity is not None: data["quantity"] = quantity
    if min_stock_level is not None: data["min_stock_level"] = min_stock_level
    if is_active is not None: data["is_active"] = is_active
    
    if not data:
        return "No fields to update provided."
        
    try:
        await update_product(store_id, product_id, data)
        return f"Successfully updated product {product_id}."
    except Exception as e:
        return f"Failed to update product: {str(e)}"
        
class DeleteProductSchema(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    product_id: str = Field(..., description="The ID of the product to delete")
    
@tool(args_schema=DeleteProductSchema)
async def tool_delete_product(store_id: str, product_id: str) -> str:
    """Soft delete a product from the catalog."""
    if not store_id:
        return "Error: store_id is required"
    try:
        await delete_product(store_id, product_id)
        return f"Successfully deleted product {product_id}."
    except Exception as e:
        return f"Failed to delete product: {str(e)}"
