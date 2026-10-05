from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import structlog
from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.billing.add_item")

class AddBillingItemInput(BaseModel):
    product_name: str = Field(description="The name of the product to add to the bill.")
    quantity: int = Field(default=1, description="The quantity of the product to add.")
    store_id: str = Field(description="The store ID")

@tool("tool_add_billing_item", args_schema=AddBillingItemInput)
async def tool_add_billing_item(product_name: str, quantity: int, store_id: str) -> dict:
    """Adds a product to the active billing cart (POS) by its name. This tool will search for the product and trigger the UI to add it to the bill. Note: the user MUST be on the billing page for this to work visually."""
    db = get_database()
    
    # Simple regex search for the product name
    query = {
        "store": store_id,
        "name": {"$regex": product_name, "$options": "i"}
    }
    
    product = await db.products.find_one(query)
    
    if not product:
        return {"error": f"Product '{product_name}' not found in inventory."}
        
    barcode = product.get("barcode")
    if not barcode:
        return {"error": f"Product '{product_name}' does not have a barcode. Cannot add to bill."}
        
    return {
        "success": True,
        "message": f"Added {quantity} x {product['name']} to the bill.",
        "barcode": barcode,
        "quantity": quantity
    }
