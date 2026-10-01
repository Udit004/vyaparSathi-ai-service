"""
app/agent/tools/sellers/write.py
===================================
LangChain tools for writing sellers (suppliers).
"""
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app.agent.service.sellers.write import create_seller, update_seller, delete_seller

class CreateSellerSchema(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    name: str = Field(..., description="Seller/Supplier name")
    business_name: Optional[str] = Field(None, description="Business name")
    phone: str = Field(..., description="Phone number")
    email: Optional[str] = Field(None, description="Email address")
    address: Optional[str] = Field(None, description="Physical address")
    gstin: Optional[str] = Field(None, description="GSTIN number")

@tool(args_schema=CreateSellerSchema)
async def tool_create_seller(store_id: str, name: str, phone: str, business_name: str = None, email: str = None, address: str = None, gstin: str = None) -> str:
    """Create a new seller (supplier) profile."""
    if not store_id: return "Error: store_id is required"
        
    data = {"name": name, "business_name": business_name, "phone": phone, "email": email, "address": address, "gstin": gstin}
    try:
        seller_id = await create_seller(store_id, data)
        return f"Successfully created seller {name}. ID: {seller_id}"
    except Exception as e:
        return f"Failed to create seller: {str(e)}"

class UpdateSellerSchema(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    seller_id: str = Field(..., description="The ID of the seller to update")
    name: Optional[str] = Field(None, description="Seller name")
    business_name: Optional[str] = Field(None, description="Business name")
    phone: Optional[str] = Field(None, description="Phone number")
    status: Optional[str] = Field(None, description="'active' or 'inactive'")

@tool(args_schema=UpdateSellerSchema)
async def tool_update_seller(store_id: str, seller_id: str, name: str = None, business_name: str = None, phone: str = None, status: str = None) -> str:
    """Update fields of an existing seller."""
    if not store_id: return "Error: store_id is required"
        
    data = {}
    if name is not None: data["name"] = name
    if business_name is not None: data["business_name"] = business_name
    if phone is not None: data["phone"] = phone
    if status is not None: data["status"] = status
    
    try:
        await update_seller(store_id, seller_id, data)
        return f"Successfully updated seller {seller_id}."
    except Exception as e:
        return f"Failed to update seller: {str(e)}"

class DeleteSellerSchema(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    seller_id: str = Field(..., description="The ID of the seller to delete")

@tool(args_schema=DeleteSellerSchema)
async def tool_delete_seller(store_id: str, seller_id: str) -> str:
    """Delete (inactivate) a seller."""
    if not store_id: return "Error: store_id is required"
    try:
        await delete_seller(store_id, seller_id)
        return f"Successfully deleted seller {seller_id}."
    except Exception as e:
        return f"Failed to delete seller: {str(e)}"
