"""
app/agent/tools/buyers/write.py
===================================
LangChain tools for writing buyers.
"""
from typing import Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from app.agent.service.buyers.write import create_buyer, update_buyer, delete_buyer

class CreateBuyerSchema(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    name: str = Field(..., description="Buyer/Customer name")
    phone: str = Field(..., description="Phone number")
    email: Optional[str] = Field(None, description="Email address")
    address: Optional[str] = Field(None, description="Physical address")
    gstin: Optional[str] = Field(None, description="GSTIN number")

@tool(args_schema=CreateBuyerSchema)
async def tool_create_buyer(store_id: str, name: str, phone: str, email: str = None, address: str = None, gstin: str = None) -> str:
    """Create a new buyer (customer) profile."""
    if not store_id: return "Error: store_id is required"
        
    data = {"name": name, "phone": phone, "email": email, "address": address, "gstin": gstin}
    try:
        buyer_id = await create_buyer(store_id, data)
        return f"Successfully created buyer {name}. ID: {buyer_id}"
    except Exception as e:
        return f"Failed to create buyer: {str(e)}"

class UpdateBuyerSchema(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    buyer_id: str = Field(..., description="The ID of the buyer to update")
    name: Optional[str] = Field(None, description="Buyer name")
    phone: Optional[str] = Field(None, description="Phone number")
    status: Optional[str] = Field(None, description="'active' or 'inactive'")

@tool(args_schema=UpdateBuyerSchema)
async def tool_update_buyer(store_id: str, buyer_id: str, name: str = None, phone: str = None, status: str = None) -> str:
    """Update fields of an existing buyer."""
    if not store_id: return "Error: store_id is required"
        
    data = {}
    if name is not None: data["name"] = name
    if phone is not None: data["phone"] = phone
    if status is not None: data["status"] = status
    
    try:
        await update_buyer(store_id, buyer_id, data)
        return f"Successfully updated buyer {buyer_id}."
    except Exception as e:
        return f"Failed to update buyer: {str(e)}"

class DeleteBuyerSchema(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    buyer_id: str = Field(..., description="The ID of the buyer to delete")

@tool(args_schema=DeleteBuyerSchema)
async def tool_delete_buyer(store_id: str, buyer_id: str) -> str:
    """Delete (inactivate) a buyer."""
    if not store_id: return "Error: store_id is required"
    try:
        await delete_buyer(store_id, buyer_id)
        return f"Successfully deleted buyer {buyer_id}."
    except Exception as e:
        return f"Failed to delete buyer: {str(e)}"
