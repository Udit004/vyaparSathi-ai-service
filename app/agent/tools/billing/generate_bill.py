from langchain_core.tools import tool
from pydantic import BaseModel, Field
import structlog

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.billing.generate_bill")

class GenerateBillInput(BaseModel):
    payment_method: str = Field(default="cash", description="The payment method (e.g. 'cash', 'upi', 'card').")

@tool("tool_generate_bill", args_schema=GenerateBillInput)
def tool_generate_bill(payment_method: str) -> dict:
    """Finalizes and generates the bill for the active cart (POS). Use this when the user is done adding items and wants to checkout. Note: the user MUST be on the billing page."""
    
    return {
        "success": True,
        "message": f"Bill generation triggered with payment method: {payment_method}",
        "payment_method": payment_method
    }
