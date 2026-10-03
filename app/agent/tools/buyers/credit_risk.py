"""
app/agent/tools/buyers/credit_risk.py
======================================
Analyzes customer credit (Udhaar) risk, identifies overdue accounts,
calculates working capital trapped in receivables, and generates ready-to-send
polite payment reminders in Hindi and English.
"""

from __future__ import annotations

import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import structlog

from app.agent.service.buyers.dues import fetch_buyer_dues

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.credit_risk")


class CustomerCreditRiskInput(BaseModel):
    store_id: Optional[str] = Field(
        default=None,
        description="The ID of the store (automatically populated if omitted).",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="The store owner / user ID (automatically populated if omitted).",
    )
    min_due_threshold: float = Field(
        default=100.0,
        ge=0.0,
        description="Minimum outstanding due amount to analyze (default: ₹100).",
    )


class DebtorRiskProfile(BaseModel):
    customer_name: str = Field(..., description="Customer's full name.")
    phone: str = Field(..., description="Contact phone number.")
    total_due: float = Field(..., description="Outstanding unpaid credit balance in ₹.")
    total_purchases: float = Field(..., description="Lifetime purchase volume in ₹.")
    risk_level: str = Field(..., description="Risk category: HIGH, MODERATE, or LOW.")
    collection_urgency: str = Field(..., description="Recommended collection urgency.")
    suggested_whatsapp_msg_hinglish: str = Field(..., description="Polite WhatsApp reminder in Hinglish.")
    suggested_whatsapp_msg_hindi: str = Field(..., description="Polite WhatsApp reminder in formal Hindi.")


class CustomerCreditRiskOutput(BaseModel):
    total_outstanding_amount: float = Field(..., description="Total amount of cash currently trapped in customer credit.")
    total_debtors_count: int = Field(..., description="Number of customers with active outstanding balance.")
    high_risk_amount: float = Field(..., description="Amount owed by high risk / high value debtors.")
    risk_profiles: List[DebtorRiskProfile] = Field(default_factory=list, description="Detailed debtor risk profiles.")
    actionable_recovery_strategy: str = Field(..., description="Strategic recommendations to improve cashflow and recover udhaar.")
    analyzed_at: str = Field(..., description="Timestamp of analysis.")


@tool("analyze_customer_credit_risk", args_schema=CustomerCreditRiskInput)
async def analyze_customer_credit_risk(
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    min_due_threshold: float = 100.0,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Analyze customer credit (Udhaar) risk across the store's customer base.
    Identifies major debtors, calculates total working capital locked in credit,
    and automatically crafts polite, personalized WhatsApp payment reminders in Hindi and Hinglish.

    Use this when:
    - The store owner asks "Kitna udhaar market me fasa hai?", "Kaun kaun paise nahi de raha?", "Udhaar recovery list".
    - The merchant wants ready-made WhatsApp reminder messages for pending customer bills.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    LOGGER.info("analyze_customer_credit_risk_invoked", store_id=store_id, user_id=user_id, min_due=min_due_threshold)

    if not store_id:
        return {
            "total_outstanding_amount": 0.0,
            "total_debtors_count": 0,
            "high_risk_amount": 0.0,
            "risk_profiles": [],
            "actionable_recovery_strategy": "Store ID not available in context.",
            "analyzed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    try:
        dues_data = await fetch_buyer_dues(store_id, min_due=min_due_threshold, limit=50)
        raw_buyers = dues_data.get("buyers", []) if isinstance(dues_data, dict) else []
        total_outstanding = float(dues_data.get("total_outstanding", 0.0))
    except Exception as exc:
        LOGGER.warning("credit_risk_fetch_error", error=str(exc))
        raw_buyers = []
        total_outstanding = 0.0

    profiles: List[Dict[str, Any]] = []
    high_risk_total = 0.0

    for b in raw_buyers:
        name = b.get("name", "Customer")
        phone = b.get("phone", "")
        due = float(b.get("total_due", 0.0))
        sales = float(b.get("total_sales", 0.0))

        # Risk classification
        if due >= 2000.0 or (sales > 0 and (due / sales) > 0.6):
            risk = "HIGH"
            urgency = "Immediate follow-up required (within 24-48 hrs)"
            high_risk_total += due
        elif due >= 500.0:
            risk = "MODERATE"
            urgency = "Follow-up before next weekend"
        else:
            risk = "LOW"
            urgency = "Gentle reminder during next visit"

        # WhatsApp reminders
        msg_hinglish = (
            f"Namaste {name} ji, aapke store bill ka pending balance ₹{due:,.2f} hai. "
            f"Kripya suvidha anusar payment UPI/Cash se clear kar dein. Dhanyawad!"
        )
        msg_hindi = (
            f"नमस्ते {name} जी, आपकी दुकान के खाते का शेष बकाया ₹{due:,.2f} है। "
            f"कृपया समय मिलते ही इसका भुगतान करने की कृपा करें। धन्यवाद।"
        )

        profiles.append({
            "customer_name": name,
            "phone": phone,
            "total_due": due,
            "total_purchases": sales,
            "risk_level": risk,
            "collection_urgency": urgency,
            "suggested_whatsapp_msg_hinglish": msg_hinglish,
            "suggested_whatsapp_msg_hindi": msg_hindi,
        })

    # Sort high risk first
    profiles.sort(key=lambda x: x["total_due"], reverse=True)

    strategy = (
        f"Total ₹{total_outstanding:,.2f} is currently locked in customer credit across {len(profiles)} accounts. "
        f"Focus immediately on the top {min(3, len(profiles))} debtors representing ₹{high_risk_total:,.2f}. "
        f"Use the generated WhatsApp reminders to initiate polite collection without hurting customer relationships."
    )

    return {
        "total_outstanding_amount": total_outstanding,
        "total_debtors_count": len(profiles),
        "high_risk_amount": high_risk_total,
        "risk_profiles": profiles[:10],
        "actionable_recovery_strategy": strategy,
        "analyzed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
