"""
app/agent/tools/sales/goal_progress.py
=======================================
Tracks the store owner's live progress towards their personal business targets
(monthly revenue goal, target profit margin, annual growth targets),
calculating daily run-rates, forecast projections, and proactive tactical adjustments.
"""

from __future__ import annotations

import datetime
import calendar
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import structlog

from app.agent.service.sales.summary import fetch_sales_summary
from app.agent.service.sales.daily_trend import fetch_daily_sales_trend
from app.agent.tools.memory.search import get_owner_goals_and_preferences

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.goal_progress")


class GoalProgressInput(BaseModel):
    store_id: Optional[str] = Field(
        default=None,
        description="The ID of the store (automatically populated if omitted).",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="The store owner / user ID (automatically populated if omitted).",
    )


class GoalProgressOutput(BaseModel):
    monthly_sales_target_inr: float = Field(..., description="The store owner's target revenue for the month in ₹.")
    current_mtd_sales_inr: float = Field(..., description="Actual month-to-date sales achieved so far in ₹.")
    percentage_target_achieved: float = Field(..., description="Percentage of monthly goal completed (0-100%).")
    projected_month_end_sales_inr: float = Field(..., description="Estimated total sales by month end at current daily run rate.")
    pacing_status: str = Field(..., description="Status: 'AHEAD OF TARGET', 'ON TRACK', or 'BEHIND TARGET'.")
    daily_sales_required_inr: float = Field(..., description="Required daily sales to successfully hit the target.")
    current_average_daily_sales_inr: float = Field(..., description="Current achieved average daily sales.")
    target_margin_percent: float = Field(..., description="Owner's target profit margin percentage.")
    strategic_recommendations: List[str] = Field(default_factory=list, description="Concrete actions to ensure target is met.")
    calculated_at: str = Field(..., description="Timestamp of report generation.")


@tool("get_store_goal_progress_report", args_schema=GoalProgressInput)
async def get_store_goal_progress_report(
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Generate an in-depth progress report comparing live sales performance against the store owner's
    personal monthly revenue and profit goals. Computes daily run-rates, forecast projections,
    and pacing alerts.

    Use this when:
    - The merchant asks "Mera goal progress kaisa chal raha hai?", "Will I reach my target this month?", "Target status report".
    - Aligning inventory and sales strategy with the merchant's financial objectives.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    LOGGER.info("get_store_goal_progress_invoked", store_id=store_id, user_id=user_id)

    if not store_id:
        return {
            "monthly_sales_target_inr": 0.0,
            "current_mtd_sales_inr": 0.0,
            "percentage_target_achieved": 0.0,
            "projected_month_end_sales_inr": 0.0,
            "pacing_status": "UNKNOWN",
            "daily_sales_required_inr": 0.0,
            "current_average_daily_sales_inr": 0.0,
            "target_margin_percent": 18.0,
            "strategic_recommendations": ["Store ID missing from context."],
            "calculated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    now = datetime.datetime.now()
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    current_day = max(1, now.day)
    days_remaining = max(1, days_in_month - current_day)

    # 1. Fetch Owner Goals from Redis/Pinecone Memory
    target_revenue = 400000.0  # Default 4 Lakh if no custom number parsed
    target_margin = 18.0
    try:
        goals_data = await get_owner_goals_and_preferences.ainvoke({"store_id": store_id, "user_id": user_id})
        goals_list = goals_data.get("owner_goals", []) if isinstance(goals_data, dict) else []
        for g in goals_list:
            g_lower = g.lower()
            if "500000" in g_lower or "5 lakh" in g_lower or "5,00,000" in g_lower:
                target_revenue = 500000.0
            elif "400000" in g_lower or "4 lakh" in g_lower or "4,00,000" in g_lower:
                target_revenue = 400000.0
            elif "300000" in g_lower or "3 lakh" in g_lower or "3,00,000" in g_lower:
                target_revenue = 300000.0
            if "margin" in g_lower:
                for word in g_lower.split():
                    if "%" in word:
                        try:
                            target_margin = float(word.replace("%", ""))
                        except Exception:
                            pass
    except Exception as exc:
        LOGGER.warning("goal_progress_memory_error", error=str(exc))

    # 2. Fetch Live Sales Data
    mtd_sales = 0.0
    try:
        sales_summary = await fetch_sales_summary(store_id, days_lookback=30)
        mtd_sales = float(sales_summary.get("total_revenue", 0.0) if isinstance(sales_summary, dict) else 0.0)
    except Exception as exc:
        LOGGER.warning("goal_progress_sales_summary_error", error=str(exc))

    # Calculations
    pct_achieved = round((mtd_sales / max(1.0, target_revenue)) * 100, 1)
    daily_avg = mtd_sales / current_day
    projected_total = daily_avg * days_in_month
    remaining_to_target = max(0.0, target_revenue - mtd_sales)
    daily_required = remaining_to_target / days_remaining

    if projected_total >= target_revenue * 1.05:
        pacing = "AHEAD OF TARGET"
    elif projected_total >= target_revenue * 0.95:
        pacing = "ON TRACK"
    else:
        pacing = "BEHIND TARGET"

    recommendations = []
    if pacing == "BEHIND TARGET":
        recommendations.append(
            f"Increase daily sales from current average ₹{daily_avg:,.0f} to ₹{daily_required:,.0f} per day for the remaining {days_remaining} days."
        )
        recommendations.append("Launch weekend promotional combos on top-moving items to boost footfall.")
        recommendations.append("Ensure zero stockouts on top 5 revenue-generating products.")
    else:
        recommendations.append(f"Excellent pace! You are on track to achieve ~₹{projected_total:,.0f} by month end.")
        recommendations.append(f"Focus on maintaining the target profit margin of {target_margin}% without unnecessary discounting.")

    return {
        "monthly_sales_target_inr": target_revenue,
        "current_mtd_sales_inr": mtd_sales,
        "percentage_target_achieved": pct_achieved,
        "projected_month_end_sales_inr": round(projected_total, 2),
        "pacing_status": pacing,
        "daily_sales_required_inr": round(daily_required, 2),
        "current_average_daily_sales_inr": round(daily_avg, 2),
        "target_margin_percent": target_margin,
        "strategic_recommendations": recommendations,
        "calculated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
