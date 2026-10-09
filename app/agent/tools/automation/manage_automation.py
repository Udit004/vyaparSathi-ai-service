"""
app/agent/tools/automation/manage_automation.py
==============================================
Agent tools to create, list, toggle, trigger, and delete automated business workflows.
Communicates with Express backend (/api/automations) to register BullMQ scheduled jobs
(Daily Low-Stock alerts, Daily/Weekly Sales summaries, AI Demand forecasts, Excel exports).
"""

from __future__ import annotations

import os
import re
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from bson import ObjectId
import httpx
import structlog

from app.config.database import get_database

LOGGER = structlog.get_logger("vyaparsathi.ai.tools.automation")

_EXPRESS_URL_ENV = "EXPRESS_BACKEND_URL"
_DEFAULT_EXPRESS_URL = "http://localhost:5000"


async def _resolve_store_and_user(store_id: str, user_id: str) -> tuple[Optional[ObjectId], Optional[ObjectId], str, str]:
    """Resolves store ObjectId, user ObjectId, store name, and owner email."""
    db = get_database()
    store_oid = None
    user_oid = None
    store_name = "Vyapar Store"
    owner_email = ""

    # 1. Resolve Store
    if store_id:
        if ObjectId.is_valid(store_id):
            s_doc = await db["stores"].find_one({"_id": ObjectId(store_id)})
        else:
            s_doc = await db["stores"].find_one({"name": store_id})
        
        if s_doc:
            store_oid = s_doc.get("_id")
            store_name = s_doc.get("name") or s_doc.get("storeName") or store_name
            owner_ref = s_doc.get("owner")
            owner_fb = s_doc.get("ownerFirebaseUid")
            if owner_ref and ObjectId.is_valid(str(owner_ref)):
                user_oid = ObjectId(str(owner_ref)) if not isinstance(owner_ref, ObjectId) else owner_ref

    # 2. Resolve User
    user_clauses = []
    if user_id:
        if ObjectId.is_valid(user_id):
            user_clauses.append({"_id": ObjectId(user_id)})
        user_clauses.extend([
            {"firebaseUid": user_id},
            {"uid": user_id},
            {"_id": user_id},
            {"email": user_id},
        ])
    if user_oid and {"_id": user_oid} not in user_clauses:
        user_clauses.append({"_id": user_oid})

    if user_clauses:
        try:
            u_doc = await db["users"].find_one({"$or": user_clauses})
            if u_doc:
                user_oid = u_doc.get("_id")
                owner_email = u_doc.get("email") or ""
        except Exception:
            pass

    return store_oid, user_oid, store_name, owner_email


def _get_express_headers(user_oid: Optional[ObjectId], user_email: str) -> dict:
    internal_key = os.getenv("INTERNAL_SERVICE_KEY", "vyapar-internal-ai-service-key")
    return {
        "x-internal-service-key": internal_key,
        "x-service-name": "vyapar-ai-service",
        "Authorization": f"Bearer {internal_key}",
        "x-user-id": str(user_oid) if user_oid else "",
        "x-user-email": user_email or "",
    }


# ===========================================================================
# Tool 1: Create Automation Rule
# ===========================================================================

class CreateAutomationInput(BaseModel):
    title: str = Field(
        ...,
        description="Descriptive title for the automation rule (e.g. 'Daily 9 AM Low Stock Email Alert', 'Weekly Monday Sales Summary Report').",
    )
    task_type: Literal[
        "LOW_STOCK_ALERT",
        "SALES_SUMMARY",
        "AI_DEMAND_FORECAST",
        "EXCEL_REPORT_EXPORT",
        "CUSTOM_PROMPT",
    ] = Field(
        ...,
        description="The type of automated task to execute: 'LOW_STOCK_ALERT', 'SALES_SUMMARY', 'AI_DEMAND_FORECAST', 'EXCEL_REPORT_EXPORT', or 'CUSTOM_PROMPT'.",
    )
    frequency: Literal["DAILY", "WEEKLY", "MONTHLY", "ONCE", "CRON"] = Field(
        default="DAILY",
        description="Schedule frequency: 'DAILY', 'WEEKLY', 'MONTHLY', 'ONCE', or 'CRON'.",
    )
    time: Optional[str] = Field(
        default="09:00",
        description="Execution time in 24-hour format HH:mm (e.g. '09:00', '21:30', '18:00'). Default is '09:00'.",
    )
    days_of_week: Optional[List[int]] = Field(
        default=[1],
        description="For WEEKLY frequency: List of days (0=Sunday, 1=Monday, 2=Tuesday, 3=Wednesday, 4=Thursday, 5=Friday, 6=Saturday). E.g. [1] for Monday.",
    )
    day_of_month: Optional[int] = Field(
        default=1,
        description="For MONTHLY frequency: Day of the month (1-31).",
    )
    cron_expression: Optional[str] = Field(
        default=None,
        description="Custom standard 5-field cron expression if frequency is 'CRON' (e.g. '0 9 * * 1-5').",
    )
    recipient_email: Optional[str] = Field(
        default=None,
        description="Optional recipient email to receive alerts/reports. If omitted, defaults to the store owner's registered email.",
    )
    threshold: Optional[int] = Field(
        default=10,
        description="For LOW_STOCK_ALERT: Minimum stock threshold count to trigger alert.",
    )
    store_id: Optional[str] = Field(default=None, description="Store ID.")
    user_id: Optional[str] = Field(default=None, description="User ID.")


@tool("tool_create_automation", args_schema=CreateAutomationInput)
async def tool_create_automation(
    title: str,
    task_type: str,
    frequency: str = "DAILY",
    time: Optional[str] = "09:00",
    days_of_week: Optional[List[int]] = None,
    day_of_month: Optional[int] = 1,
    cron_expression: Optional[str] = None,
    recipient_email: Optional[str] = None,
    threshold: Optional[int] = 10,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Schedule a new automated background rule in Vyapar Sathi (via Express BullMQ worker).

    Supports:
    - `LOW_STOCK_ALERT`: Automatically scans inventory and emails formatted low stock warnings to the merchant.
    - `SALES_SUMMARY`: Automatically calculates daily/weekly sales count & revenue and delivers email summaries.
    - `AI_DEMAND_FORECAST`: Runs periodic restock & predictive demand forecasting.
    - `EXCEL_REPORT_EXPORT`: Exports periodic Excel reports.

    Use when the merchant asks:
    - "Roz subah 9 baje low stock ka email bhej diya karo"
    - "Setup an automation to send me daily sales report at 9:00 PM"
    - "Schedule weekly sales summary every Monday"
    - "Set up alert when stock drops below 5 units"
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    store_oid, user_oid, store_name, owner_email = await _resolve_store_and_user(store_id, user_id)
    target_email = (recipient_email or owner_email or "").strip()

    express_base = os.getenv(_EXPRESS_URL_ENV, _DEFAULT_EXPRESS_URL).rstrip("/")
    endpoint = f"{express_base}/api/automations"

    payload = {
        "title": title,
        "store": str(store_oid) if store_oid else store_id,
        "user": str(user_oid) if user_oid else user_id,
        "taskType": task_type,
        "frequency": frequency.upper(),
        "time": time or "09:00",
        "daysOfWeek": days_of_week or [1],
        "dayOfMonth": day_of_month or 1,
        "cronExpression": cron_expression,
        "config": {
            "recipientEmail": target_email,
            "threshold": threshold or 10,
        },
    }

    headers = _get_express_headers(user_oid, target_email)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

            if resp.status_code in (200, 201) and "automation" in data:
                auto_doc = data["automation"]
                return {
                    "success": True,
                    "automation_id": str(auto_doc.get("_id")),
                    "title": auto_doc.get("title"),
                    "task_type": auto_doc.get("taskType"),
                    "frequency": auto_doc.get("frequency"),
                    "time": auto_doc.get("time"),
                    "status": auto_doc.get("status", "ACTIVE"),
                    "recipient_email": target_email,
                    "message": f"Automation rule '{title}' scheduled successfully! It will run {frequency.lower()} at {time} IST.",
                }
            else:
                err_msg = data.get("error") or data.get("message") or f"HTTP {resp.status_code}"
                LOGGER.warning("express_create_automation_failed", error=err_msg)
                return {
                    "success": False,
                    "error": err_msg,
                    "message": f"Could not create automation rule: {err_msg}",
                }
    except Exception as exc:
        LOGGER.error("create_automation_exception", error=str(exc))
        return {
            "success": False,
            "error": str(exc),
            "message": f"Failed to connect to automation backend: {str(exc)}",
        }


# ===========================================================================
# Tool 2: List Automations
# ===========================================================================

class ListAutomationsInput(BaseModel):
    status: Optional[Literal["ACTIVE", "PAUSED", "COMPLETED"]] = Field(
        default=None,
        description="Optional filter by status ('ACTIVE', 'PAUSED', or 'COMPLETED').",
    )
    store_id: Optional[str] = Field(default=None, description="Store ID.")
    user_id: Optional[str] = Field(default=None, description="User ID.")


@tool("tool_list_automations", args_schema=ListAutomationsInput)
async def tool_list_automations(
    status: Optional[str] = None,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    List all automated workflows and scheduled tasks configured for this store.
    Returns automation title, schedule frequency, execution time, active status, last execution date, and errors.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")

    store_oid, user_oid, store_name, owner_email = await _resolve_store_and_user(store_id, user_id)

    db = get_database()
    filter_query: Dict[str, Any] = {}
    if store_oid:
        filter_query["store"] = store_oid
    if status:
        filter_query["status"] = status.upper()

    try:
        cursor = db["automations"].find(filter_query).sort("createdAt", -1).limit(20)
        automations = await cursor.to_list(length=20)

        formatted = []
        for a in automations:
            formatted.append({
                "automation_id": str(a.get("_id")),
                "title": a.get("title"),
                "task_type": a.get("taskType"),
                "frequency": a.get("frequency"),
                "time": a.get("time"),
                "status": a.get("status"),
                "recipient_email": (a.get("config") or {}).get("recipientEmail"),
                "last_run_at": a.get("lastRunAt").isoformat() if a.get("lastRunAt") else None,
                "next_run_at": a.get("nextRunAt").isoformat() if a.get("nextRunAt") else None,
                "run_count": a.get("runCount", 0),
                "last_execution_status": a.get("lastExecutionStatus", "PENDING"),
            })

        return {
            "success": True,
            "total_count": len(formatted),
            "automations": formatted,
            "message": f"Found {len(formatted)} automation rule(s) for {store_name}." if formatted else "No automation rules currently configured.",
        }
    except Exception as exc:
        LOGGER.error("list_automations_error", error=str(exc))
        return {
            "success": False,
            "error": str(exc),
            "automations": [],
            "message": f"Error fetching automations: {str(exc)}",
        }


# ===========================================================================
# Tool 3: Toggle Automation Status (Pause / Resume)
# ===========================================================================

class ToggleAutomationInput(BaseModel):
    automation_id: str = Field(..., description="The ID of the automation rule to pause or resume.")
    status: Literal["ACTIVE", "PAUSED"] = Field(..., description="The target status: 'ACTIVE' to resume, 'PAUSED' to pause.")
    store_id: Optional[str] = Field(default=None, description="Store ID.")
    user_id: Optional[str] = Field(default=None, description="User ID.")


@tool("tool_toggle_automation", args_schema=ToggleAutomationInput)
async def tool_toggle_automation(
    automation_id: str,
    status: str,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Pause or resume an existing automation rule.
    Use when the merchant asks:
    - "Pause the daily low stock email alert"
    - "Resume my weekly sales report"
    - "Turn off the 9 AM alert temporarily"
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")
    _, user_oid, _, owner_email = await _resolve_store_and_user(store_id, user_id)

    express_base = os.getenv(_EXPRESS_URL_ENV, _DEFAULT_EXPRESS_URL).rstrip("/")
    endpoint = f"{express_base}/api/automations/{automation_id}/status"
    headers = _get_express_headers(user_oid, owner_email)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.patch(endpoint, json={"status": status.upper()}, headers=headers)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

            if resp.status_code == 200:
                return {
                    "success": True,
                    "automation_id": automation_id,
                    "status": status.upper(),
                    "message": f"Automation has been successfully set to {status.upper()}.",
                }
            else:
                err_msg = data.get("error") or data.get("message") or f"HTTP {resp.status_code}"
                return {
                    "success": False,
                    "error": err_msg,
                    "message": f"Failed to update automation status: {err_msg}",
                }
    except Exception as exc:
        LOGGER.error("toggle_automation_exception", error=str(exc))
        return {
            "success": False,
            "error": str(exc),
            "message": f"Connection error updating automation: {str(exc)}",
        }


# ===========================================================================
# Tool 4: Trigger Automation Immediately
# ===========================================================================

class TriggerAutomationInput(BaseModel):
    automation_id: str = Field(..., description="The ID of the automation rule to trigger immediately.")
    store_id: Optional[str] = Field(default=None, description="Store ID.")
    user_id: Optional[str] = Field(default=None, description="User ID.")


@tool("tool_trigger_automation", args_schema=TriggerAutomationInput)
async def tool_trigger_automation(
    automation_id: str,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Trigger an existing automation workflow to run immediately out-of-schedule.
    Use when the merchant says "Run the low stock alert right now", "Trigger the sales summary now", "Test the automation".
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")
    _, user_oid, _, owner_email = await _resolve_store_and_user(store_id, user_id)

    express_base = os.getenv(_EXPRESS_URL_ENV, _DEFAULT_EXPRESS_URL).rstrip("/")
    endpoint = f"{express_base}/api/automations/{automation_id}/trigger-now"
    headers = _get_express_headers(user_oid, owner_email)

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(endpoint, json={}, headers=headers)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

            if resp.status_code == 200:
                return {
                    "success": True,
                    "automation_id": automation_id,
                    "message": "Automation workflow triggered and executed successfully!",
                }
            else:
                err_msg = data.get("error") or data.get("message") or f"HTTP {resp.status_code}"
                return {
                    "success": False,
                    "error": err_msg,
                    "message": f"Failed to trigger automation: {err_msg}",
                }
    except Exception as exc:
        LOGGER.error("trigger_automation_exception", error=str(exc))
        return {
            "success": False,
            "error": str(exc),
            "message": f"Connection error triggering automation: {str(exc)}",
        }


# ===========================================================================
# Tool 5: Delete Automation Rule
# ===========================================================================

class DeleteAutomationInput(BaseModel):
    automation_id: str = Field(..., description="The ID of the automation rule to delete.")
    store_id: Optional[str] = Field(default=None, description="Store ID.")
    user_id: Optional[str] = Field(default=None, description="User ID.")


@tool("tool_delete_automation", args_schema=DeleteAutomationInput)
async def tool_delete_automation(
    automation_id: str,
    store_id: Optional[str] = None,
    user_id: Optional[str] = None,
    config: RunnableConfig = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Delete an automation rule and remove its scheduled job from BullMQ.
    Use when the merchant asks to delete or remove an automated alert or schedule permanently.
    """
    configurable = (config or {}).get("configurable", {}) if isinstance(config, dict) else {}
    if hasattr(config, "get"):
        configurable = config.get("configurable", {})

    user_id = user_id or kwargs.get("user_id") or configurable.get("user_id", "")
    store_id = store_id or kwargs.get("store_id") or configurable.get("store_id", "")
    _, user_oid, _, owner_email = await _resolve_store_and_user(store_id, user_id)

    express_base = os.getenv(_EXPRESS_URL_ENV, _DEFAULT_EXPRESS_URL).rstrip("/")
    endpoint = f"{express_base}/api/automations/{automation_id}"
    headers = _get_express_headers(user_oid, owner_email)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.delete(endpoint, headers=headers)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

            if resp.status_code == 200:
                return {
                    "success": True,
                    "automation_id": automation_id,
                    "message": "Automation rule has been deleted and unscheduled successfully.",
                }
            else:
                err_msg = data.get("error") or data.get("message") or f"HTTP {resp.status_code}"
                return {
                    "success": False,
                    "error": err_msg,
                    "message": f"Failed to delete automation: {err_msg}",
                }
    except Exception as exc:
        LOGGER.error("delete_automation_exception", error=str(exc))
        return {
            "success": False,
            "error": str(exc),
            "message": f"Connection error deleting automation: {str(exc)}",
        }
