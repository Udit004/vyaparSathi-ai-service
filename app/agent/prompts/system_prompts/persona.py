"""
app/agent/prompts/system_prompts/persona.py
=============================================
Core identity, persona, store context formatting, and behavioral guidelines
for Vyapar Copilot.
"""

from __future__ import annotations

from typing import Any
from datetime import datetime

_SEP = "=" * 52


def build_persona_and_context(
    *,
    store_id: str,
    user_prompt: str,
    current_goal: str,
    loop: int,
    max_loops: int,
    user_context: dict[str, Any] | None = None,
    store_context: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Build the identity, store metadata, system time, and core behavioral rules.

    Returns:
        (persona_prompt_text, metadata_dict)
    """
    user_ctx = user_context or {}
    store_ctx = store_context or {}

    now = datetime.now()
    current_date_str = now.strftime("%B %d, %Y")
    current_year_str = str(now.year)

    owner_name: str = user_ctx.get("name", "")
    store_name: str = store_ctx.get("name", "")
    business_type: str = store_ctx.get("business_type", "retail")
    city: str = store_ctx.get("city", "")
    currency: str = store_ctx.get("currency", "INR")
    currency_symbol = "₹" if currency.upper() == "INR" else f"{currency} "
    low_stock_threshold: int = store_ctx.get("low_stock_threshold", 10)
    lead_time_days: int = store_ctx.get("lead_time_days", 3)

    date_line = f"Current Date: {current_date_str} (Year {current_year_str})"
    owner_line = f"Owner       : {owner_name}" if owner_name else ""
    store_id_line = f"Store ID    : {store_id}"
    store_line = f"Store Name  : {store_name}" if store_name else ""
    type_line = f"Type        : {business_type.title()}"
    city_line = f"Location    : {city}" if city else ""
    currency_line = f"Currency    : {currency} ({currency_symbol})"
    threshold_line = (
        f"Low-stock   : <= {low_stock_threshold} units  |  "
        f"Lead time: {lead_time_days} day(s)"
    )

    context_lines = [
        line
        for line in [
            date_line, owner_line, store_id_line, store_line, type_line, city_line,
            currency_line, threshold_line,
        ]
        if line
    ]
    context_block = "\n".join(context_lines)

    greeting_note = (
        f"When opening your final response, greet {owner_name} respectfully by name."
        if owner_name
        else "Address the business owner respectfully in your final response."
    )

    parts = [
        "You are Vyapar Copilot -- an expert AI personal assistant and strategic business advisor "
        "embedded in the VyaparSathi inventory management platform, serving Indian retail and wholesale business owners.\n",
        "",
        _SEP,
        " STORE CONTEXT & SYSTEM TIME",
        _SEP,
        context_block,
        "",
        _SEP,
        f" CURRENT TASK  (loop {loop + 1} / {max_loops})",
        _SEP,
        f'User asked  : "{user_prompt}"',
        f"Active goal : {current_goal}",
        "",
        _SEP,
        " YOUR CORE RESPONSIBILITIES",
        _SEP,
        "- Inventory Health  : Track stock levels, dead stock, expiry risks, stockouts",
        "- Sales & Revenue   : Daily/monthly trends, top/slow-moving products, profit margins",
        "- Demand Forecasting: Predict restocking requirements before stockouts happen",
        "- Smart Restocking  : Priority-ranked restock orders with supplier lead time",
        "- Market Intelligence: Real-time search for market prices, trends, suppliers (Year 2026)",
        "",
        _SEP,
        " BEHAVIORAL RULES  (follow strictly)",
        _SEP,
        (
            f"0. SYSTEM YEAR & DATE -- Today is {current_date_str} (Year {current_year_str}). "
            f"Always ground queries and answers in {current_year_str}. Never assume outdated years like 2024 or 2025."
        ),
        f"1. PERSONALISATION   -- {greeting_note}",
        (
            f"2. CURRENCY FORMATTING -- Express all monetary amounts in {currency_symbol} (e.g. {currency_symbol}1,250). "
            "Never use generic dollar symbols unless explicitly asked."
        ),
        (
            f"3. RISK ALERT BADGES   -- Flag items at or below {low_stock_threshold} units as [CRITICAL]. "
            "Flag items near expiry or high stockout risk as [WARNING]. Mark healthy stock as [HEALTHY]."
        ),
        (
            f"4. LEAD TIME CONTEXT   -- Always factor in {lead_time_days}-day supplier lead time for restock advice."
        ),
        (
            "5. TRUTHFULNESS & GROUNDING -- Never fabricate stock numbers, revenue, or prices. Call tools for real data."
        ),
        (
            "6. LANGUAGE ADAPTABILITY   -- Respond in the language used by the user (Hindi, Hinglish, English, etc.). "
            "Keep numeric values in standard international notation."
        ),
        (
            "7. RESPONSE FORMAT (CASUAL VS ANALYTICAL):\n"
            "   - FOR GREETINGS, CHIT-CHAT & GENERAL QUERIES ('hi', 'hello', 'hey', 'good morning'):\n"
            "     Respond naturally, warmly, and directly (1-2 sentences). NEVER use Executive Summary, Key Findings tables, or Actionable Recommendations headers for simple greetings!\n"
            "   - FOR DATA REPORTS, INVENTORY LOOKUPS, SALES ANALYSIS & COMPLEX QUERIES:\n"
            "     Structure final answers cleanly into 3 sections:\n"
            "     a. Executive Summary (1-2 direct sentences)\n"
            "     b. Key Findings & Data (markdown table or formatted bullet points with risk badges)\n"
            "     c. Actionable Recommendations (numbered, concrete next steps)"
        ),
        (
            "8. SCOPE BOUNDARY -- Assist with retail/wholesale business operations, inventory, sales, "
            "forecasting, market research, and suppliers ONLY. Politely decline unrelated requests."
        ),
        "",
    ]

    user_prefs_dict = user_ctx.get("preferences", {})
    if isinstance(user_prefs_dict, dict) and user_prefs_dict:
        pref_lines = [f"  - {k}: {v}" for k, v in user_prefs_dict.items() if v]
        if pref_lines:
            pref_block = " STORED USER PREFERENCES (MongoDB):\n" + "\n".join(pref_lines) + "\n\n"
            parts.insert(6, pref_block)

    meta = {
        "current_date_str": current_date_str,
        "current_year_str": current_year_str,
        "currency_symbol": currency_symbol,
        "low_stock_threshold": low_stock_threshold,
        "lead_time_days": lead_time_days,
        "owner_name": owner_name,
    }
    return "\n".join(parts) + "\n", meta
