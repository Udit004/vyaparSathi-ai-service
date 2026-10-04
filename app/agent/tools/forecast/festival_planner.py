"""
app/agent/tools/forecast/festival_planner.py
================================================
Indian Festival Demand & Restock Planner Tool.

Analyzes upcoming Indian calendar festivals (Diwali, Holi, Rakhi, Navratri, Eid,
Makar Sankranti, etc.) and generates proactive restock recommendations 3-4 weeks in advance.
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import structlog

from app.agent.service.products.search import search_products

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.tools.forecast.festival_planner")

# Standard Indian Festival & Seasonal Peak Reference Data
INDIAN_FESTIVALS_DB = [
    {
        "name": "Diwali & Dhanteras (दीपावली)",
        "month_approx": 10,  # Oct / Nov
        "day_approx": 25,
        "categories": ["Sweets", "Dry Fruits", "Pooja Items", "Beverages", "Snacks", "Gifting", "Lighting", "Electronics", "Groceries"],
        "keywords": ["sweet", "mithai", "dry fruit", "kaju", "badam", "ghee", "oil", "chocolate", "gift", "snack"],
        "surge_multiplier": 2.8,
        "lead_time_weeks": 4,
        "description": "Peak Indian shopping festival. High demand for sweets, dry fruit hampers, ghee, cooking oil, and gifting items."
    },
    {
        "name": "Holi (होली)",
        "month_approx": 3,  # March
        "day_approx": 15,
        "categories": ["Sweets", "Beverages", "Snacks", "Colors", "Dairy"],
        "keywords": ["gujiya", "thandai", "sweet", "namkeen", "color", "khoya", "snack"],
        "surge_multiplier": 2.2,
        "lead_time_weeks": 3,
        "description": "Festival of colors. High demand for Gujiya ingredients (Mawa/Khoya, Suji, Maida), Thandai, Cold Drinks, and Snacks."
    },
    {
        "name": "Navratri & Durga Puja (नवरात्रि)",
        "month_approx": 10,  # Oct
        "day_approx": 3,
        "categories": ["Fasting Items", "Pooja Items", "Dairy", "Dry Fruits"],
        "keywords": ["sabudana", "kuttu", "sendha namak", "ghee", "pooja", "dry fruit"],
        "surge_multiplier": 2.5,
        "lead_time_weeks": 3,
        "description": "9-day fasting period. High demand for Sabudana, Kuttu Atta, Sendha Namak, Ghee, Milk, and Pooja Items."
    },
    {
        "name": "Raksha Bandhan (रक्षाबंधन)",
        "month_approx": 8,  # August
        "day_approx": 18,
        "categories": ["Sweets", "Chocolates", "Gifting", "Pooja Items"],
        "keywords": ["rakhi", "sweet", "chocolate", "cadbury", "gift"],
        "surge_multiplier": 2.0,
        "lead_time_weeks": 3,
        "description": "Brother-sister celebration. Heavy demand for Cadbury celebration packs, boxed sweets, dry fruit boxes, and Rakhis."
    },
    {
        "name": "Eid-ul-Fitr & Ramzan (ईद)",
        "month_approx": 4,  # April / March
        "day_approx": 10,
        "categories": ["Sweets", "Dry Fruits", "Spices", "Beverages", "Gifting"],
        "keywords": ["sevaiyan", "dates", "khajoor", "rooh afza", "dry fruit"],
        "surge_multiplier": 2.4,
        "lead_time_weeks": 3,
        "description": "Ramzan fasting & Eid festivities. High demand for Sevaiyan (vermicelli), Dates, Rooh Afza, Dry Fruits, and Spices."
    },
    {
        "name": "Makar Sankranti & Pongal (मकर संक्रांति)",
        "month_approx": 1,  # January
        "day_approx": 14,
        "categories": ["Sweets", "Grains", "Jaggery"],
        "keywords": ["til", "sesame", "jaggery", "gud", "chikki"],
        "surge_multiplier": 1.8,
        "lead_time_weeks": 2,
        "description": "Harvest festival. High demand for Til (sesame), Gud (jaggery), Chikki, and special rice/dal items."
    },
    {
        "name": "Ganesh Chaturthi (गणेश चतुर्थी)",
        "month_approx": 9,  # September
        "day_approx": 7,
        "categories": ["Sweets", "Pooja Items", "Dairy"],
        "keywords": ["modak", "besan", "rawa", "ghee", "sugar"],
        "surge_multiplier": 2.0,
        "lead_time_weeks": 3,
        "description": "Ganesh festival. High demand for Modak ingredients (Besan, Rawa, Ghee, Sugar, Coconut) and Pooja materials."
    },
    {
        "name": "New Year & Christmas (नया साल)",
        "month_approx": 12,  # December
        "day_approx": 31,
        "categories": ["Beverages", "Snacks", "Chocolates", "Bakery"],
        "keywords": ["cake", "chocolate", "cold drink", "chips", "snack"],
        "surge_multiplier": 1.9,
        "lead_time_weeks": 2,
        "description": "Year-end party season. High demand for Cakes, Chocolates, Cold Drinks, Chips, and Party Snacks."
    },
]


class FestivalPlannerInput(BaseModel):
    store_id: str = Field(..., description="The ID of the store.")
    upcoming_months: int = Field(
        default=2,
        description="Number of months ahead to scan for upcoming Indian festivals and seasonal demand peaks."
    )
    festival_name: Optional[str] = Field(
        default=None,
        description="Optional specific festival name to focus on (e.g., 'Diwali', 'Holi', 'Navratri', 'Eid', 'Rakhi')."
    )
    user_id: Optional[str] = Field(None, description="Optional user ID context")

    class Config:
        extra = "ignore"


@tool("get_festival_demand_planner", args_schema=FestivalPlannerInput)
async def get_festival_demand_planner(
    store_id: str,
    upcoming_months: int = 2,
    festival_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Analyze upcoming Indian festivals and recommend stock restock quantities 3-4 weeks in advance
    for high-demand seasonal products.
    """
    LOGGER.info("get_festival_demand_planner_start", store_id=store_id, upcoming_months=upcoming_months, festival_name=festival_name)
    now = datetime.now()

    # 1. Identify relevant upcoming festivals
    matching_festivals = []
    for fest in INDIAN_FESTIVALS_DB:
        if festival_name:
            if festival_name.lower() in fest["name"].lower() or any(k in festival_name.lower() for k in fest["keywords"]):
                matching_festivals.append(fest)
                continue

        fest_month = fest["month_approx"]
        target_year = now.year if fest_month >= now.month else now.year + 1
        try:
            fest_date = datetime(target_year, fest_month, fest["day_approx"])
        except ValueError:
            fest_date = datetime(target_year, fest_month, 20)

        days_until = (fest_date - now).days
        months_until = days_until / 30.0

        if 0 <= months_until <= upcoming_months or (festival_name and fest in matching_festivals):
            fest_info = dict(fest)
            fest_info["target_date_str"] = fest_date.strftime("%d %B %Y")
            fest_info["days_until"] = days_until
            deadline_date = fest_date - timedelta(weeks=fest["lead_time_weeks"])
            fest_info["restock_deadline_str"] = deadline_date.strftime("%d %B %Y")
            fest_info["order_urgent"] = (deadline_date - now).days <= 7
            matching_festivals.append(fest_info)

    if not matching_festivals:
        fest = INDIAN_FESTIVALS_DB[0]  # Diwali fallback
        fest_date = datetime(now.year, fest["month_approx"], fest["day_approx"])
        if fest_date < now:
            fest_date = datetime(now.year + 1, fest["month_approx"], fest["day_approx"])
        days_until = (fest_date - now).days
        deadline_date = fest_date - timedelta(weeks=fest["lead_time_weeks"])
        fest_info = dict(fest)
        fest_info["target_date_str"] = fest_date.strftime("%d %B %Y")
        fest_info["days_until"] = days_until
        fest_info["restock_deadline_str"] = deadline_date.strftime("%d %B %Y")
        fest_info["order_urgent"] = (deadline_date - now).days <= 7
        matching_festivals.append(fest_info)

    matching_festivals.sort(key=lambda x: x.get("days_until", 999))

    # 2. Match store inventory concurrently using top keywords
    search_queries = []
    for fest in matching_festivals[:2]:
        for kw in fest["keywords"][:3]:
            search_queries.append((fest, kw))

    tasks = [search_products(store_id=store_id, query=kw) for _, kw in search_queries]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)

    recommended_restocks = []
    seen_products = set()

    for idx, (fest, kw) in enumerate(search_queries):
        res = results_list[idx]
        if isinstance(res, list):
            for prod in res:
                pid = prod.get("product_id") or prod.get("name")
                if pid in seen_products:
                    continue
                seen_products.add(pid)

                curr_qty = prod.get("quantity", 10)
                price = prod.get("price", 100.0)
                surge_mult = fest["surge_multiplier"]

                suggested_stock = int(max(25, curr_qty * surge_mult))
                order_qty = max(0, suggested_stock - curr_qty)
                estimated_cost = round(order_qty * price * 0.75, 2)

                recommended_restocks.append({
                    "product_name": prod.get("name"),
                    "category": prod.get("category", "General"),
                    "current_stock": curr_qty,
                    "unit_retail_price": f"₹{price}",
                    "festival_target": fest["name"],
                    "demand_surge_multiplier": f"{surge_mult}x",
                    "recommended_festival_stock": suggested_stock,
                    "recommended_order_quantity": order_qty,
                    "estimated_investment": f"₹{estimated_cost:,.2f}",
                    "restock_deadline": fest["restock_deadline_str"],
                    "urgent_action_required": fest["order_urgent"],
                    "reasoning": f"High demand anticipated during {fest['name']}. Placing order before {fest['restock_deadline_str']} prevents price spikes."
                })

    recommended_restocks = recommended_restocks[:8]

    top_fest = matching_festivals[0]
    summary_text = (
        f"Upcoming Festival Alert: {top_fest['name']} is in approx {top_fest['days_until']} days ({top_fest['target_date_str']}). "
        f"To avoid wholesale price spikes, place your restock orders before {top_fest['restock_deadline_str']}. "
        f"Key high-demand categories: {', '.join(top_fest['categories'][:4])}."
    )

    return {
        "status": "success",
        "current_date": now.strftime("%d %B %Y"),
        "upcoming_festivals_scanned": len(matching_festivals),
        "festivals": [
            {
                "name": f["name"],
                "target_date": f["target_date_str"],
                "days_until": f["days_until"],
                "restock_deadline": f["restock_deadline_str"],
                "urgent": f["order_urgent"],
                "description": f["description"],
                "key_categories": f["categories"]
            }
            for f in matching_festivals[:3]
        ],
        "recommended_restocks": recommended_restocks,
        "executive_summary": summary_text
    }
