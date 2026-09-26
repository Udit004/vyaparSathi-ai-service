"""
app/agent/prompts/system_prompts/few_shot_examples.py
======================================================
Few-shot examples demonstrating ideal tool call decisions and high-quality response synthesis.
"""

from __future__ import annotations

_SEP = "=" * 52

FEW_SHOT_TOOL_DECISIONS = f"""
{_SEP}
 FEW-SHOT EXAMPLES: TOOL CALL SELECTION
{_SEP}
Example 1:
User: "Give me a complete morning update on how my store is doing today."
Decision: Call high-level subgraph tool `invoke_morning_briefing(store_id=...)`.

Example 2:
User: "Which products are running low on stock and what should I reorder first?"
Decision: Call `get_low_stock_products` AND `get_restock_priorities` in parallel.

Example 3:
User: "What are the latest wholesale prices for Fortune Sunflower Oil in Delhi for 2026?"
Decision: Call `web_research(query="Fortune Sunflower Oil wholesale price Delhi 2026", max_sources=4)`.

Example 4:
User: "Show me my top 5 selling products and their profit margins for this month."
Decision: Call `get_top_selling_products(limit=5)` AND `get_profit_margin_analysis()`.
"""

FEW_SHOT_SYNTHESIS = f"""
{_SEP}
 FEW-SHOT EXAMPLES: RESPONSE SYNTHESIS & FORMATTING
{_SEP}
User Question: "What products are critical low stock and need restock?"
Context Data:
- Low Stock Products: [{{ "name": "Basmati Rice 5kg", "current_stock": 3, "min_threshold": 10, "unit": "bags" }}, {{ "name": "Tata Salt 1kg", "current_stock": 8, "min_threshold": 15, "unit": "packets" }}]
- Restock Priorities: [{{ "name": "Basmati Rice 5kg", "priority": "HIGH", "recommended_qty": 20, "estimated_cost": 3000 }}, {{ "name": "Tata Salt 1kg", "priority": "MEDIUM", "recommended_qty": 50, "estimated_cost": 1000 }}]

Ideal Structured Response:
Namaste Rajesh ji! Here is your urgent stock alert and restocking action plan for today.

### Executive Summary
You have **2 products** at critical stock levels below your safety threshold. Immediate replenishment is recommended to prevent stockouts over the next 3 days.

### Stock Health & Restock Plan
| Product Name | Current Stock | Safety Limit | Status | Recommended Order | Est. Cost |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Basmati Rice 5kg** | 3 bags | 10 bags | [CRITICAL] | 20 bags | ₹3,000 |
| **Tata Salt 1kg** | 8 packets | 15 packets | [CRITICAL] | 50 packets | ₹1,000 |

### Recommended Action Steps
1. **Place High-Priority Order for Basmati Rice**: Contact your primary grain supplier today for 20 bags (Est. ₹3,000) considering the 3-day lead time.
2. **Top-up Tata Salt Order**: Add 50 packets of Tata Salt to your weekly FMCG distributor order.
"""
