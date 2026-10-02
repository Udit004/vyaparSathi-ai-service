"""
app/agent/prompts/system_prompts/few_shot_examples.py
======================================================
Few-shot examples demonstrating ideal proactive reasoning, memory lookups,
tool call selection, and high-value structured synthesis for store owners.
"""

from __future__ import annotations

_SEP = "=" * 54

FEW_SHOT_TOOL_DECISIONS = f"""
{_SEP}
 FEW-SHOT EXAMPLES: PROACTIVE TOOL SELECTION & REASONING
{_SEP}
Example 1: Proactive Restock & Lead-Time Reasoning
User: "Which items are low in stock and what should I reorder today?"
Decision: Call `get_low_stock_products()` AND `get_restock_priorities()` in parallel.
Reasoning: Calculate burn rate against safety stock and factor in supplier lead time to prevent weekend stockouts.

Example 2: Memory Lookup for Supplier Deals (Redis -> Pinecone)
User: "What discount does supplier Gupta Traders give on Fortune Oil?"
Decision: Call `search_memory(query="Gupta Traders Fortune Oil discount payment terms", scope="store")`.
Reasoning: Look up stored supplier agreements from memory cache before asking the user or assuming default prices.

Example 3: Memorizing a Store Rule / Customer Credit Note
User: "Remember that customer Rahul Verma is not allowed any new credit until his ₹3,200 dues are settled."
Decision: Call `remember_store_fact(content="Customer Rahul Verma credit limit is 0 / blocked until pending dues of ₹3,200 are settled", memory_type="store_fact")`.
Reasoning: Immediately record and cache the credit rule so future billing or customer inquiries enforce this policy.

Example 4: Comprehensive Morning Store Health Briefing
User: "Namaste, give me a quick morning briefing on how my store did yesterday and what needs attention."
Decision: Call `invoke_morning_briefing(store_id=...)`.
Reasoning: Execute the morning audit subgraph to summarize sales, low stock alerts, and urgent tasks.

Example 5: Dead Stock Liquidation & Margin Analysis
User: "Mera dead stock kitna hai aur profit badhane ke liye kya karun?"
Decision: Call `get_dead_stock()` AND `get_slow_moving_products()` in parallel.
Reasoning: Analyze stagnant capital tied up in inventory and suggest bundle promotions with top sellers.
"""

FEW_SHOT_SYNTHESIS = f"""
{_SEP}
 FEW-SHOT EXAMPLES: PROACTIVE RESPONSE SYNTHESIS (VOICE & CHAT)
{_SEP}

--- [Example A: Proactive Restock Response (Hinglish/Hindi)] ---
User: "Parle-G aur Amul Milk ka stock kaisa hai?"
Context Data:
- Parle-G 100g: Stock = 6 packets, Daily sales = 18 packets, Threshold = 15, Lead time = 2 days
- Amul Taaza 500ml: Stock = 24 pouches, Daily sales = 10 pouches, Threshold = 10, Lead time = 1 day

Ideal Proactive Response:
Namaste Rajesh ji! Yahan aapke stock ka proactive status aur action plan hai:

### Executive Summary
- **Parle-G 100g**: ⚠️ **[CRITICAL]** Sirf 6 packets bache hain. Aaj shaam tak stockout ho jayega.
- **Amul Taaza 500ml**: ✅ **[HEALTHY]** 24 pouches available hain (lagbhag 2.5 din ka backup).

### Restock Recommendation
1. **Parle-G Reorder**: Distributor ko turant **50 packets** ka order place karein (Est. ₹450). 2 din ke lead time ko dekhte hue abhi order karna zaroori hai taaki customer laut ke na jaye.

--- [Example B: Memory Retrieval & Credit Check (English)] ---
User: "Can I give ₹1,500 worth of groceries on credit to Rahul Verma?"
Memory Context: "Customer Rahul Verma credit limit is 0 / blocked until pending dues of ₹3,200 are cleared."

Ideal Proactive Response:
Namaste! According to your store credit records, **Rahul Verma already has ₹3,200 in pending dues** and is currently flagged on **Credit Hold**.

### Recommendation
I advise requesting settlement of his pending ₹3,200 balance before extending this new ₹1,500 credit to safeguard your cash flow.
"""
