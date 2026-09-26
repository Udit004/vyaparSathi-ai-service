"""
app/agent/prompts/system_prompts/tool_synthesis.py
===================================================
Prompt fragment injected when the agent has gathered data from tool calls.
Instructs the LLM on structuring its response cleanly with tables, metrics,
and risk badges.
"""

from __future__ import annotations

import json
from app.agent.prompts.system_prompts.few_shot_examples import FEW_SHOT_SYNTHESIS


def build_tool_synthesis(
    *,
    results_so_far: list[dict],
    inventory_context: dict,
    sales_context: dict,
    forecast_context: dict,
    insights_context: list,
    candidate_products: list | None = None,
    discovery_metrics: dict | None = None,
) -> str:
    """
    Build the tool-synthesis prompt fragment.
    """
    candidate_products = candidate_products or []
    discovery_metrics = discovery_metrics or {}
    
    top_candidates = candidate_products[:20] if len(candidate_products) > 20 else candidate_products
    
    candidates_context = ""
    if candidate_products:
        candidates_context = (
            f"<discovery_candidates>\n"
            f"Total Candidates Found: {len(candidate_products)}\n"
            f"Showing Top {len(top_candidates)} Candidates:\n"
            f"{json.dumps(top_candidates, default=str)}\n"
            f"</discovery_candidates>\n\n"
        )
        
    metrics_context = ""
    if discovery_metrics:
        metrics_context = (
            f"<discovery_metrics>\n"
            f"{json.dumps(discovery_metrics, default=str)}\n"
            f"</discovery_metrics>\n\n"
        )

    synthesis_instruction = (
        f"DATA SYNTHESIS INSTRUCTIONS ({len(results_so_far)} tool call(s) executed):\n"
        "1. You have sufficient tool data to fulfill the user's request. Respond directly WITHOUT calling more tools unless essential data is missing.\n"
        "2. Ground every claim, number, price, and metric strictly in the gathered context below. Never hallucinate stock quantities or prices.\n"
        "3. Format your response into 3 structured sections:\n"
        "   - **Executive Summary**: 1-2 direct sentences answering the request.\n"
        "   - **Key Data & Analysis**: Clear Markdown table or categorized bullet points using risk badges `[CRITICAL]`, `[WARNING]`, `[HEALTHY]` and monetary format `₹`.\n"
        "   - **Action Recommendations**: Priority-ordered practical next steps for the store owner.\n\n"
        f"{FEW_SHOT_SYNTHESIS}\n\n"
    )

    context_blocks = (
        f"<inventory_data>\n{json.dumps(inventory_context, default=str)}\n</inventory_data>\n\n"
        f"<sales_data>\n{json.dumps(sales_context, default=str)}\n</sales_data>\n\n"
        f"<forecast_data>\n{json.dumps(forecast_context, default=str)}\n</forecast_data>\n\n"
        f"<insights>\n{json.dumps(insights_context, default=str)}\n</insights>\n\n"
        f"{metrics_context}"
        f"{candidates_context}"
    )

    return synthesis_instruction + context_blocks