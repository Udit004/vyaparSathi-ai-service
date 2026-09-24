"""
app/agent/prompts/system_prompts/tool_synthesis.py
===================================================
Prompt fragment injected when the agent has already gathered data from
one or more tool calls. Instructs the LLM to synthesize the gathered
data into a comprehensive, actionable response rather than calling more
tools unnecessarily.
"""

import json


def build_tool_synthesis(*, results_so_far, inventory_context, sales_context, forecast_context, insights_context, candidate_products=None, discovery_metrics=None) -> str:
    """
    Build the tool-synthesis prompt fragment.

    Args:
        results_so_far:       Accumulated list of ToolResult dicts.
        inventory_context:    Dict of inventory tool outputs.
        sales_context:        Dict of sales tool outputs.
        forecast_context:     Dict of forecast/restock tool outputs.
        insights_context:     List of insight dicts.
        candidate_products:   List of candidate products from discovery.
        discovery_metrics:    Metrics from discovery operations.

    Returns:
        A multi-line string with the synthesis instruction and the context
        buckets serialized as XML-ish blocks for the LLM.
    """
    candidate_products = candidate_products or []
    discovery_metrics = discovery_metrics or {}
    
    # Context Builder Logic: Filter and limit candidates sent to LLM
    # If there are many candidates, we only send the top ones.
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
    return (
        f"You have already gathered data from {len(results_so_far)} tool call(s). "
        "If you have enough information to fully answer the user's question, "
        "respond directly WITHOUT calling any more tools. "
        "Synthesize the gathered data into a comprehensive, actionable response. "
        "Be specific — reference actual numbers from the data. "
        "Format clearly with bullet points or sections if appropriate.\n\n"
        f"<inventory_data>\n{json.dumps(inventory_context, default=str)}\n</inventory_data>\n\n"
        f"<sales_data>\n{json.dumps(sales_context, default=str)}\n</sales_data>\n\n"
        f"<forecast_data>\n{json.dumps(forecast_context, default=str)}\n</forecast_data>\n\n"
        f"<insights>\n{json.dumps(insights_context, default=str)}\n</insights>\n\n"
        f"{metrics_context}"
        f"{candidates_context}"
    )