"""
app/agent/prompts/system_prompts/tool_synthesis.py
===================================================
Prompt fragment injected when the agent has already gathered data from
one or more tool calls. Instructs the LLM to synthesize the gathered
data into a comprehensive, actionable response rather than calling more
tools unnecessarily.
"""

import json


def build_tool_synthesis(*, results_so_far, inventory_context, sales_context, forecast_context, insights_context) -> str:
    """
    Build the tool-synthesis prompt fragment.

    Args:
        results_so_far:       Accumulated list of ToolResult dicts.
        inventory_context:    Dict of inventory tool outputs.
        sales_context:        Dict of sales tool outputs.
        forecast_context:     Dict of forecast/restock tool outputs.
        insights_context:     List of insight dicts.

    Returns:
        A multi-line string with the synthesis instruction and the context
        buckets serialized as XML-ish blocks for the LLM.
    """
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
    )