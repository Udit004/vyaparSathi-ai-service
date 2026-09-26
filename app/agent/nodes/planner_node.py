"""
app/agent/nodes/planner_node.py
===============================
Planner node — decides what steps are required to fulfill a complex goal.
It generates a structured plan without executing any tools.
"""

from __future__ import annotations

import json
import structlog
from typing import Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from app.agent.state import VyaparAgentState
from app.lib.llm import get_llm
from app.agent.tools.registry import VYAPAR_TOOLS

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.planner")

_PLANNER_PROMPT = """You are the Vyapar Copilot Planner.
Your job is to analyze the user's complex request and build a simple execution plan of 2 to 4 actionable steps.
Do NOT execute tools directly. Do NOT invent hypothetical external APIs.
Only recommend steps that utilize the system's available tools.

Available capabilities:
{tool_descriptions}

EXAMPLE:
User Request: "Audit my store inventory and prepare a restock recommendation."
Output:
{{
  "goal": "Perform inventory audit and generate prioritized restock recommendation",
  "steps": [
    "1. Gather low stock products and dead stock alerts using inventory tools.",
    "2. Fetch restocking priorities and demand forecast metrics.",
    "3. Synthesize findings into a structured report with risk badges and action items."
  ]
}}

Output your response strictly as JSON with no extra markdown formatting:
{{
  "goal": "Clear summary of the user's goal",
  "steps": [
    "Step 1...",
    "Step 2..."
  ]
}}
"""

async def planner_node(state: VyaparAgentState) -> Dict[str, Any]:
    prompt = state.get("user_prompt", "").strip()
    store_id = state.get("store_id", "unknown")
    
    LOGGER.info("planner_node_start", store_id=store_id, prompt_len=len(prompt))
    
    llm = get_llm()
    if not llm:
        LOGGER.error("planner_node_no_llm")
        return {}
    
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in VYAPAR_TOOLS])
    sys_content = _PLANNER_PROMPT.format(tool_descriptions=tool_descriptions)
    
    messages = [
        SystemMessage(content=sys_content),
        HumanMessage(content=f"Create a plan for this request: {prompt}")
    ]
    
    try:
        response = await llm.ainvoke(messages)
        content = response.content.strip()
        
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        plan_data = json.loads(content.strip())
        
        LOGGER.info("planner_node_success", steps=len(plan_data.get("steps", [])))
        
        return {
            "plan": plan_data
        }
    except Exception as exc:
        LOGGER.error("planner_node_error", error=str(exc))
        return {
            "plan": {
                "goal": prompt,
                "steps": ["Analyze the request and use available tools to answer."]
            }
        }
