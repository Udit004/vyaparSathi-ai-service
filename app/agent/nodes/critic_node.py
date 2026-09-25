"""
app/agent/nodes/critic_node.py
==============================
Critic node — evaluates the draft answer against the goal and available data.
"""

from typing import Dict, Any, List
import structlog
from pydantic import BaseModel, Field

from app.agent.state import VyaparAgentState
from app.lib.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.critic")

class CriticResult(BaseModel):
    status: str = Field(description="Must be 'PASS' if the answer is good, or 'FAIL' if it is missing information or contradicts the context.")
    reason: str = Field(description="A short explanation of why it passed or failed.")
    missing_information: List[str] = Field(description="A list of information that is missing from the answer but required by the goal. Empty if PASS.")
    needs_retry: bool = Field(description="True if the agent should retry to get the missing information.")

async def critic_node(state: VyaparAgentState) -> Dict[str, Any]:
    llm = get_llm()
    if not llm:
        return {"goal_status": "complete"}
    
    goal = state.get("goal", "")
    final_answer = state.get("final_answer", "")
    
    # We strip callbacks so this evaluation is hidden from the SSE stream.
    structured_llm = llm.with_structured_output(CriticResult).with_config({"callbacks": []})
    
    system_prompt = (
        "You are an evaluator for Vyapar Copilot.\n"
        "Your job is to check if the drafted 'Final Answer' satisfies the user's 'Goal', "
        "and if it's supported by the retrieved context.\n"
        "Check for:\n"
        "1. Goal completion: Did the agent actually answer the question?\n"
        "2. Data support: Are the claims supported by the available data?\n"
        "3. Tool coverage: Were the necessary tools used?\n"
        "4. Completeness: Is any important part of the question ignored?\n"
        "Do NOT invent data. Do not execute tools. Just evaluate.\n"
    )
    
    tool_results = state.get("tool_results", [])
    
    # Create an abbreviated context summary to avoid blowing up the context window
    context_str = "Available context summary:\n"
    for r in tool_results:
        context_str += f"- Tool: {r.get('tool_name')} | Success: {r.get('success')}\n"
        # We optionally add small context data here, but omit huge arrays
    
    user_msg = (
        f"Goal: {goal}\n\n"
        f"{context_str}\n\n"
        f"Drafted Answer:\n{final_answer}\n"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg)
    ]
    
    try:
        LOGGER.info("critic_node_evaluating")
        result: CriticResult = await structured_llm.ainvoke(messages)
    except Exception as e:
        LOGGER.error("critic_node_error", error=str(e))
        return {"goal_status": "complete"} # Fail open
        
    status = result.status.upper()
    LOGGER.info("critic_node_result", status=status, reason=result.reason)
    
    if status == "FAIL" and result.needs_retry:
        return {
            "goal_status": "reflect",
            "critic_status": status,
            "critic_reason": result.reason,
            "critic_missing_information": result.missing_information
        }
        
    return {
        "goal_status": "complete",
        "critic_status": status,
        "critic_reason": result.reason,
        "critic_missing_information": result.missing_information
    }
