"""
app/agent/nodes/reflection_node.py
==================================
Reflection node — generates an improved approach based on the critic's feedback.
"""

from typing import Dict, Any
import structlog

from app.agent.state import VyaparAgentState
from app.lib.llm import get_llm
from langchain_core.messages import SystemMessage, HumanMessage

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.reflection")

async def reflection_node(state: VyaparAgentState) -> Dict[str, Any]:
    llm = get_llm()
    if not llm:
        return {"goal_status": "complete"}
        
    reason = state.get("critic_reason", "")
    missing = state.get("critic_missing_information", [])
    
    LOGGER.info("reflection_node_start", reason=reason)
    
    system_prompt = (
        "You are the reflection component of Vyapar Copilot. "
        "The previous approach failed. The Critic identified what is missing. "
        "Provide an improved execution direction. Which tools should be used next to get the missing information? "
        "Respond with a short, direct paragraph on what to do differently to fix it. "
        "Do not repeat successful work (e.g. if sales is already gathered, only gather forecast)."
    )
    
    user_msg = (
        f"Critic Reason: {reason}\n"
        f"Missing Information: {', '.join(missing)}\n\n"
        "What should we do differently next?"
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg)
    ]
    
    # Hide from SSE stream by stripping callbacks
    try:
        response = await llm.ainvoke(messages, config={"callbacks": []})
        reflection_text = response.content
    except Exception as e:
        LOGGER.error("reflection_node_error", error=str(e))
        return {"goal_status": "complete"} # Fail open
        
    LOGGER.info("reflection_node_result", reflection=reflection_text)
    
    # Add the reflection back into the conversation history so think_node sees it
    reflection_msg = HumanMessage(content=f"Evaluation failed.\nCritic says: {reason}\nReflection for next step: {reflection_text}")
    
    # Increment loop count so reflection respects the global max_loops limit
    current_loop = state.get("loop_count", 0)
    
    return {
        "goal_status": "in_progress",
        "reflection": reflection_text,
        "messages": [reflection_msg],
        "loop_count": current_loop + 1
    }
