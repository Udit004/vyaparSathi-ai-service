"""
app/agent/nodes/interrupt_node.py
===================================
Interrupt node — generates a clarification question when the agent
cannot proceed confidently and pauses graph execution using LangGraph's
native ``interrupt()`` function.

This node is invoked when the think node determines that:
    1. Tool results are contradictory or inconclusive
    2. The user's query is too ambiguous to answer with available tools
    3. A critical decision requires user confirmation
    4. The agent has exhausted its tool-calling ability without progress

The node uses a small/fast LLM call to formulate a targeted
clarification question, then calls ``langgraph.types.interrupt()``
to pause the graph. The graph state is saved by the checkpointer,
and the question payload surfaces to the caller via the ``__interrupt__``
key in the stream chunk.

When the user responds, the caller resumes the graph by re-invoking
``astream_events`` with ``Command(resume=<user_answer>)`` using the
same ``thread_id``. The ``interrupt()`` call returns the user's answer,
which is stored in ``clarification_history``, and execution continues
from the interrupt node — the graph routes back to ``think`` so the LLM
can incorporate the clarification and proceed.
"""

from __future__ import annotations

from typing import Any, Dict

import structlog

from langgraph.types import interrupt

from app.agent.state import VyaparAgentState
from app.agent.prompts.interrupt_prompt import INTERRUPT_INSTRUCTION
from app.agent.utils import latest_human_prompt, message_text
from app.lib.summarizer import summarize

LOGGER = structlog.get_logger("vyaparsathi.ai.agent.interrupt")


async def interrupt_node(state: VyaparAgentState) -> Dict[str, Any]:
    """
    Generate a clarification question and pause graph execution.

    The node builds a targeted question (reusing the reason from the
    think node when available, or generating one via a small LLM call),
    then calls ``interrupt()`` to suspend the graph.

    On the **first invocation**, ``interrupt()`` raises ``GraphInterrupt``
    which suspends execution and saves the checkpoint. The question
    payload is surfaced to the caller via ``__interrupt__`` in the
    stream chunk.

    On **resume** (via ``Command(resume=<answer>)``), the node
    re-executes from the beginning. ``interrupt()`` returns the user's
    answer immediately. The Q&A pair is appended to ``clarification_history``
    and the node returns, routing the graph back to ``think``.

    Important: any code before ``interrupt()`` runs on **both** the first
    invocation and the resume. The LLM question-generation call is
    idempotent, so this is safe. The ``clarification_prompt`` set by
    the think node is available in the checkpoint state on both runs,
    so the LLM fallback is only triggered when the think node did not
    provide a reason.
    """
    store_id = state.get("store_id", "unknown")
    user_id = state.get("user_id", "unknown")
    goal = state.get("goal", "")

    LOGGER.info(
        "interrupt_node_start",
        store_id=store_id,
        user_id=user_id,
        goal=goal,
        loop_count=state.get("loop_count", 0),
    )

    # ------------------------------------------------------------------
    # Build the clarification question.
    #
    # The think node sets `clarification_prompt` to the reason/explanation
    # when it decides clarification is needed. We reuse that as the basis
    # for the question — this avoids an extra LLM call and ensures
    # consistency across the first run and the resume (the checkpoint
    # preserves the think node's `clarification_prompt`).
    #
    # If the think node didn't provide a reason (empty string), we
    # fall back to an LLM call that generates a targeted question.
    # ------------------------------------------------------------------

    existing_prompt = state.get("clarification_prompt", "")

    if not existing_prompt:
        # Fallback: generate a clarification question via LLM
        user_prompt = latest_human_prompt(
            state.get("messages", []),
            state.get("user_prompt", ""),
        )
        context_summary = _build_context_summary(state)

        clarification_history = state.get("clarification_history", [])
        history_note = ""
        if clarification_history:
            history_note = (
                "\nPrevious clarifications in this conversation:\n"
                + "\n".join(
                    f"- Q: {c.get('question', '')}  A: {c.get('answer', '')}"
                    for c in clarification_history[-3:]
                )
            )

        full_prompt = (
            f"User's original question: \"{user_prompt}\"\n\n"
            f"Agent's current goal: {goal}\n\n"
            f"Situation:\n{context_summary}\n"
            f"{history_note}\n\n"
            "Based on this, determine if you need to ask the user a "
            "clarification question. If yes, what should it be?"
        )

        try:
            result = await summarize(
                full_prompt,
                instruction=INTERRUPT_INSTRUCTION,
                max_tokens=256,
            )
            _, _, question = _parse_interrupt_result(result)
            if not question:
                question = existing_prompt or (
                    "I need some additional information to give you "
                    "an accurate answer. Could you help me with a few "
                    "details?"
                )
        except Exception as exc:
            LOGGER.error(
                "interrupt_node_llm_failed",
                store_id=store_id,
                error=str(exc),
            )
            question = (
                "I need some additional information to give you "
                "an accurate answer. Could you help me with a few "
                "details?"
            )
    else:
        question = existing_prompt

    LOGGER.info(
        "interrupt_node_clarification_prepared",
        store_id=store_id,
        question_len=len(question),
    )

    # ------------------------------------------------------------------
    # Pause graph execution.
    #
    # On the first invocation, this raises GraphInterrupt and the
    # checkpoint is saved. The `value` here is the payload that
    # surfaces to the caller (via __interrupt__ in the stream or
    # result["__interrupt__"] with invoke).
    #
    # On resume, `interrupt()` returns the user's answer (from
    # Command(resume=...)), and execution continues below.
    # ------------------------------------------------------------------

    interrupt_payload = {
        "question": question,
        "threadId": state.get("thread_id", ""),
    }

    user_answer = interrupt(interrupt_payload)

    # ------------------------------------------------------------------
    # We only reach here on resume — `user_answer` is the value passed
    # to Command(resume=...).
    # ------------------------------------------------------------------

    answer_text = user_answer if isinstance(user_answer, str) else str(user_answer)

    LOGGER.info(
        "interrupt_node_answer_received",
        store_id=store_id,
        answer_len=len(answer_text),
    )

    # Append the Q&A pair to the clarification history
    existing_history = state.get("clarification_history", [])
    history_entry = {
        "question": question,
        "answer": answer_text,
        "reason": existing_prompt or question,
    }
    updated_history = existing_history + [history_entry]

    return {
        "needs_clarification": False,
        "clarification_prompt": "",
        "clarification_history": updated_history,
        "resume_from_clarification": True,
    }


def _build_context_summary(state: VyaparAgentState) -> str:
    """Build a compact summary of the current situation for the LLM."""
    parts: list[str] = []

    goal = state.get("goal", "")
    if goal:
        parts.append(f"Goal: {goal}")

    goal_status = state.get("goal_status", "")
    if goal_status:
        parts.append(f"Goal status: {goal_status}")

    loop = state.get("loop_count", 0)
    if loop > 0:
        parts.append(f"Loops completed: {loop}")

    tool_results = state.get("tool_results", [])
    if tool_results:
        parts.append(f"Tool results collected: {len(tool_results)}")
        for r in tool_results[-5:]:
            name = r.get("tool_name", "unknown")
            success = r.get("success", False)
            error = r.get("error", "")
            parts.append(f"  - {name}: {'success' if success else f'FAILED: {error}'}")

    for bucket, label in [
        ("inventory_context", "Inventory data"),
        ("sales_context", "Sales data"),
        ("forecast_context", "Forecast data"),
        ("restock_context", "Restock data"),
        ("insights_context", "Insights data"),
    ]:
        data = state.get(bucket, {})
        if data:
            if isinstance(data, list):
                parts.append(f"{label}: {len(data)} items")
            else:
                keys = list(data.keys())[:5]
                parts.append(f"{label}: keys={keys}")

    return "\n".join(parts) if parts else "No data collected yet."


def _parse_interrupt_result(result: str) -> tuple[bool, str, str]:
    """
    Parse the LLM's structured output from the interrupt call.

    Expected format:
        needs_clarification: YES|NO
        reason: ...
        question: ...

    Returns:
        (needs_clarification, reason, question)
    """
    needs = False
    reason = ""
    question = ""

    lines = (result or "").strip().split("\n")
    for line in lines:
        line = line.strip()
        lower = line.lower()
        if lower.startswith("needs_clarification:"):
            value = line.split(":", 1)[1].strip().lower()
            needs = value in ("yes", "true", "1")
        elif lower.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
        elif lower.startswith("question:"):
            question = line.split(":", 1)[1].strip()

    if not needs:
        question = ""

    return needs, reason, question
