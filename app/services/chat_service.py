from __future__ import annotations

import json
from typing import Any, Iterator, Literal, TypedDict, cast

from app.lib.llm import get_llm
from app.schemas.chat import ChatRequest, ChatResponse, CopilotRequest, CopilotResponse

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover
    END = None
    StateGraph = None


def generate_chat_response(request: ChatRequest) -> ChatResponse:
    llm = get_llm()

    if not llm:
        return ChatResponse(
            response="AI chat is unavailable because the Gemini API key is not configured."
        )

    response = llm.invoke(request.message)
    return ChatResponse(response=_as_text(getattr(response, "content", "")))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_as_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)


class CopilotState(TypedDict):
    question: str
    context_snapshot: str
    basis: str
    summary: str
    key_signals: list[str]
    recommended_actions: list[str]
    risk_level: str
    llm_used: bool
    fallback_used: bool


def _build_context_snapshot(request: CopilotRequest) -> str:
    forecast_lines = [
        f"- {item.product_name}: demand7d={item.predicted_demand_7d:.1f}, trend={item.trend_percent:.1f}%"
        for item in request.context.forecast_items[:5]
    ]
    restock_lines = [
        f"- {item.product_name}: recommended={item.recommended_qty:.1f}, priority={item.priority}, days_to_stockout={item.days_to_stockout}"
        for item in request.context.restock_items[:5]
    ]
    anomaly_lines = [
        f"- {item.product_name}: {item.direction} (z={item.z_score:.2f})"
        for item in request.context.anomalies[:5]
    ]
    insight_lines = [
        f"- [{item.severity}] {item.title}: {item.summary}"
        for item in request.context.insights[:5]
    ]

    sections = [
        f"Store: {request.context.store_name}",
        f"Data basis: {request.context.basis}",
        "Top forecast items:",
        "\n".join(forecast_lines) if forecast_lines else "- None",
        "Top restock items:",
        "\n".join(restock_lines) if restock_lines else "- None",
        "Anomalies:",
        "\n".join(anomaly_lines) if anomaly_lines else "- None",
        "Insights:",
        "\n".join(insight_lines) if insight_lines else "- None",
    ]

    return "\n".join(sections)


def _fallback_from_context(request: CopilotRequest) -> CopilotResponse:
    restock = request.context.restock_items[:3]
    forecast = request.context.forecast_items[:3]
    anomaly = request.context.anomalies[:1]

    if restock:
        top_restock = restock[0]
        summary = (
            f"{request.context.store_name} should prioritize {top_restock.product_name} first; "
            f"recommended quantity is {top_restock.recommended_qty:.1f} and priority is {top_restock.priority}."
        )
    elif forecast:
        top_forecast = forecast[0]
        summary = (
            f"{request.context.store_name} demand is currently led by {top_forecast.product_name} "
            f"with projected 7-day demand {top_forecast.predicted_demand_7d:.1f}."
        )
    else:
        summary = (
            f"{request.context.store_name} has limited analytics context right now; "
            "collect more transaction data for stronger recommendations."
        )

    key_signals = []
    if forecast:
        key_signals.append(
            f"Top demand item: {forecast[0].product_name} ({forecast[0].predicted_demand_7d:.1f} units/7d)."
        )
    if anomaly:
        key_signals.append(
            f"Primary anomaly: {anomaly[0].product_name} shows a {anomaly[0].direction}."
        )
    if not key_signals:
        key_signals.append("No strong signal detected from current context.")

    recommended_actions = [
        f"Review red/yellow restock items for {request.context.store_name} before next replenishment cycle.",
        "Track next 7 days of actual sales and compare with forecast to recalibrate demand confidence.",
    ]

    return CopilotResponse(
        summary=summary,
        key_signals=key_signals,
        recommended_actions=recommended_actions,
        risk_level="medium",
        basis=request.context.basis,
        llm_used=False,
        fallback_used=True,
    )


def _invoke_llm_response(state: CopilotState) -> CopilotState:
    llm = get_llm()
    if not llm:
        state["fallback_used"] = True
        return state

    prompt = (
        "You are an AI retail copilot for inventory decisions. "
        "Answer only from provided context. Keep output concise and actionable.\n\n"
        f"User question: {state['question']}\n\n"
        f"Store context:\n{state['context_snapshot']}\n\n"
        "Return strict JSON with keys: summary, key_signals, recommended_actions, risk_level. "
        "risk_level must be one of: low, medium, high. "
        "key_signals and recommended_actions must be arrays of short strings."
    )

    try:
        raw = _as_text(llm.invoke(prompt).content)
        parsed = json.loads(raw)
        state["summary"] = str(parsed.get("summary", "")).strip()
        state["key_signals"] = [str(item).strip() for item in parsed.get("key_signals", []) if str(item).strip()]
        state["recommended_actions"] = [
            str(item).strip() for item in parsed.get("recommended_actions", []) if str(item).strip()
        ]
        risk = str(parsed.get("risk_level", "medium")).strip().lower()
        state["risk_level"] = risk if risk in {"low", "medium", "high"} else "medium"
        state["llm_used"] = True
    except Exception:
        state["fallback_used"] = True

    return state


def _finalize_copilot(state: CopilotState) -> CopilotState:
    if not state["summary"]:
        state["summary"] = "The Copilot could not generate a fully grounded response from the provided context."
    if not state["key_signals"]:
        state["key_signals"] = ["Insufficient high-confidence signals in context."]
    if not state["recommended_actions"]:
        state["recommended_actions"] = [
            "Validate forecast and restock priorities against latest store transactions.",
        ]
    return state


def _run_copilot_graph(initial_state: CopilotState) -> CopilotState:
    if StateGraph is None or END is None:
        return _finalize_copilot(initial_state)

    graph = StateGraph(CopilotState)
    graph.add_node("reason", _invoke_llm_response)
    graph.add_node("finalize", _finalize_copilot)
    graph.set_entry_point("reason")
    graph.add_edge("reason", "finalize")
    graph.add_edge("finalize", END)
    app = graph.compile()
    return cast(CopilotState, app.invoke(initial_state))


def generate_copilot_response(request: CopilotRequest) -> CopilotResponse:
    initial_state: CopilotState = {
        "question": request.question,
        "context_snapshot": _build_context_snapshot(request),
        "basis": request.context.basis,
        "summary": "",
        "key_signals": [],
        "recommended_actions": [],
        "risk_level": "medium",
        "llm_used": False,
        "fallback_used": False,
    }

    state = _run_copilot_graph(initial_state)

    if state["fallback_used"] and not state["llm_used"]:
        return _fallback_from_context(request)

    risk_level: Literal["low", "medium", "high"] = (
        cast(Literal["low", "medium", "high"], state["risk_level"])
        if state["risk_level"] in {"low", "medium", "high"}
        else "medium"
    )

    return CopilotResponse(
        summary=state["summary"],
        key_signals=state["key_signals"],
        recommended_actions=state["recommended_actions"],
        risk_level=risk_level,
        basis=request.context.basis,
        llm_used=state["llm_used"],
        fallback_used=state["fallback_used"],
    )


def _sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _chunk_text(text: str, size: int = 48) -> Iterator[str]:
    if not text:
        return
    for index in range(0, len(text), size):
        yield text[index : index + size]


def _stream_prompt(state: CopilotState) -> str:
    return (
        "You are an inventory copilot for a retail store. "
        "Use only the provided context and keep the answer concise. "
        "Return plain text with three headings exactly: Summary, Key Signals, Recommended Actions.\n\n"
        f"User question: {state['question']}\n\n"
        f"Store context:\n{state['context_snapshot']}"
    )


def _fallback_stream_text(request: CopilotRequest) -> tuple[str, dict[str, Any]]:
    fallback = _fallback_from_context(request)
    lines = [
        f"Summary\n{fallback.summary}",
        "Key Signals",
        *[f"- {signal}" for signal in fallback.key_signals],
        "Recommended Actions",
        *[f"- {action}" for action in fallback.recommended_actions],
    ]
    return "\n".join(lines), fallback.model_dump()


def generate_copilot_stream_events(request: CopilotRequest) -> Iterator[str]:
    initial_state: CopilotState = {
        "question": request.question,
        "context_snapshot": _build_context_snapshot(request),
        "basis": request.context.basis,
        "summary": "",
        "key_signals": [],
        "recommended_actions": [],
        "risk_level": "medium",
        "llm_used": False,
        "fallback_used": False,
    }

    yield _sse_event("start", {"basis": request.context.basis})

    llm = get_llm()
    if not llm:
        text, payload = _fallback_stream_text(request)
        for chunk in _chunk_text(text):
            yield _sse_event("token", {"text": chunk})
        yield _sse_event("done", payload)
        return

    full_text = ""
    try:
        for chunk in llm.stream(_stream_prompt(initial_state)):
            text = getattr(chunk, "content", "")
            if not text:
                continue
            if isinstance(text, list):
                text = "".join(str(item) for item in text)
            text = str(text)
            full_text += text
            yield _sse_event("token", {"text": text})

        if not full_text.strip():
            raise RuntimeError("Empty LLM stream response")

        yield _sse_event(
            "done",
            {
                "summary": full_text.strip(),
                "key_signals": [],
                "recommended_actions": [],
                "risk_level": "medium",
                "basis": request.context.basis,
                "llm_used": True,
                "fallback_used": False,
            },
        )
    except Exception:
        text, payload = _fallback_stream_text(request)
        for chunk in _chunk_text(text):
            yield _sse_event("token", {"text": chunk})
        yield _sse_event("done", payload)
