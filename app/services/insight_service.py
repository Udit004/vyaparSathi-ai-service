import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.lib.llm import get_llm
from app.schemas.insight import (
    InsightExplanationRequest,
    InsightExplanationResponse,
    StoreInsightExplanationRequest,
)
from app.utils.forecasting import summarize_metrics


LOGGER = logging.getLogger("uvicorn.error")


def _invoke_llm_with_system_prompt(system_prompt: str, user_prompt: str):
    llm = get_llm()

    if not llm:
        return None

    try:
        return llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
    except Exception as exc:
        LOGGER.warning("Gemini invocation failed, switching to local fallback: %s", exc)
        return None


def build_explanation_prompt(request: InsightExplanationRequest) -> str:
    return f"""
You are helping an inventory decision-support system explain one product insight.
Subject: {request.subject}
Insight type: {request.insight_type}
Basis: {request.basis}
Metrics: {request.metrics}

Return three short sections:
1. Title
2. Summary
3. Recommendation

Be concrete, operational, and avoid hype.
""".strip()


def build_store_explanation_prompt(request: StoreInsightExplanationRequest) -> str:
    payload = {
        "store_name": request.store_name,
        "basis": request.basis,
        "forecast_items": [item.model_dump() for item in request.forecast_items],
        "restock_items": [item.model_dump() for item in request.restock_items],
        "anomalies": [item.model_dump() for item in request.anomalies],
        "insights": [item.model_dump() for item in request.insights],
        "products": request.products,
    }

    return json.dumps(payload, ensure_ascii=False, default=str, indent=2)


def _normalize_content(raw_content) -> str:
    if raw_content is None:
        return ""
    if isinstance(raw_content, str):
        return raw_content
    return str(raw_content)


def _parse_heading_key(line: str) -> str | None:
    normalized = line.strip()
    normalized = re.sub(r"^#+\s*", "", normalized)
    normalized = re.sub(r"^\d+\.?\s*", "", normalized)
    normalized = normalized.replace("**", "").strip().rstrip(":").strip()

    lowered = normalized.lower()
    if lowered == "title":
        return "title"
    if lowered == "summary":
        return "summary"
    if lowered == "recommendation":
        return "recommendation"
    return None


def _extract_sections(raw_content: str) -> dict[str, str]:
    lines = raw_content.replace("\r\n", "\n").split("\n")
    sections: dict[str, list[str]] = {}
    current_key: str | None = None

    for line in lines:
        heading_key = _parse_heading_key(line)
        if heading_key:
            current_key = heading_key
            sections.setdefault(current_key, [])
            continue

        if current_key is None:
            continue

        stripped = line.strip()
        if stripped:
            sections[current_key].append(stripped)

    return {
        key: "\n".join(values).strip()
        for key, values in sections.items()
        if values
    }


def parse_llm_explanation(
    raw_content,
    fallback: dict[str, str],
    basis: str,
    llm_used: bool = True,
) -> InsightExplanationResponse:
    normalized = _normalize_content(raw_content)
    sections = _extract_sections(normalized)

    title = sections.get("title", "").strip()
    summary = sections.get("summary", "").strip()
    recommendation = sections.get("recommendation", "").strip()

    if not (title and summary and recommendation):
        # Fallback for plain unstructured LLM output
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        content_lines = [line for line in lines if not _parse_heading_key(line)]
        if content_lines:
            lines = content_lines
        title = title or (lines[0] if len(lines) > 0 else "")
        summary = summary or (lines[1] if len(lines) > 1 else "")
        recommendation = recommendation or (lines[2] if len(lines) > 2 else "")

    parsed = {
        "title": title or fallback["title"],
        "summary": summary or fallback["summary"],
        "recommendation": recommendation or fallback["recommendation"],
        "basis": basis,
        "llmUsed": llm_used,
    }

    return InsightExplanationResponse(**parsed)


def generate_insight_explanation(
    request: InsightExplanationRequest,
) -> InsightExplanationResponse:
    fallback = summarize_metrics(request.subject, request.metrics, request.basis)
    fallback["llmUsed"] = False

    system_prompt = (
        "You are a senior inventory analyst. Use the provided product data only. "
        "Write a concise, practical explanation with exactly three sections: Title, Summary, Recommendation. "
        "Avoid hype and do not mention internal implementation details."
    )
    prompt = build_explanation_prompt(request)
    response = _invoke_llm_with_system_prompt(system_prompt, prompt)

    if response is None:
        return InsightExplanationResponse(**fallback)

    try:
        return parse_llm_explanation(response.content, fallback, request.basis)
    except Exception:
        return InsightExplanationResponse(**fallback)


def _build_store_fallback(request: StoreInsightExplanationRequest) -> InsightExplanationResponse:
    forecast_items = sorted(
        request.forecast_items,
        key=lambda item: item.predictedDemand7d,
        reverse=True,
    )
    urgent_restock = [item for item in request.restock_items if item.priority == "red"]
    anomaly_count = len(request.anomalies)
    top_product = forecast_items[0].productName if forecast_items else request.store_name
    summary_bits = [
        f"{len(request.products)} products analysed",
        f"{len(forecast_items)} forecasted items",
        f"{len(urgent_restock)} urgent restock items",
        f"{anomaly_count} anomalies detected",
    ]

    recommendation = (
        f"Prioritize {urgent_restock[0].productName} immediately. "
        if urgent_restock
        else "No urgent restock action is required based on the current forecast window. "
    )
    recommendation += (
        "Review low-selling products and compare live sales against forecasted demand to refine ordering."
    )

    return InsightExplanationResponse(
        title=f"{request.store_name} sales overview",
        summary=
            f"{request.store_name} shows {', '.join(summary_bits)}. "
            f"Top demand is currently centered on {top_product}.",
        recommendation=recommendation,
        basis=request.basis,
        llmUsed=False,
    )


def generate_store_insight_explanation(
    request: StoreInsightExplanationRequest,
) -> InsightExplanationResponse:
    system_prompt = (
        "You are a senior retail analyst. Summarize the full store performance using the provided JSON context. "
        "Your answer must be practical, human readable, and based only on the supplied data. "
        "Return exactly three sections: Title, Summary, Recommendation. "
        "In the summary, synthesize demand, restock risk, anomalies, and category patterns at store level. "
        "In the recommendation, give concrete next actions for the store owner."
    )
    prompt = build_store_explanation_prompt(request)
    response = _invoke_llm_with_system_prompt(system_prompt, prompt)

    fallback = _build_store_fallback(request)

    if response is None:
        return fallback

    try:
        return parse_llm_explanation(response.content, fallback.model_dump(), request.basis)
    except Exception:
        return fallback
