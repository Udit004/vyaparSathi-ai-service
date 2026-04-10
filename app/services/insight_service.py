import re

from app.lib.llm import get_llm
from app.schemas.insight import InsightExplanationRequest, InsightExplanationResponse
from app.utils.forecasting import summarize_metrics


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
        key: " ".join(values).strip()
        for key, values in sections.items()
        if values
    }


def parse_llm_explanation(
    raw_content,
    fallback: dict[str, str],
    basis: str,
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
    }

    return InsightExplanationResponse(**parsed)


def generate_insight_explanation(
    request: InsightExplanationRequest,
) -> InsightExplanationResponse:
    fallback = summarize_metrics(request.subject, request.metrics, request.basis)
    llm = get_llm()

    if not llm:
        return InsightExplanationResponse(**fallback)

    prompt = build_explanation_prompt(request)

    try:
        response = llm.invoke(prompt)
        return parse_llm_explanation(response.content, fallback, request.basis)
    except Exception:
        return InsightExplanationResponse(**fallback)
