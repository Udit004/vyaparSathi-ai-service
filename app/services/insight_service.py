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


def parse_llm_explanation(
    raw_content: str,
    fallback: dict[str, str],
    basis: str,
) -> InsightExplanationResponse:
    lines = [line.strip() for line in raw_content.splitlines() if line.strip()]

    parsed = {
        "title": lines[0] if len(lines) > 0 else fallback["title"],
        "summary": lines[1] if len(lines) > 1 else fallback["summary"],
        "recommendation": lines[2]
        if len(lines) > 2
        else fallback["recommendation"],
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
