from pydantic import BaseModel, Field


class InsightExplanationRequest(BaseModel):
    insight_type: str
    subject: str
    basis: str = "live"
    metrics: dict = Field(default_factory=dict)


class InsightExplanationResponse(BaseModel):
    title: str
    summary: str
    recommendation: str
    basis: str
