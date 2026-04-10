from pydantic import BaseModel, Field

from app.schemas.analytics import AnomalyInput, ForecastItemInput, InsightResult, RestockItemInput


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
    llmUsed: bool = False


class StoreInsightExplanationRequest(BaseModel):
    store_name: str
    basis: str = "live"
    forecast_items: list[ForecastItemInput] = Field(default_factory=list)
    restock_items: list[RestockItemInput] = Field(default_factory=list)
    anomalies: list[AnomalyInput] = Field(default_factory=list)
    insights: list[InsightResult] = Field(default_factory=list)
    products: list[dict] = Field(default_factory=list)
