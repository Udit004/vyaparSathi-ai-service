from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


class CopilotForecastItem(BaseModel):
    product_name: str
    predicted_demand_7d: float
    trend_percent: float = 0


class CopilotRestockItem(BaseModel):
    product_name: str
    recommended_qty: float
    priority: str
    days_to_stockout: float | None = None


class CopilotAnomalyItem(BaseModel):
    product_name: str
    direction: str
    z_score: float = 0


class CopilotInsightItem(BaseModel):
    type: str
    title: str
    summary: str
    severity: str = "info"
    metrics: list[dict[str, Any]] = Field(default_factory=list)


class CopilotContext(BaseModel):
    store_name: str = "Store"
    basis: Literal["live", "demo-assisted", "model"] = "live"
    forecast_items: list[CopilotForecastItem] = Field(default_factory=list)
    restock_items: list[CopilotRestockItem] = Field(default_factory=list)
    anomalies: list[CopilotAnomalyItem] = Field(default_factory=list)
    insights: list[CopilotInsightItem] = Field(default_factory=list)


class CopilotRequest(BaseModel):
    question: str
    context: CopilotContext


class CopilotResponse(BaseModel):
    summary: str
    key_signals: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "medium"
    basis: Literal["live", "demo-assisted", "model"] = "live"
    llm_used: bool = False
    fallback_used: bool = False
