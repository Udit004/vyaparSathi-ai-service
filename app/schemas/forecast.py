from pydantic import BaseModel, Field


class DailyValue(BaseModel):
    date: str
    quantity: float = 0
    revenue: float = 0


class ForecastSeriesInput(BaseModel):
    product_id: str
    product_name: str
    current_stock: float = 0
    unit_price: float = 0
    basis: str = "live"
    values: list[DailyValue] = Field(default_factory=list)


class ForecastRequest(BaseModel):
    horizon_days: int = 7
    series: list[ForecastSeriesInput] = Field(default_factory=list)


class ForecastResult(BaseModel):
    product_id: str
    product_name: str
    predicted_daily_demand: float
    predicted_demand_7d: float
    trend_percent: float
    confidence: str


class ForecastResponse(BaseModel):
    results: list[ForecastResult]
