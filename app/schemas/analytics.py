from pydantic import BaseModel, Field


class ForecastItemInput(BaseModel):
    """Forecast item data from Express backend"""
    productId: str
    productName: str
    category: str = "General"
    currentStock: float = 0
    currentPrice: float = 0
    predictedDailyDemand: float = 0
    predictedDemand7d: float = 0
    trendPercent: float = 0
    confidence: str = "low"
    forecastSource: str = "unknown"
    basis: str = "demo-assisted"
    totalQuantity30d: float = 0
    totalRevenue30d: float = 0
    lastSoldAt: str | None = None
    soldDays: int = 0
    daysToStockout: float | None = None


class RestockItemInput(BaseModel):
    """Restock item data from Express backend"""
    productId: str
    productName: str
    currentStock: float = 0
    predictedDemand7d: float = 0
    recommendedQty: float = 0
    recommendedStock: float = 0
    reorderDate: str | None = None
    leadTimeDays: int = 3
    priority: str = "green"
    daysToStockout: float | None = None
    basis: str = "demo-assisted"


class AnomalyInput(BaseModel):
    """Anomaly detection data from Express"""
    productId: str
    productName: str
    direction: str  # "spike" or "drop"
    zScore: float = 0
    recentAverage: float = 0
    baselineAverage: float = 0
    basis: str = "live"


class SeriesValueInput(BaseModel):
    """Time series daily value"""
    date: str
    quantity: float
    revenue: float


class SeriesCollectionInput(BaseModel):
    """Product time series for anomaly detection"""
    productId: str
    productName: str
    category: str = "General"
    currentStock: float = 0
    unitPrice: float = 0
    values: list[SeriesValueInput] = Field(default_factory=list)
    totalQuantity30d: float = 0
    totalRevenue30d: float = 0
    soldDays: int = 0
    lastSoldAt: str | None = None
    basis: str = "demo-assisted"


class InsightMetricItem(BaseModel):
    """Base metric item in insight"""
    productId: str | None = None
    productName: str | None = None


class InsightResult(BaseModel):
    """Individual insight result"""
    type: str  # "fastest_selling", "slow_moving", "dead_stock", "category_mix", "restock_priority"
    title: str
    summary: str
    severity: str  # "info", "warning", "danger"
    metrics: list[dict] = Field(default_factory=list)
    basis: str = "live"


class GenerateInsightsRequest(BaseModel):
    """Request to generate insights"""
    forecast_items: list[ForecastItemInput] = Field(default_factory=list)
    restock_items: list[RestockItemInput] = Field(default_factory=list)
    anomalies: list[AnomalyInput] = Field(default_factory=list)
    products: list[dict] = Field(default_factory=list)  # Simplified product data


class GenerateInsightsResponse(BaseModel):
    """Response with generated insights"""
    insights: list[InsightResult]
    count: int
    computed_at: str


class DetectAnomaliesRequest(BaseModel):
    """Request to detect anomalies"""
    series_collection: list[SeriesCollectionInput] = Field(default_factory=list)
    min_anomaly_score: float = 2.0  # Z-score threshold


class AnomalyResult(BaseModel):
    """Detected anomaly"""
    productId: str
    productName: str
    direction: str  # "spike" or "drop"
    zScore: float
    recentAverage: float
    baselineAverage: float
    basis: str = "live"


class DetectAnomaliesResponse(BaseModel):
    """Response with detected anomalies"""
    anomalies: list[AnomalyResult] = Field(default_factory=list)
    count: int
    computed_at: str
