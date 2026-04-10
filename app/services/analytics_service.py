import logging
from datetime import datetime, timezone
import math

from app.schemas.analytics import (
    GenerateInsightsRequest,
    GenerateInsightsResponse,
    InsightResult,
    DetectAnomaliesRequest,
    DetectAnomaliesResponse,
    AnomalyResult,
)


LOGGER = logging.getLogger("uvicorn.error")
DEAD_STOCK_DAYS = 30


def compute_insights(
    request: GenerateInsightsRequest,
) -> GenerateInsightsResponse:
    """
    Generate actionable insights from forecast items, restock items, anomalies, and products.
    Ported from Express backend buildInsights().
    """
    
    forecast_items = request.forecast_items
    restock_items = request.restock_items
    anomalies = request.anomalies
    products = request.products
    
    insights = []
    
    # 1. Fastest selling products
    top_forecast = sorted(
        forecast_items,
        key=lambda x: x.predictedDemand7d,
        reverse=True
    )[:5]
    
    if top_forecast:
        insights.append(InsightResult(
            type="fastest_selling",
            title="Fastest selling products",
            summary=f"{top_forecast[0].productName} is projected to lead next-week demand with {top_forecast[0].predictedDemand7d:.0f} units.",
            severity="info",
            metrics=[
                {
                    "productId": item.productId,
                    "productName": item.productName,
                    "predictedDemand7d": item.predictedDemand7d,
                }
                for item in top_forecast
            ],
            basis="model" if any(item.basis == "model" for item in top_forecast)
                else "demo-assisted" if any(item.basis != "live" for item in top_forecast)
                else "live",
        ))
    
    # 2. Slow-moving inventory
    slow_movers = [
        item for item in forecast_items
        if item.totalQuantity30d > 0 and item.totalQuantity30d <= 5 and item.currentStock > 0
    ]
    slow_movers = sorted(slow_movers, key=lambda x: x.totalQuantity30d)[:5]
    
    if slow_movers:
        insights.append(InsightResult(
            type="slow_moving",
            title="Slow-moving inventory",
            summary=f"{slow_movers[0].productName} sold only {slow_movers[0].totalQuantity30d:.0f} units in the last 30 days.",
            severity="warning",
            metrics=[
                {
                    "productId": item.productId,
                    "productName": item.productName,
                    "soldLast30Days": item.totalQuantity30d,
                    "currentStock": item.currentStock,
                }
                for item in slow_movers
            ],
            basis="model" if any(item.basis == "model" for item in slow_movers)
                else "demo-assisted" if any(item.basis != "live" for item in slow_movers)
                else "live",
        ))
    
    # 3. Dead stock detection
    dead_stock_no_sales = [
        item for item in forecast_items
        if not item.lastSoldAt and item.currentStock > 0
    ]
    dead_stock_no_sales = sorted(dead_stock_no_sales, key=lambda x: x.currentStock, reverse=True)[:5]
    
    long_idle = []
    now = datetime.now(timezone.utc)
    for item in forecast_items:
        if item.lastSoldAt and item.currentStock > 0:
            try:
                last_sold_date = datetime.fromisoformat(item.lastSoldAt.replace('Z', '+00:00'))
                days_idle = (now - last_sold_date).days
                if days_idle > DEAD_STOCK_DAYS:
                    long_idle.append(item)
            except (ValueError, AttributeError):
                continue
    
    long_idle = sorted(long_idle, key=lambda x: x.currentStock, reverse=True)[:5]
    
    dead_stock_items = (dead_stock_no_sales + long_idle)[:5]
    if dead_stock_items:
        insights.append(InsightResult(
            type="dead_stock",
            title="Dead stock detection",
            summary=f"{dead_stock_items[0].productName} has stock on hand but no meaningful recent sales activity.",
            severity="danger",
            metrics=[
                {
                    "productId": item.productId,
                    "productName": item.productName,
                    "currentStock": item.currentStock,
                    "lastSoldAt": item.lastSoldAt,
                }
                for item in dead_stock_items
            ],
            basis="model" if any(item.basis == "model" for item in dead_stock_items)
                else "demo-assisted" if any(item.basis != "live" for item in dead_stock_items)
                else "live",
        ))
    
    # 4. Anomalies (sales spikes/drops)
    if anomalies:
        anomaly = anomalies[0]
        insights.append(InsightResult(
            type="sales_spike" if anomaly.direction == "spike" else "sales_drop",
            title="Sudden demand spike alert" if anomaly.direction == "spike" else "Sudden sales drop alert",
            summary=f"{anomaly.productName} shows a {anomaly.direction} versus its recent baseline.",
            severity="warning" if anomaly.direction == "spike" else "danger",
            metrics=[
                {
                    "productId": a.productId,
                    "productName": a.productName,
                    "direction": a.direction,
                    "zScore": a.zScore,
                }
                for a in anomalies[:5]
            ],
            basis="live",
        ))
    
    # 5. Category mix analysis
    if products:
        category_stats = {}
        for product in products:
            category = product.get("category", "General")
            if category not in category_stats:
                category_stats[category] = {"revenue": 0, "stock_value": 0}
            
            matching_forecast = next(
                (item for item in forecast_items if item.productId == str(product.get("_id", product.get("id")))),
                None
            )
            if matching_forecast:
                category_stats[category]["revenue"] += matching_forecast.totalRevenue30d
            category_stats[category]["stock_value"] += (
                product.get("quantity", 0) * product.get("price", 0)
            )
        
        total_revenue = sum(v["revenue"] for v in category_stats.values())
        total_stock_value = sum(v["stock_value"] for v in category_stats.values())
        
        category_mix = []
        for category, stats in category_stats.items():
            category_mix.append({
                "category": category,
                "revenueShare": (stats["revenue"] / total_revenue * 100) if total_revenue > 0 else 0,
                "stockShare": (stats["stock_value"] / total_stock_value * 100) if total_stock_value > 0 else 0,
            })
        
        category_mix = sorted(
            category_mix,
            key=lambda x: abs(x["revenueShare"] - x["stockShare"]),
            reverse=True
        )
        
        if category_mix:
            dominant = category_mix[0]
            insights.append(InsightResult(
                type="category_mix",
                title="Revenue versus stock mix",
                summary=f"{dominant['category']} contributes {dominant['revenueShare']:.0f}% of recent revenue versus {dominant['stockShare']:.0f}% of stock value.",
                severity="info",
                metrics=category_mix[:5],
                basis="model" if any(item.basis == "model" for item in forecast_items)
                    else "demo-assisted" if any(item.basis != "live" for item in forecast_items)
                    else "live",
            ))
    
    # 6. Urgent restock actions
    urgent_restock = [item for item in restock_items if item.priority == "red"][:5]
    if urgent_restock:
        insights.append(InsightResult(
            type="restock_priority",
            title="Urgent restock actions",
            summary=f"{urgent_restock[0].productName} should be reordered immediately to avoid a stockout.",
            severity="danger",
            metrics=[
                {
                    "productId": item.productId,
                    "productName": item.productName,
                    "currentStock": item.currentStock,
                    "recommendedQty": item.recommendedQty,
                    "priority": item.priority,
                }
                for item in urgent_restock
            ],
            basis="model" if any(item.basis == "model" for item in urgent_restock)
                else "demo-assisted" if any(item.basis != "live" for item in urgent_restock)
                else "live",
        ))
    
    LOGGER.info(
        "Generated insights count=%s basis_distribution=%s",
        len(insights),
        {insight.type: insight.basis for insight in insights}
    )
    
    return GenerateInsightsResponse(
        insights=insights,
        count=len(insights),
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def detect_anomalies(
    request: DetectAnomaliesRequest,
) -> DetectAnomaliesResponse:
    """
    Detect anomalies in product sales time series using statistical methods.
    Ported from Express backend detectAnomalies().
    """
    
    series_collection = request.series_collection
    min_score = request.min_anomaly_score
    anomalies = []
    
    for series in series_collection:
        quantities = [v.quantity for v in series.values]
        
        # Need at least 8 data points
        if len(quantities) < 8:
            continue
        
        # Split into recent (last 7 days) and baseline (prior days)
        recent_avg = sum(quantities[-7:]) / 7 if len(quantities) >= 7 else sum(quantities) / len(quantities)
        baseline = quantities[:-7] if len(quantities) > 7 else quantities
        baseline_avg = sum(baseline) / len(baseline) if baseline else 0
        
        # Calculate variance and standard deviation
        if baseline:
            variance = sum((v - baseline_avg) ** 2 for v in baseline) / len(baseline)
            std_dev = math.sqrt(variance)
        else:
            std_dev = 0
        
        direction = None
        score = 0.0
        
        # Determine if anomaly exists
        if std_dev > 0:
            score = (recent_avg - baseline_avg) / std_dev
            if score >= min_score:
                direction = "spike"
            elif score <= -min_score:
                direction = "drop"
        elif baseline_avg > 0:
            # Fallback: use ratio when std_dev is 0
            ratio = recent_avg / baseline_avg if baseline_avg > 0 else 1
            if ratio >= 1.5:
                direction = "spike"
            elif ratio <= 0.5:
                direction = "drop"
        
        if not direction:
            continue
        
        anomalies.append(AnomalyResult(
            productId=series.productId,
            productName=series.productName,
            direction=direction,
            zScore=round(score, 2),
            recentAverage=round(recent_avg, 2),
            baselineAverage=round(baseline_avg, 2),
            basis=series.basis,
        ))
    
    # Sort by absolute z-score (strongest anomalies first)
    anomalies = sorted(anomalies, key=lambda x: abs(x.zScore), reverse=True)
    
    LOGGER.info(
        "Detected anomalies count=%s spike=%s drop=%s",
        len(anomalies),
        sum(1 for a in anomalies if a.direction == "spike"),
        sum(1 for a in anomalies if a.direction == "drop"),
    )
    
    return DetectAnomaliesResponse(
        anomalies=anomalies,
        count=len(anomalies),
        computed_at=datetime.now(timezone.utc).isoformat(),
    )
