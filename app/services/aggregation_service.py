from datetime import datetime, timedelta
from bson import ObjectId
from bson.errors import InvalidId
from app.config.database import get_database
from app.schemas.forecast import ForecastRequest, ForecastSeriesInput, DailyValue
from app.services.forecast_service import generate_forecast_response
from app.schemas.analytics import InsightResult
import logging


async def _resolve_store_id(db, store_id: str) -> ObjectId | None:
    """
    Resolve a store identifier to an ObjectId.

    Accepts either a valid 24-char hex ObjectId string or a store name.
    Returns None if not found.
    """
    try:
        return ObjectId(store_id)
    except (InvalidId, TypeError):
        pass

    doc = await db["stores"].find_one(
        {"name": store_id},
        {"_id": 1},
    )
    if doc:
        return doc["_id"]

    return None


async def get_store_products(store_id: str):
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return []
    cursor = db["products"].find({"store": store_oid, "isActive": True})
    products = await cursor.to_list(length=1000)
    for p in products:
        p["_id"] = str(p["_id"])
        p["store"] = str(p.get("store"))
    return products


async def get_store_sales(store_id: str, days: int = 30):
    db = get_database()
    store_oid = await _resolve_store_id(db, store_id)
    if not store_oid:
        return []
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    cursor = db["sales"].find({
        "store": store_oid,
        "completedAt": {"$gte": cutoff_date}
    })
    sales = await cursor.to_list(length=5000)
    for s in sales:
        s["_id"] = str(s["_id"])
        s["store"] = str(s.get("store"))
    return sales

def aggregate_sales_to_series(products, sales, days=30):
    date_keys = [(datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days-1, -1, -1)]
    date_idx_map = {d: i for i, d in enumerate(date_keys)}
    
    product_series_map = {}
    for p in products:
        product_series_map[p["_id"]] = {
            "store_id": p.get("store", "default"),
            "product_id": p["_id"],
            "product_name": p.get("name", "Unknown"),
            "current_stock": p.get("quantity", 0),
            "unit_price": p.get("price", 0),
            "values": [{"date": d, "quantity": 0, "revenue": 0} for d in date_keys]
        }
        
    for s in sales:
        completed_at = s.get("completedAt")
        if not completed_at: continue
        
        # completedAt might be string or datetime
        if isinstance(completed_at, str):
            try:
                date_str = completed_at[:10]
            except Exception:
                continue
        else:
            date_str = completed_at.strftime("%Y-%m-%d")
            
        idx = date_idx_map.get(date_str)
        if idx is None: continue
        
        for item in s.get("items", []):
            pid = str(item.get("productId"))
            if pid in product_series_map:
                product_series_map[pid]["values"][idx]["quantity"] += item.get("quantity", 0)
                product_series_map[pid]["values"][idx]["revenue"] += item.get("lineTotal", 0)

    # Convert to ForecastSeriesInput list
    series_inputs = []
    for pid, data in product_series_map.items():
        daily_values = [DailyValue(**v) for v in data["values"]]
        series_inputs.append(ForecastSeriesInput(
            store_id=data["store_id"],
            product_id=data["product_id"],
            product_name=data["product_name"],
            current_stock=data["current_stock"],
            unit_price=data["unit_price"],
            basis="live",
            values=daily_values
        ))
        
    return series_inputs

async def get_forecast_for_store(store_id: str):
    products = await get_store_products(store_id)
    sales = await get_store_sales(store_id, 30)
    series_inputs = aggregate_sales_to_series(products, sales, 30)
    
    request = ForecastRequest(horizon_days=7, series=series_inputs)
    response = generate_forecast_response(request)
    
    # Map results to match frontend format
    results_map = {r.product_id: r for r in response.results}
    
    final_output = []
    for s in series_inputs:
        res = results_map.get(s.product_id)
        if not res: continue
        
        predicted_daily = res.predicted_daily_demand
        days_to_stockout = None
        if predicted_daily > 0:
            days_to_stockout = round(s.current_stock / predicted_daily, 1)
            
        final_output.append({
            "productId": s.product_id,
            "productName": s.product_name,
            "currentStock": s.current_stock,
            "currentPrice": s.unit_price,
            "predictedDailyDemand": predicted_daily,
            "predictedDemand7d": res.predicted_demand_7d,
            "trendPercent": res.trend_percent,
            "confidence": res.confidence,
            "forecastSource": res.source,
            "daysToStockout": days_to_stockout
        })
        
    return final_output

async def get_restock_for_store(store_id: str):
    forecasts = await get_forecast_for_store(store_id)
    lead_time_days = 3
    safety_stock = 10
    
    restock_items = []
    for f in forecasts:
        recommended_stock = int(f["predictedDailyDemand"] * lead_time_days + safety_stock)
        recommended_qty = max(0, recommended_stock - f["currentStock"])
        
        priority = "green"
        if f["currentStock"] == 0 or (f["daysToStockout"] is not None and f["daysToStockout"] <= lead_time_days):
            priority = "red"
        elif recommended_qty > 0 or (f["daysToStockout"] is not None and f["daysToStockout"] <= lead_time_days * 2):
            priority = "yellow"
            
        restock_items.append({
            "productId": f["productId"],
            "productName": f["productName"],
            "currentStock": f["currentStock"],
            "predictedDemand7d": f["predictedDemand7d"],
            "recommendedQty": recommended_qty,
            "recommendedStock": recommended_stock,
            "leadTimeDays": lead_time_days,
            "priority": priority,
            "daysToStockout": f["daysToStockout"]
        })
        
    return restock_items


async def get_insights_for_store(store_id: str) -> list[InsightResult]:
    forecasts = await get_forecast_for_store(store_id)
    restock_items = await get_restock_for_store(store_id)
    products = await get_store_products(store_id)
    
    insights: list[InsightResult] = []
    
    if not forecasts:
        return insights
    
    sorted_forecasts = sorted(forecasts, key=lambda x: x["predictedDemand7d"], reverse=True)
    
    if sorted_forecasts:
        top = sorted_forecasts[0]
        insights.append(InsightResult(
            type="fastest_selling",
            title="Fastest Selling Product",
            summary=f"{top['productName']} has the highest predicted 7-day demand ({top['predictedDemand7d']:.1f} units).",
            severity="info",
            metrics=[{
                "productId": top["productId"],
                "productName": top["productName"],
                "predictedDemand7d": top["predictedDemand7d"],
                "predictedDailyDemand": top["predictedDailyDemand"],
                "trendPercent": top["trendPercent"]
            }],
            basis="live"
        ))
    
    slow_moving = [f for f in forecasts if f["predictedDailyDemand"] < 1 and f["currentStock"] > 0]
    if slow_moving:
        slow = slow_moving[0]
        insights.append(InsightResult(
            type="slow_moving",
            title="Slow Moving Inventory",
            summary=f"{slow['productName']} has low predicted daily demand ({slow['predictedDailyDemand']:.2f}) with {slow['currentStock']} units in stock.",
            severity="warning",
            metrics=[{
                "productId": slow["productId"],
                "productName": slow["productName"],
                "currentStock": slow["currentStock"],
                "predictedDailyDemand": slow["predictedDailyDemand"],
                "daysToStockout": slow.get("daysToStockout")
            }],
            basis="live"
        ))
    
    dead_stock = [f for f in forecasts if f["predictedDailyDemand"] == 0 and f["currentStock"] > 0]
    if dead_stock:
        dead = dead_stock[0]
        insights.append(InsightResult(
            type="dead_stock",
            title="Dead Stock Alert",
            summary=f"{dead['productName']} has zero predicted demand but {dead['currentStock']} units in stock. Consider discounting or removing.",
            severity="danger",
            metrics=[{
                "productId": dead["productId"],
                "productName": dead["productName"],
                "currentStock": dead["currentStock"],
                "predictedDailyDemand": 0
            }],
            basis="live"
        ))
    
    urgent_restock = [r for r in restock_items if r["priority"] == "red"]
    if urgent_restock:
        urgent = urgent_restock[0]
        insights.append(InsightResult(
            type="restock_priority",
            title="Urgent Restock Required",
            summary=f"{urgent['productName']} needs immediate restocking. Recommended: {urgent['recommendedQty']} units. Days to stockout: {urgent.get('daysToStockout', 'N/A')}",
            severity="danger",
            metrics=[{
                "productId": urgent["productId"],
                "productName": urgent["productName"],
                "currentStock": urgent["currentStock"],
                "recommendedQty": urgent["recommendedQty"],
                "daysToStockout": urgent.get("daysToStockout")
            }],
            basis="live"
        ))
    
    category_counts = {}
    for p in products:
        cat = p.get("category", "General")
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    if category_counts:
        top_category = max(category_counts, key=category_counts.get)
        insights.append(InsightResult(
            type="category_mix",
            title="Category Mix",
            summary=f"Store has {len(products)} products across {len(category_counts)} categories. Top category: {top_category} ({category_counts[top_category]} products).",
            severity="info",
            metrics=[{
                "category": k,
                "count": v
            } for k, v in category_counts.items()],
            basis="live"
        ))
    
    return insights
