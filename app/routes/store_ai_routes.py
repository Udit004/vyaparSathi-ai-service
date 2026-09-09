from fastapi import APIRouter, Request
from app.services.aggregation_service import get_forecast_for_store, get_restock_for_store, get_insights_for_store, get_store_products
from app.controllers.chat_controller import copilot_chat_stream
from pydantic import BaseModel
from typing import Any

router = APIRouter(tags=["store_ai"])

class ApiResponse(BaseModel):
    data: Any
    message: str
    statusCode: int = 200

@router.get("/{store_id}/forecast", response_model=ApiResponse)
async def get_forecast(store_id: str):
    data = await get_forecast_for_store(store_id)
    return ApiResponse(data=data, message="Forecast generated successfully", statusCode=200)

@router.get("/{store_id}/restock", response_model=ApiResponse)
async def get_restock(store_id: str):
    data = await get_restock_for_store(store_id)
    return ApiResponse(data=data, message="Restock plan generated successfully", statusCode=200)

@router.get("/{store_id}/insights", response_model=ApiResponse)
async def get_insights(store_id: str):
    data = await get_insights_for_store(store_id)
    return ApiResponse(data=data, message="Insights generated successfully", statusCode=200)

@router.get("/{store_id}/summary", response_model=ApiResponse)
async def get_summary(store_id: str):
    # Simplified placeholder for summary
    data = {"title": "Store summary", "summary": "Store summary data placeholder."}
    return ApiResponse(data=data, message="Store summary generated successfully", statusCode=200)

@router.get("/{store_id}/product/{product_id}", response_model=ApiResponse)
async def get_product_insight(store_id: str, product_id: str):
    data = {"productId": product_id, "insight": "No specific insight available in local fallback."}
    return ApiResponse(data=data, message="Product insight generated successfully", statusCode=200)

@router.post("/{store_id}/copilot", response_model=ApiResponse)
async def get_copilot(store_id: str):
    data = {"answer": "Copilot placeholder."}
    return ApiResponse(data=data, message="Copilot response generated successfully", statusCode=200)


@router.post("/{store_id}/copilot/stream")
async def get_copilot_stream(store_id: str, request: Request):
    from app.schemas.chat import CopilotRequest, CopilotContext
    
    body = await request.json()
    question = body.get("question", "")
    
    forecasts = await get_forecast_for_store(store_id)
    restocks = await get_restock_for_store(store_id)
    insights = await get_insights_for_store(store_id)
    products = await get_store_products(store_id)
    
    store_name = products[0].get("store", "Store") if products else "Store"
    
    context = CopilotContext(
        store_name=store_name,
        basis="live",
        forecast_items=[
            {"product_name": f["productName"], "predicted_demand_7d": f["predictedDemand7d"], "trend_percent": f.get("trendPercent", 0)}
            for f in forecasts
        ],
        restock_items=[
            {"product_name": r["productName"], "recommended_qty": r["recommendedQty"], "priority": r["priority"], "days_to_stockout": r.get("daysToStockout")}
            for r in restocks
        ],
        anomalies=[],
        insights=[
            {"type": i.type, "title": i.title, "summary": i.summary, "severity": i.severity, "metrics": i.metrics}
            for i in insights
        ],
    )
    
    copilot_request = CopilotRequest(question=question, context=context)
    return await copilot_chat_stream(copilot_request)
