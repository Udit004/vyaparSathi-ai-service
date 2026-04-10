from fastapi import APIRouter

from app.routes.analytics_routes import router as analytics_router
from app.routes.chat_routes import router as chat_router
from app.routes.forecast_routes import router as forecast_router
from app.routes.health_routes import router as health_router
from app.routes.insight_routes import router as insight_router


api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(chat_router)
api_router.include_router(forecast_router)
api_router.include_router(analytics_router)
api_router.include_router(insight_router)
