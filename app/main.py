from fastapi import FastAPI

from app.routes.index import api_router
from app.services.forecast_service import _lazy_load_artifacts


def create_app() -> FastAPI:
    application = FastAPI(title="VyaparSathi AI Service")
    application.include_router(api_router)

    @application.on_event("startup")
    async def verify_forecast_artifacts() -> None:
        model, feature_columns = _lazy_load_artifacts()
        print(
            "[FORECAST DEBUG] startup verification status="
            f"{'model-loaded' if model is not None and feature_columns is not None else 'fallback-only'}"
        )

    return application


app = create_app()
