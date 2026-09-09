import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
import structlog

from app.config.logging import configure_logging
from app.config.database import close_connection, get_database
from app.routes.index import api_router
from app.services.forecast_service import _lazy_load_artifacts


@asynccontextmanager
async def lifespan(application: FastAPI):
    configure_logging()
    application.state.db = get_database()
    model, feature_columns = _lazy_load_artifacts()
    application.state.forecast_model_loaded = model is not None and feature_columns is not None
    yield
    await close_connection()


def create_app() -> FastAPI:
    application = FastAPI(title="VyaparSathi AI Service", lifespan=lifespan)
    application.include_router(api_router)

    @application.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        logger = structlog.get_logger("vyaparsathi.ai.http")

        logger.info(
            "http_request_start",
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            client_host=request.client.host if request.client else None,
        )

        response = await call_next(request)

        process_time = (time.time() - start_time) * 1000
        logger.info(
            "http_request_complete",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(process_time, 2),
        )

        return response

    @application.get("/metrics", tags=["monitoring"])
    async def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()