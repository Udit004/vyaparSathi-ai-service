import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
import structlog

from app.config.logging import configure_logging
from app.config.database import close_connection, get_database
from app.config.settings import get_settings
from app.routes.index import api_router
from app.services.forecast_service import _lazy_load_artifacts
from app.agent.checkpointer import get_checkpointer, close_checkpointer


def _configure_langsmith():
    settings = get_settings()
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    if settings.langchain_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_VERBOSE"] = str(settings.langchain_verbose).lower()
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2).lower()


@asynccontextmanager
async def lifespan(application: FastAPI):
    configure_logging()
    _configure_langsmith()
    application.state.db = get_database()
    model, feature_columns = _lazy_load_artifacts()
    application.state.forecast_model_loaded = model is not None and feature_columns is not None
    # Warm up the MongoDB checkpointer connection at startup
    await get_checkpointer()
    yield
    await close_connection()
    await close_checkpointer()



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