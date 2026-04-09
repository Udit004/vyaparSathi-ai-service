from fastapi import FastAPI

from app.routes.index import api_router


def create_app() -> FastAPI:
    application = FastAPI(title="VyaparSathi AI Service")
    application.include_router(api_router)
    return application


app = create_app()
