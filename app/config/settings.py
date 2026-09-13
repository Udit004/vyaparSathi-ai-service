from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str | None = None
    mongo_url: str = "mongodb://localhost:27017"
    mongo_db_name: str = "vyapar-sathi"
    langchain_api_key: str | None = None
    langchain_project: str = "vyaparSathi-ai"
    langchain_verbose: bool = True
    langchain_tracing_v2: bool = True
    langgraph_checkpointer_ttl_seconds: int = 604800
    # Mem0 long-term memory
    mem0_api_key: str | None = None
    mem0_enabled: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
