from functools import lru_cache
from dotenv import load_dotenv

from pydantic_settings import BaseSettings

load_dotenv()


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
    # Small/fast summarizer providers (OpenAI-compatible endpoints)
    groq_api_key: str | None = None
    nvidia_api_key: str | None = None
    # Vector database (Pinecone)
    pinecone_api_key: str | None = None
    pincone_api_key: str | None = None
    pinecone_index_name: str = "vyapar-sathi"

    @property
    def effective_pinecone_api_key(self) -> str | None:
        return self.pinecone_api_key or self.pincone_api_key

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
