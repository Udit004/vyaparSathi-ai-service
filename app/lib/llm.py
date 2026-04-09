from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config.settings import get_settings


@lru_cache()
def get_llm():
    settings = get_settings()
    if not settings.gemini_api_key:
        return None

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
    )
