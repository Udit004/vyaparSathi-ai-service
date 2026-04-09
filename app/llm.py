from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import get_settings


def get_llm():
    settings = get_settings()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.gemini_api_key,
        temperature=0.3,
    )

    return llm