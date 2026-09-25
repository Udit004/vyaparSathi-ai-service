"""
app/lib/pinecone.py
===================
Pinecone vector store integration helper for VyaparSathi AI Service.

Provides:
- get_pinecone_client(): Returns initialized Pinecone client instance.
- get_vector_store(): Returns a LangChain PineconeVectorStore for semantic search.
"""
from __future__ import annotations

import os
from typing import Any

import structlog
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config.settings import get_settings

LOGGER = structlog.get_logger("vyaparsathi.ai.lib.pinecone")

_pinecone_client: Pinecone | None = None


def get_pinecone_client() -> Pinecone | None:
    """
    Initialize and return Pinecone client singleton using PINCONE_API_KEY.
    """
    global _pinecone_client

    if _pinecone_client is not None:
        return _pinecone_client

    settings = get_settings()
    api_key = getattr(settings, "effective_pinecone_api_key", None) or settings.pinecone_api_key or os.getenv("PINECONE_API_KEY") or os.getenv("PINCONE_API_KEY")

    if not api_key:
        LOGGER.warning("pinecone_api_key_missing", message="PINECONE_API_KEY not configured.")
        return None

    try:
        _pinecone_client = Pinecone(api_key=api_key)
        LOGGER.info("pinecone_client_initialized")
        return _pinecone_client
    except Exception as e:
        LOGGER.exception("pinecone_init_failed", error=str(e))
        return None


def get_vector_store(
    index_name: str | None = None,
    embedding_model: str = "models/gemini-embedding-001",
) -> PineconeVectorStore | None:
    """
    Get or create a LangChain PineconeVectorStore wrapper.

    Args:
        index_name: Name of the Pinecone index (defaults to settings.pinecone_index_name).
        embedding_model: Gemini embedding model to use for vector embeddings.

    Returns:
        PineconeVectorStore instance or None if API key/index is unavailable.
    """
    settings = get_settings()
    idx_name = index_name or settings.pinecone_index_name
    api_key = settings.pinecone_api_key or os.getenv("PINCONE_API_KEY")
    gemini_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY")

    if not api_key:
        LOGGER.warning("pinecone_api_key_missing", message="PINCONE_API_KEY is not set.")
        return None

    if not gemini_key:
        LOGGER.warning("gemini_api_key_missing", message="GEMINI_API_KEY is required for embeddings.")
        return None

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=gemini_key,
        )

        vector_store = PineconeVectorStore(
            index_name=idx_name,
            embedding=embeddings,
            pinecone_api_key=api_key,
        )

        LOGGER.info("pinecone_vector_store_initialized", index_name=idx_name)
        return vector_store
    except Exception as e:
        LOGGER.exception("pinecone_vector_store_init_failed", index_name=idx_name, error=str(e))
        return None


def ensure_index_exists(
    index_name: str | None = None,
    dimension: int = 3072,
    metric: str = "cosine",
    cloud: str = "aws",
    region: str = "us-east-1",
) -> bool:
    """
    Ensure that a serverless Pinecone index exists with the specified dimension.

    Args:
        index_name: Name of index (defaults to settings.pinecone_index_name).
        dimension: Embedding dimension (768 for models/text-embedding-004).
        metric: Distance metric ('cosine', 'euclidean', or 'dotproduct').
        cloud: Cloud provider for serverless index ('aws' or 'gcp').
        region: Cloud region ('us-east-1').

    Returns:
        True if index exists or was created successfully.
    """
    client = get_pinecone_client()
    if not client:
        return False

    settings = get_settings()
    idx_name = index_name or settings.pinecone_index_name

    try:
        existing_indexes = [i.name for i in client.list_indexes()]
        if idx_name in existing_indexes:
            LOGGER.info("pinecone_index_exists", index_name=idx_name)
            return True

        LOGGER.info("creating_pinecone_index", index_name=idx_name, dimension=dimension)
        client.create_index(
            name=idx_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        LOGGER.info("pinecone_index_created", index_name=idx_name)
        return True
    except Exception as e:
        LOGGER.exception("ensure_pinecone_index_failed", index_name=idx_name, error=str(e))
        return False
