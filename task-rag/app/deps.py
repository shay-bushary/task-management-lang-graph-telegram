"""Dependency injection helpers for FastAPI."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever

from .config import Settings, settings


@lru_cache()
def get_settings() -> Settings:
    """Get application settings (cached)."""
    return settings


def get_llm(settings: Annotated[Settings, Depends(get_settings)]) -> ChatOpenAI:
    """Get OpenAI LLM instance."""
    return ChatOpenAI(
        model=settings.model_name,
        api_key=settings.openai_api_key,
        temperature=0.7,
        streaming=True,
    )


def get_embeddings(
    settings: Annotated[Settings, Depends(get_settings)],
) -> OpenAIEmbeddings:
    """Get OpenAI embeddings instance."""
    return OpenAIEmbeddings(
        model=settings.embeddings_model,
        api_key=settings.openai_api_key,
    )


def get_vectorstore(
    settings: Annotated[Settings, Depends(get_settings)],
    embeddings: Annotated[OpenAIEmbeddings, Depends(get_embeddings)],
) -> Chroma:
    """Get Chroma vector store instance."""
    # Ensure the chroma directory exists
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        persist_directory=str(settings.chroma_dir),
        embedding_function=embeddings,
        collection_name="task_documents",
    )


def get_retriever(
    settings: Annotated[Settings, Depends(get_settings)],
    vectorstore: Annotated[Chroma, Depends(get_vectorstore)],
) -> VectorStoreRetriever:
    """Get vector store retriever instance."""
    return vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": settings.retrieval_k}
    )


def ensure_directories(settings: Annotated[Settings, Depends(get_settings)]) -> None:
    """Ensure required directories exist."""
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
