from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq

from app.config import settings


@lru_cache
def get_chat_model(temperature: float = 0.2) -> ChatGroq:
    """Shared ChatGroq factory. Cached per temperature so nodes reuse one client."""
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=temperature,
        timeout=settings.request_timeout_seconds,
    )
