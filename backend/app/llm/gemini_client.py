from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings


@lru_cache
def get_judge_model() -> ChatGoogleGenerativeAI:
    """Only used by eval/judge.py — deliberately a different provider than `groq_client.
    get_chat_model`, which generates the reports being graded, so the eval harness isn't
    having the same model judge its own output."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_judge_model,
        google_api_key=settings.gemini_api_key,
        temperature=0,
        timeout=settings.request_timeout_seconds,
    )
