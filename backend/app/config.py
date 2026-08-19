from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ai-stock-research-agent"
    environment: str = "development"

    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    finnhub_api_key: str = ""
    twelvedata_api_key: str = ""

    max_tickers: int = 5
    request_timeout_seconds: int = 20

    log_dir: Path = REPO_ROOT / "logs"
    log_level: str = "INFO"

    data_dir: Path = BACKEND_DIR / "data"
    session_ttl_seconds: int = 3600

    cors_origins: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
