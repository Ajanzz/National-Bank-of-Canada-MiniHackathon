"""Configuration settings for the FastAPI application."""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    api_title: str = "SignalForge Trading Analytics API"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 7000
    
    # CORS
    allowed_origins: list[str] = ["http://localhost:5173", "http://localhost:3000", "http://localhost:*"]
    
    # CSV Upload
    max_csv_size_mb: int = 50
    
    # Analysis Config
    rolling_window_burst: int = 10  # minutes for overtrading burst detection
    recency_prev_trade_window: int = 5  # minutes for prev-trade reaction
    recency_streak_window: int = 10  # minutes for streak reaction
    recency_window_size: int = 50  # trades for recency window stride analysis
    recency_window_stride: int = 25  # trades
    revenge_max_size_multiplier: float = 1.25
    revenge_streak_size_multiplier: float = 1.1
    revenge_timing_window: int = 15  # minutes
    
    # OpenAI / LangChain Config - will be set by main.py's load_dotenv
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    langchain_tracing_v2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    
    class Config:
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    print(f"✓ Settings loaded: openai_api_key length = {len(settings.openai_api_key)}")
    if settings.openai_api_key and len(settings.openai_api_key) > 20:
        print(f"  First 25 chars: {settings.openai_api_key[:25]}")
    return settings
