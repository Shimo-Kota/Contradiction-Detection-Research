"""
Configuration Settings for Contradiction Detection System.

This module defines configuration settings for the application, including:
- Default LLM provider selection
- API keys and endpoints for multiple LLM providers
- Default model selections for each provider
- Request timeout and concurrency limits

Settings can be overridden through environment variables or .env file.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    """
    Application settings with LLM provider configurations.
    
    Attributes:
        provider: Default LLM provider to use
        anthropic_api_key: Authentication key for Anthropic Claude API
        anthropic_model: Default Claude model to use
        google_api_key: Authentication key for Google Gemini API
        google_model: Default Gemini model to use
        openai_api_key: Authentication key for OpenAI API
        openai_model: Default OpenAI model to use
        request_timeout: API request timeout in seconds
        max_concurrency: Maximum number of concurrent API requests
    """
    provider: Literal["anthropic", "google", "openai"] = "anthropic"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    google_api_key: str | None = None
    google_model: str = "gemini-2.0-flash"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-2024-11-20"
    request_timeout: int = 180
    max_concurrency: int = 32
    model_config = SettingsConfigDict(env_file=".env", env_prefix="")

@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses functools.lru_cache to avoid re-reading settings on every access.
    
    Returns:
        Settings object with application configuration
    """
    return Settings()