"""
LLM API Client Implementations.

This module provides a unified interface for interacting with multiple LLM providers:
- Anthropic (Claude models)
- Google (Gemini models)
- OpenAI (GPT models)
"""
import httpx
import asyncio
from contextlib import asynccontextmanager
from app.core.config import get_settings
import logging

_settings = get_settings()
logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # Base retry delay in seconds

# Semaphore (global) to limit concurrent LLM API calls
llm_semaphore = asyncio.Semaphore(25)

@asynccontextmanager
async def _anthropic_client():
    """
    Create an async HTTP client configured for Anthropic Claude API.
    
    Uses httpx AsyncClient with appropriate authentication headers,
    timeout settings, and connection limits from configuration.
    """
    async with httpx.AsyncClient(
        base_url="https://api.anthropic.com/v1",
        headers={
            "x-api-key": _settings.anthropic_api_key,
            "anthropic-version": "2023-06-01"
        },
        timeout=_settings.request_timeout,
        limits=httpx.Limits(max_connections=_settings.max_concurrency),
        http2=False,
    ) as client:
        yield client

@asynccontextmanager
async def _google_client():
    """
    Create an async HTTP client configured for Google Gemini API.
    
    Uses httpx AsyncClient with API key as query parameter,
    timeout settings, and connection limits from configuration.
    """
    async with httpx.AsyncClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        params={"key": _settings.google_api_key},
        timeout=_settings.request_timeout,
        limits=httpx.Limits(max_connections=_settings.max_concurrency),
        http2=False,
    ) as client:
        yield client

@asynccontextmanager
async def _openai_client():
    """
    Create an async HTTP client configured for OpenAI API.
    
    Uses httpx AsyncClient with appropriate authentication headers,
    timeout settings, and connection limits from configuration.
    """
    async with httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        headers={
            "Authorization": f"Bearer {_settings.openai_api_key}",
            "Content-Type": "application/json"
        },
        timeout=_settings.request_timeout,
        limits=httpx.Limits(max_connections=_settings.max_concurrency),
        http2=False,
    ) as client:
        yield client

async def chat_completion(messages: list[dict], model: str | None = None, provider: str | None = None, **kw) -> str:
    """
    Unified chat completion function supporting multiple LLM providers.
    
    This function serves as the primary interface for all LLM interactions in the
    contradiction detection system. It routes requests to the appropriate provider-specific
    implementation based on configuration or explicit provider parameter.
    
    As noted in Section 4.2 of the paper, different providers may respond differently
    to the same prompts, particularly with Chain-of-Thought reasoning.
    
    Args:
        messages: List of message dictionaries in ChatML format
        model: Specific model to use (overrides config default)
        provider: Provider to use (overrides config default)
        **kw: Additional keyword arguments (e.g., temperature)
        
    Returns:
        Generated text response from the model
        
    Raises:
        ValueError: If the specified provider is not supported
    """
    use_provider = provider or _settings.provider
    async with llm_semaphore:
        if use_provider == "anthropic":
            return await _anthropic_chat_completion(messages, model, **kw)
        elif use_provider == "google":
            return await _google_chat_completion(messages, model, **kw)
        elif use_provider == "openai":
            return await _openai_chat_completion(messages, model, **kw)
        else:
            raise ValueError(f"Unsupported provider: {use_provider}")
            raise ValueError(f"Unsupported provider: {use_provider}")

async def _anthropic_chat_completion(messages: list[dict], model: str | None = None, **kw) -> str:
    """
    Call Anthropic's Claude API with retry logic.
    
    Converts ChatML format to Anthropic's API format. Implements exponential
    backoff for rate limits and server errors.
    
    As noted in the paper, Claude models showed particularly strong performance
    when using Chain-of-Thought reasoning compared to other model families.
    
    Args:
        messages: List of message dictionaries in ChatML format
        model: Specific Claude model to use (e.g., "claude-3-7-sonnet-latest")
        **kw: Additional API parameters
        
    Returns:
        Generated text response from the model
        
    Raises:
        httpx.HTTPStatusError: On non-retryable HTTP errors
        Exception: On other errors
    """
    # Convert ChatML format to Anthropic's format
    system_message = ""
    anthropic_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_message += msg["content"] + "\\n"
        elif msg["role"] == "user":
            anthropic_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            anthropic_messages.append({"role": "assistant", "content": msg["content"]})
    
    payload = {
        "model": model or _settings.anthropic_model,
        "max_tokens": kw.get("max_tokens", 1024), # Add max_tokens, default to 1024
        "temperature": kw.get("temperature", 0),
        "messages": anthropic_messages,
    }
    print(f"prompt: {anthropic_messages}")
    if system_message:
        payload["system"] = system_message.strip()
    
    for attempt in range(MAX_RETRIES):
        try:
            async with _anthropic_client() as c:
                r = await c.post("/messages", json=payload)
                r.raise_for_status()
                # Corrected token usage keys
                print(f"Input tokens: {r.json()['usage']['input_tokens']}")
                print(f"Output tokens: {r.json()['usage']['output_tokens']}")
                # Anthropic API does not return total_tokens in the same way, calculate if needed or log separately
                return r.json()["content"][0]["text"]
        except httpx.HTTPStatusError as e:
            # Retry on rate limits (429) or server errors (5xx)
            if e.response.status_code in [429, 500, 502, 503, 504] and attempt < MAX_RETRIES - 1:
                retry_delay = RETRY_DELAY * (attempt + 1)  # Exponential backoff
                logger.warning(f"Anthropic API error {e.response.status_code}, retrying in {retry_delay}s ({attempt+1}/{MAX_RETRIES})...")
                await asyncio.sleep(retry_delay)
                continue
            logger.error(f"Anthropic API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Anthropic API: {e}")
            raise

async def _google_chat_completion(messages: list[dict], model: str | None = None, **kw) -> str:
    """
    Call Google's Gemini API with retry logic.
    
    Converts ChatML format to Gemini's API format. Implements exponential
    backoff for rate limits and server errors.
    
    Args:
        messages: List of message dictionaries in ChatML format
        model: Specific Gemini model to use (e.g., "gemini-2.5-pro-preview-05-06")
        **kw: Additional API parameters
        
    Returns:
        Generated text response from the model
        
    Raises:
        httpx.HTTPStatusError: On non-retryable HTTP errors
        Exception: On other errors
    """
    # Convert ChatML format to Gemini's format
    gemini_msgs = []
    for msg in messages:
        if msg["role"] == "user":
            gemini_msgs.append({"role": "user", "parts": [{"text": msg["content"]}]})
        elif msg["role"] == "system":
            # Treat system messages as user messages for Gemini
            gemini_msgs.append({"role": "user", "parts": [{"text": msg["content"]}]})
        elif msg["role"] == "assistant":
            gemini_msgs.append({"role": "model", "parts": [{"text": msg["content"]}]})
    payload = {
        "contents": gemini_msgs,
        "generationConfig": {
            "temperature": kw.get("temperature", 0.0),
            "thinkingConfig": {
                "thinkingBudget": 128 # Minimum thinking budget for Gemini-2.5-pro.
            }
        }
    }
    model_name = model or _settings.google_model
    endpoint = f"models/{model_name}:generateContent"
    
    print(f"Google API request - Model: {model_name}, Endpoint: {endpoint}")
    print(f"Payload: {payload}")
    
    for attempt in range(MAX_RETRIES):
        try:
            async with _google_client() as c:
                r = await c.post(endpoint, json=payload)
                r.raise_for_status()
                # Gemini API doesn't provide token usage information
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        except httpx.HTTPStatusError as e:
            # Retry on rate limits (429) or server errors (5xx)
            if e.response.status_code in [429, 500, 502, 503, 504] and attempt < MAX_RETRIES - 1:
                retry_delay = RETRY_DELAY * (attempt + 1)  # Exponential backoff
                logger.warning(f"Google API error {e.response.status_code}, retrying in {retry_delay}s ({attempt+1}/{MAX_RETRIES})...")
                await asyncio.sleep(retry_delay)
                continue
            logger.error(f"Google API error: {e}. Response: {e.response.text if hasattr(e, 'response') else 'No response'}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Google API: {e}")
            raise

async def _openai_chat_completion(messages: list[dict], model: str | None = None, **kw) -> str:
    """
    Call OpenAI's GPT API with retry logic.
    
    Uses ChatML format directly as supported by OpenAI API. Implements exponential
    backoff for rate limits and server errors.
    
    Args:
        messages: List of message dictionaries in ChatML format
        model: Specific GPT model to use (e.g., "gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18")
        **kw: Additional API parameters
        
    Returns:
        Generated text response from the model
        
    Raises:
        httpx.HTTPStatusError: On non-retryable HTTP errors
        Exception: On other errors
    """
    payload = {
        "model": model or _settings.openai_model,
        "messages": messages,
        "temperature": kw.get("temperature", 0),
        "max_tokens": kw.get("max_tokens", 1024),
    }
    
    print(f"OpenAI API request - Model: {payload['model']}")
    print(f"Messages: {messages}")
    
    for attempt in range(MAX_RETRIES):
        try:
            async with _openai_client() as c:
                r = await c.post("/chat/completions", json=payload)
                r.raise_for_status()
                response_data = r.json()
                
                # Log token usage
                if "usage" in response_data:
                    usage = response_data["usage"]
                    print(f"Input tokens: {usage.get('prompt_tokens', 0)}")
                    print(f"Output tokens: {usage.get('completion_tokens', 0)}")
                    print(f"Total tokens: {usage.get('total_tokens', 0)}")
                
                return response_data["choices"][0]["message"]["content"]
                
        except httpx.HTTPStatusError as e:
            # Retry on rate limits (429) or server errors (5xx)
            if e.response.status_code in [429, 500, 502, 503, 504] and attempt < MAX_RETRIES - 1:
                retry_delay = RETRY_DELAY * (attempt + 1)  # Exponential backoff
                logger.warning(f"OpenAI API error {e.response.status_code}, retrying in {retry_delay}s ({attempt+1}/{MAX_RETRIES})...")
                await asyncio.sleep(retry_delay)
                continue
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI API: {e}")
            raise