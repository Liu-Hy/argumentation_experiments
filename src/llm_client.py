"""LLM client using OpenRouter as a unified gateway.

All models are accessed through OpenRouter's OpenAI-compatible API,
so only one API key and one billing account are needed.

Set the OPENROUTER_API_KEY environment variable before use.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration for a specific model on OpenRouter."""

    name: str  # display name
    model_id: str  # OpenRouter model identifier (provider/model)
    max_tokens: int = 4096
    temperature: float = 0.0


# OpenRouter model IDs follow the pattern "provider/model-name".
# Browse https://openrouter.ai/models for the full catalogue.
MODELS: Dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        model_id="openai/gpt-4o",
    ),
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o-mini",
        model_id="openai/gpt-4o-mini",
    ),
    "o3-mini": ModelConfig(
        name="o3-mini",
        model_id="openai/o3-mini",
    ),
    "claude-3.5-sonnet": ModelConfig(
        name="Claude 3.5 Sonnet",
        model_id="anthropic/claude-3.5-sonnet",
    ),
    "deepseek-v3": ModelConfig(
        name="DeepSeek-V3",
        model_id="deepseek/deepseek-chat",
    ),
    "deepseek-r1": ModelConfig(
        name="DeepSeek-R1",
        model_id="deepseek/deepseek-r1",
    ),
}


# ---------------------------------------------------------------------------
# Response container
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Container for an LLM API response."""

    content: str = ""
    model: str = ""
    usage: Dict = field(default_factory=dict)
    latency_s: float = 0.0
    raw: Optional[dict] = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 15]  # seconds


class LLMClient:
    """OpenRouter-based LLM client.

    All models are called through a single OpenAI-compatible endpoint.
    """

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                raise EnvironmentError(
                    "OPENROUTER_API_KEY environment variable is not set. "
                    "Get your key at https://openrouter.ai/keys"
                )
            self._client = OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=api_key,
                default_headers={
                    "HTTP-Referer": "https://github.com/baf-extraction",
                    "X-Title": "BAF Extraction Experiments",
                },
            )
        return self._client

    def generate(
        self, messages: List[Dict[str, str]], model_key: str
    ) -> LLMResponse:
        """Send messages to the specified model via OpenRouter."""
        if model_key not in MODELS:
            raise ValueError(
                f"Unknown model '{model_key}'. Available: {list(MODELS.keys())}"
            )
        config = MODELS[model_key]

        for attempt in range(MAX_RETRIES):
            try:
                return self._call(messages, config)
            except Exception as e:
                wait = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{MAX_RETRIES}): "
                    f"{e}. Retrying in {wait}s..."
                )
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"All retries exhausted for {model_key}")
                    return LLMResponse(content="", model=config.model_id)
                time.sleep(wait)

        return LLMResponse(content="", model=config.model_id)

    def _call(
        self, messages: List[Dict[str, str]], config: ModelConfig
    ) -> LLMResponse:
        client = self._get_client()

        t0 = time.time()
        response = client.chat.completions.create(
            model=config.model_id,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        latency = time.time() - t0

        content = response.choices[0].message.content or ""
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            model=config.model_id,
            usage=usage,
            latency_s=latency,
            raw=response.model_dump()
            if hasattr(response, "model_dump")
            else None,
        )
