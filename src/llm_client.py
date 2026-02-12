"""LLM client using OpenRouter as a unified gateway.

All models are accessed through OpenRouter's OpenAI-compatible API,
so only one API key and one billing account are needed.

Set OPENROUTER_API_KEY via environment variable or a .env file.
"""

from __future__ import annotations

import difflib
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
    temperature: Optional[float] = 0.0  # None = model default (for reasoning models)


# OpenRouter model IDs follow the pattern "provider/model-name".
# Browse https://openrouter.ai/models for the full catalogue.
MODELS: Dict[str, ModelConfig] = {
    # --- Primary experiment models ---
    "gpt-5-mini": ModelConfig(
        name="GPT-5 mini",
        model_id="openai/gpt-5-mini",
    ),
    "gpt-5-nano": ModelConfig(
        name="GPT-5 nano",
        model_id="openai/gpt-5-nano",
    ),
    "claude-haiku-4.5": ModelConfig(
        name="Claude Haiku 4.5",
        model_id="anthropic/claude-haiku-4.5",
    ),
    "gemini-3-flash-preview": ModelConfig(
        name="Gemini 3 Flash Preview",
        model_id="google/gemini-3-flash-preview",
    ),
    "gemini-2.5-flash-lite": ModelConfig(
        name="Gemini 2.5 Flash Lite",
        model_id="google/gemini-2.5-flash-lite",
    ),
    "kimi-k2.5": ModelConfig(
        name="Kimi K2.5",
        model_id="moonshotai/kimi-k2.5",
    ),
    "deepseek-v3.2": ModelConfig(
        name="DeepSeek V3.2",
        model_id="deepseek/deepseek-v3.2",
    ),
    "minimax-m2.1": ModelConfig(
        name="MiniMax M2.1",
        model_id="minimax/minimax-m2.1",
    ),
    "grok-4.1-fast": ModelConfig(
        name="Grok 4.1 Fast",
        model_id="x-ai/grok-4.1-fast",
    ),
    "qwen3-235b": ModelConfig(
        name="Qwen 3 235B A22B",
        model_id="qwen/qwen3-235b-a22b-2507",
    ),
    # --- SOTA frontier models ---
    "gpt-5.2": ModelConfig(
        name="GPT-5.2",
        model_id="openai/gpt-5.2",
        temperature=None,  # reasoning model; does not support temperature
    ),
    "claude-sonnet-4.5": ModelConfig(
        name="Claude Sonnet 4.5",
        model_id="anthropic/claude-sonnet-4.5",
    ),
    "gemini-3-pro-preview": ModelConfig(
        name="Gemini 3 Pro Preview",
        model_id="google/gemini-3-pro-preview",
    ),
}


def suggest_model(user_input: str) -> List[str]:
    """Suggest model keys that are close to user_input (fuzzy matching)."""
    all_keys = list(MODELS.keys())
    all_ids = [m.model_id for m in MODELS.values()]
    all_names = [m.name.lower() for m in MODELS.values()]

    # Check exact match first
    low = user_input.lower().strip()
    if low in MODELS:
        return [low]

    # Check if user typed an OpenRouter model_id directly
    for key, cfg in MODELS.items():
        if low == cfg.model_id.lower():
            return [key]

    # Fuzzy match against keys, model_ids, and display names
    candidates = []
    for key in all_keys:
        ratio = difflib.SequenceMatcher(None, low, key.lower()).ratio()
        if ratio > 0.5:
            candidates.append((ratio, key))
    for key, cfg in MODELS.items():
        ratio = difflib.SequenceMatcher(None, low, cfg.model_id.lower()).ratio()
        if ratio > 0.5:
            candidates.append((ratio, key))
        ratio = difflib.SequenceMatcher(None, low, cfg.name.lower()).ratio()
        if ratio > 0.5:
            candidates.append((ratio, key))

    # Deduplicate and sort by score
    seen = set()
    unique = []
    for score, key in sorted(candidates, reverse=True):
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique[:5]


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
# .env loading
# ---------------------------------------------------------------------------


def _load_dotenv():
    """Load .env file from the project directory if present."""
    # Walk up from this file to find .env
    d = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(d, ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


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
            _load_dotenv()  # load .env if present

            from openai import OpenAI

            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                raise EnvironmentError(
                    "OPENROUTER_API_KEY is not set. Either:\n"
                    "  1. export OPENROUTER_API_KEY='sk-or-...'\n"
                    "  2. Put it in a .env file in the project root.\n"
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

    def check_model(self, model_key: str) -> bool:
        """Verify a model is reachable on OpenRouter with a minimal request."""
        if model_key not in MODELS:
            return False
        config = MODELS[model_key]
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=config.model_id,
                messages=[{"role": "user", "content": "Say OK."}],
                max_tokens=16,
                temperature=0.0,
            )
            # Some models return empty content for trivial prompts;
            # a successful HTTP response (no exception) is enough.
            return response.choices is not None and len(response.choices) > 0
        except Exception as e:
            logger.warning(f"Model check failed for {model_key}: {e}")
            return False

    def generate(
        self, messages: List[Dict[str, str]], model_key: str
    ) -> LLMResponse:
        """Send messages to the specified model via OpenRouter."""
        if model_key not in MODELS:
            suggestions = suggest_model(model_key)
            hint = ""
            if suggestions:
                hint = f" Did you mean: {', '.join(suggestions)}?"
            raise ValueError(
                f"Unknown model '{model_key}'.{hint}\n"
                f"Available models: {', '.join(MODELS.keys())}"
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

        kwargs = dict(
            model=config.model_id,
            messages=messages,
            max_tokens=config.max_tokens,
        )
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature

        t0 = time.time()
        response = client.chat.completions.create(**kwargs)
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
